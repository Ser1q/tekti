"""Deterministic fact extraction and cross-document consistency comparisons."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Fact,
    RawFinding,
    Role,
    Severity,
)
from tekti.rules.common import (
    DocVerifier,
    FactCollector,
    Table,
    clean_heading,
    endpoint_scope,
    is_blank_cell,
    is_boilerplate_section,
    split_sentences_with_lines,
    strip_markdown,
)

# --------------------------------------------------------------------------- 1. monetary units

_TENGE_RE = re.compile(
    r"(?:₸|\bтенге\b|\bв\s+KZT\b|\b(?:сумма|порог|лимит)\b[^.\n]{0,25}\bKZT\b)",
    re.IGNORECASE,
)
_TIYN_RE = re.compile(r"\bтиын(?:а|ов|е|ам|ами|ах)?\b", re.IGNORECASE)
_RUBLE_RE = re.compile(r"(?:₽|\bрубл(?:ь|я|ей|ю|ям|ях)?\b|\bруб\.)", re.IGNORECASE)
_KOPECK_RE = re.compile(r"\bкопе(?:йка|йки|ек|йкам|йках)\b", re.IGNORECASE)

_CONVERSION_RULE_RE = re.compile(
    r"(?:100\s*(?:тиын|копе)|1\s*(?:тенге|рубл[ья]?)\s*=\s*100|"
    r"(?:делит(?:ь|ся)|умнож(?:ать|ается|ить)|пересч[её]т|конверт\w+)[^.\n]{0,40}(?:100|тиын|тенге|копе))",
    re.IGNORECASE,
)


def _check_monetary_units(doc: Document, verifier: DocVerifier, collector: FactCollector) -> list[RawFinding]:
    unit_hits: dict[str, list[Block]] = {"tenge": [], "tiyn": [], "ruble": [], "kopeck": []}
    has_conversion = False

    for block in doc.blocks:
        if block.kind in (BlockKind.HEADING, BlockKind.CODE):
            continue
        if is_boilerplate_section(block.section_path):
            continue
        cleaned = strip_markdown(block.text)
        if _CONVERSION_RULE_RE.search(cleaned):
            has_conversion = True

        for unit_name, pattern in (
            ("tenge", _TENGE_RE),
            ("tiyn", _TIYN_RE),
            ("ruble", _RUBLE_RE),
            ("kopeck", _KOPECK_RE),
        ):
            if pattern.search(cleaned):
                collector.add(
                    name="currency.unit",
                    value=unit_name,
                    unit=unit_name,
                    quote=block.text,
                    block_id=block.id,
                )
                unit_hits[unit_name].append(block)

    findings: list[RawFinding] = []
    pairs = (("tiyn", "tenge", "тиынах", "тенге"), ("kopeck", "ruble", "копейках", "рублях"))
    for minor_key, major_key, minor_ru, major_ru in pairs:
        minor_blocks = unit_hits[minor_key]
        major_blocks = unit_hits[major_key]
        if not (minor_blocks and major_blocks) or has_conversion:
            continue

        # Prefer a block that explicitly mixes both units (e.g., data storage rule).
        same_ids = {b.id for b in minor_blocks} & {b.id for b in major_blocks}
        if same_ids:
            mixed = next(b for b in minor_blocks if b.id in same_ids)
            ev_pairs = [(mixed.text, mixed.id)]
        else:
            ev_pairs = [
                (minor_blocks[0].text, minor_blocks[0].id),
                (major_blocks[0].text, major_blocks[0].id),
            ]

        finding = verifier.make_finding(
            criterion=Criterion.SET_CONSISTENT,
            severity=Severity.MAJOR,
            title=f"Несогласованные денежные единицы: значения указаны в {minor_ru} и в {major_ru} без правила пересчёта",
            explanation=(
                f"В документе суммы указаны одновременно в {minor_ru} и в {major_ru}, но не задано правило их пересчёта "
                "при сравнении и обмене данными. Разработчику придётся самостоятельно догадываться о приведении единиц."
            ),
            evidence_pairs=ev_pairs,
            question=f"Как именно приводить значения в {minor_ru} и в {major_ru} при сравнении и передаче (например, 100 к 1), или нужно унифицировать единицу?",
            addressee=Role.DEVELOPER,
        )
        if finding is not None:
            findings.append(finding)
    return findings


# --------------------------------------------------------------------------- 2. numeric ranges

_TEXT_RANGE_RE = re.compile(
    r"\bот\s+(\d[\d\s\u00a0]*\d|\d)\s+до\s+(\d[\d\s\u00a0]*\d|\d)(?:\s*(₸|тенге|тиын|%|руб\.?|₽))?",
    re.IGNORECASE,
)
_CELL_RANGE_RE = re.compile(r"^(\d[\d\s\u00a0]*\d|\d)\s*[–—\-]\s*(\d[\d\s\u00a0]*\d|\d)$")

_CONCEPT_MAP: tuple[tuple[str, str], ...] = (
    ("порог", "threshold"),
    ("лимит", "threshold"),
    ("сумма", "amount"),
    ("срок", "duration"),
    ("возраст", "age"),
    ("размер", "size"),
)

_CONCEPT_LABEL_RU: dict[str, str] = {
    "threshold": "порог",
    "amount": "сумма",
    "duration": "срок",
    "age": "возраст",
    "size": "размер",
}


@dataclass(frozen=True)
class _RangeEntry:
    concept: str
    low: int
    high: int
    unit: str | None
    quote: str
    block_id: str


def _parse_int(raw: str) -> int:
    return int(re.sub(r"[\s\u00a0]+", "", raw))


def _infer_concept(text_before: str) -> str | None:
    low = text_before.lower()
    for ru, key in _CONCEPT_MAP:
        if ru in low:
            return key
    return None


def _unit_key(raw: str | None) -> str | None:
    if not raw:
        return None
    low = raw.lower().strip()
    if low in ("₸", "тенге"):
        return "tenge"
    if low.startswith("тиын"):
        return "tiyn"
    if low in ("₽", "руб", "руб."):
        return "ruble"
    return low


def _extract_and_compare_ranges(
    doc: Document,
    tables: list[Table],
    verifier: DocVerifier,
    collector: FactCollector,
) -> list[RawFinding]:
    entries: list[_RangeEntry] = []

    # 1. Ranges inside parameter tables (tagged by parameter name + column role).
    table_row_ids: set[str] = set()
    for table in tables:
        for row_block, _ in table.rows:
            table_row_ids.add(row_block.id)
        if is_boilerplate_section(table.section_path):
            continue
        headers = [strip_markdown(h).lower() for h in table.header_cells]
        param_idx = _find_col(headers, ("parameter", "параметр", "field", "поле", "name", "имя"))
        if param_idx is None:
            continue
        desc_idx = _find_col(headers, ("description", "описание"))
        for row_block, cells in table.rows:
            if param_idx >= len(cells):
                continue
            pname = strip_markdown(cells[param_idx]).strip()
            if not pname:
                continue
            desc = strip_markdown(cells[desc_idx]) if desc_idx is not None and desc_idx < len(cells) else ""
            for col_idx, cell in enumerate(cells):
                if col_idx == param_idx:
                    continue
                col_name = headers[col_idx] if col_idx < len(headers) else f"col{col_idx}"
                m = _CELL_RANGE_RE.match(strip_markdown(cell))
                if not m:
                    continue
                low, high = _parse_int(m.group(1)), _parse_int(m.group(2))
                # Map threshold/limit descriptions to a shared concept when in a Logic/Value column.
                concept = f"param.{pname}.{col_name}"
                inferred = _infer_concept(desc)
                if inferred == "threshold" and col_name in ("logic", "логика", "значение", "диапазон", "range"):
                    concept = "threshold"
                unit = _unit_key("тенге" if "тенге" in desc.lower() or "₸" in desc else None)
                collector.add(
                    name=f"range.{concept}.min",
                    value=str(low),
                    unit=unit,
                    quote=row_block.text,
                    block_id=row_block.id,
                )
                collector.add(
                    name=f"range.{concept}.max",
                    value=str(high),
                    unit=unit,
                    quote=row_block.text,
                    block_id=row_block.id,
                )
                entries.append(
                    _RangeEntry(
                        concept=concept,
                        low=low,
                        high=high,
                        unit=unit,
                        quote=row_block.text,
                        block_id=row_block.id,
                    )
                )

    # 2. Ranges inside prose / list items (checked per sentence so concepts stay scoped).
    for block in doc.blocks:
        if block.id in table_row_ids or block.kind in (BlockKind.HEADING, BlockKind.CODE):
            continue
        if is_boilerplate_section(block.section_path):
            continue
        for sentence in split_sentences_with_lines(block.text):
            sent_clean = strip_markdown(sentence)
            for m in _TEXT_RANGE_RE.finditer(sent_clean):
                low, high = _parse_int(m.group(1)), _parse_int(m.group(2))
                unit = _unit_key(m.group(3))
                concept = _infer_concept(sent_clean[: m.start()]) or "general"
                collector.add(
                    name=f"range.{concept}.min",
                    value=str(low),
                    unit=unit,
                    quote=sentence,
                    block_id=block.id,
                )
                collector.add(
                    name=f"range.{concept}.max",
                    value=str(high),
                    unit=unit,
                    quote=sentence,
                    block_id=block.id,
                )
                entries.append(
                    _RangeEntry(
                        concept=concept,
                        low=low,
                        high=high,
                        unit=unit,
                        quote=sentence,
                        block_id=block.id,
                    )
                )

    # Compare entries sharing the same concrete concept (skip generic "general").
    by_concept: dict[str, list[_RangeEntry]] = {}
    for entry in entries:
        if entry.concept == "general":
            continue
        by_concept.setdefault(entry.concept, []).append(entry)

    findings: list[RawFinding] = []
    for concept, group in by_concept.items():
        distinct_bounds = {(e.low, e.high) for e in group}
        if len(distinct_bounds) < 2:
            continue
        first = group[0]
        other = next(e for e in group[1:] if (e.low, e.high) != (first.low, first.high))
        concept_ru = _CONCEPT_LABEL_RU.get(concept, concept)
        finding = verifier.make_finding(
            criterion=Criterion.SET_CONSISTENT,
            severity=Severity.CRITICAL,
            title=f"Противоречивые числовые границы диапазона ({concept_ru})",
            explanation=(
                f"В одном месте документа задан диапазон {first.low}–{first.high}, "
                f"а в другом для того же параметра указано {other.low}–{other.high}. "
                "Оба ограничения не могут выполняться одновременно."
            ),
            evidence_pairs=[(first.quote, first.block_id), (other.quote, other.block_id)],
            question=f"Какой диапазон значений является верным: {first.low}–{first.high} или {other.low}–{other.high}?",
            addressee=Role.PRODUCT_OWNER,
        )
        if finding is not None:
            findings.append(finding)
    return findings


# --------------------------------------------------------------------------- 3. API request vs response types

_TYPE_ALIASES = {
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "integer": "integer",
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
}


def _find_col(headers: list[str], candidates: tuple[str, ...]) -> int | None:
    for idx, h in enumerate(headers):
        if any(cand == h or cand in h for cand in candidates):
            return idx
    return None


def _normalize_type(raw: str) -> str:
    cleaned = strip_markdown(raw).lower().strip()
    return _TYPE_ALIASES.get(cleaned, cleaned)


def _direction_from_section(section_path: list[str]) -> str | None:
    joined = " ".join(clean_heading(p).lower() for p in section_path)
    if any(kw in joined for kw in ("запрос", "request", "query", "path", "входн")):
        return "request"
    if any(kw in joined for kw in ("ответ", "response", "выходн")):
        return "response"
    return None


@dataclass(frozen=True)
class _ParamSpec:
    scope: tuple[str, ...]
    direction: str
    name: str
    raw_type: str
    norm_type: str
    db_field: str | None
    block: Block


def _check_api_param_types(
    tables: list[Table],
    verifier: DocVerifier,
    collector: FactCollector,
) -> list[RawFinding]:
    specs: list[_ParamSpec] = []

    for table in tables:
        headers = [strip_markdown(h).lower() for h in table.header_cells]
        param_idx = _find_col(headers, ("parameter", "параметр", "field", "поле", "name", "имя"))
        type_idx = _find_col(headers, ("format", "type", "тип", "формат"))
        if param_idx is None or type_idx is None:
            continue
        direction = _direction_from_section(table.section_path)
        if direction is None:
            continue
        db_idx = _find_col(headers, ("db field", "db_field", "поле бд", "колонка бд", "столбец бд"))
        scope = endpoint_scope(table.section_path)

        for row_block, cells in table.rows:
            if max(param_idx, type_idx) >= len(cells):
                continue
            pname = strip_markdown(cells[param_idx]).strip()
            raw_type = strip_markdown(cells[type_idx]).strip()
            if not pname or is_blank_cell(raw_type):
                continue
            db_field: str | None = None
            if db_idx is not None and db_idx < len(cells) and not is_blank_cell(cells[db_idx]):
                db_field = strip_markdown(cells[db_idx]).strip()

            collector.add(
                name=f"api.{pname}.{direction}_type",
                value=raw_type,
                quote=row_block.text,
                block_id=row_block.id,
            )
            if db_field:
                collector.add(
                    name=f"api.{pname}.{direction}_db_field",
                    value=db_field,
                    quote=row_block.text,
                    block_id=row_block.id,
                )
            specs.append(
                _ParamSpec(
                    scope=scope,
                    direction=direction,
                    name=pname,
                    raw_type=raw_type,
                    norm_type=_normalize_type(raw_type),
                    db_field=db_field,
                    block=row_block,
                )
            )

    findings: list[RawFinding] = []
    seen_pairs: set[tuple[str, str]] = set()
    reqs = [s for s in specs if s.direction == "request"]
    resps = [s for s in specs if s.direction == "response"]

    for req_spec in reqs:
        for resp_spec in resps:
            if req_spec.scope != resp_spec.scope:
                continue
            same_name = req_spec.name.lower() == resp_spec.name.lower()
            same_db = (
                req_spec.db_field is not None
                and resp_spec.db_field is not None
                and req_spec.db_field.lower() == resp_spec.db_field.lower()
            )
            if not (same_name or same_db):
                continue
            if req_spec.norm_type == resp_spec.norm_type:
                continue
            pair_key = (req_spec.block.id, resp_spec.block.id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            label = req_spec.name if same_name else f"{req_spec.name} / {resp_spec.name} ({req_spec.db_field})"
            finding = verifier.make_finding(
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.MAJOR,
                title=f"Несогласованный тип параметра «{label}» между запросом ({req_spec.raw_type}) и ответом ({resp_spec.raw_type})",
                explanation=(
                    f"Поле «{label}» описано в таблице параметров запроса как `{req_spec.raw_type}`, "
                    f"а в таблице параметров ответа — как `{resp_spec.raw_type}`"
                    + (f" (при том же столбце БД `{req_spec.db_field}`)." if req_spec.db_field else ".")
                    + " Различие типов заставит клиента или сервер приводить типы вслепую."
                ),
                evidence_pairs=[(req_spec.block.text, req_spec.block.id), (resp_spec.block.text, resp_spec.block.id)],
                question=f"Какой тип должен использоваться для поля «{label}» в запросе и ответе: `{req_spec.raw_type}` или `{resp_spec.raw_type}`?",
                addressee=Role.ARCHITECT,
            )
            if finding is not None:
                findings.append(finding)
    return findings


# --------------------------------------------------------------------------- 4. HTTP methods & Request body

_HTTP_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")
_URL_MUTATION_RE = re.compile(
    r"(?<![a-z])(?:save|create|update|delete|remove|set|edit|add)(?=[A-Z0-9/_-]|$)"
)
_PROSE_MUTATION_RE = re.compile(
    r"(?:замени(?:ть|ет)\s+настройк|измени(?:ть|ет)\s+(?:настройк|запис|данн|состояни)|"
    r"сохранени[ея]\s+(?:порога|настройк|данн|запис)|результат\s+сохранения|"
    r"созда(?:ть|ет|ёт)\s+(?:запис|объект|ресурс)|удал(?:ить|яет)\s+(?:запис|объект|ресурс))",
    re.IGNORECASE,
)
_CONTENT_TYPE_RE = re.compile(r"content-type\s*:\s*`?([a-z0-9.+_-]+/[a-z0-9.+_-]+)", re.IGNORECASE)
_NO_BODY_RE = re.compile(
    r"(?:тело\s+(?:запроса\s+)?не\s+используется|без\s+тела\s+запроса|тело\s+отсутствует|no\s+request\s+body)",
    re.IGNORECASE,
)


def _check_http_methods_and_body(
    doc: Document,
    tables: list[Table],
    verifier: DocVerifier,
    collector: FactCollector,
) -> list[RawFinding]:
    findings: list[RawFinding] = []
    methods_by_scope: dict[tuple[str, ...], tuple[str, Block]] = {}
    url_rows_by_scope: dict[tuple[str, ...], Block] = {}

    for table in tables:
        if len(table.header_cells) != 2:
            continue
        scope = endpoint_scope(table.section_path)
        for row_block, cells in table.rows:
            key = strip_markdown(cells[0]).lower()
            val = strip_markdown(cells[1])
            if key in ("rest type", "http method", "http-метод", "метод", "method"):
                m = _HTTP_METHOD_RE.search(val.upper())
                if m:
                    method = m.group(1)
                    collector.add(
                        name="api.http_method",
                        value=method,
                        quote=row_block.text,
                        block_id=row_block.id,
                    )
                    methods_by_scope[scope] = (method, row_block)
            elif key in ("request url", "url", "endpoint"):
                url_rows_by_scope[scope] = row_block

    # Check GET vs state mutation per endpoint scope.
    for scope, (method, method_block) in methods_by_scope.items():
        if method != "GET":
            continue
        mutation_block: Block | None = None
        for block in doc.blocks:
            if endpoint_scope(block.section_path) != scope:
                continue
            if block.kind in (BlockKind.HEADING, BlockKind.CODE):
                continue
            if _PROSE_MUTATION_RE.search(strip_markdown(block.text)):
                mutation_block = block
                # Prefer method logic / special rules over attribute description if available.
                sec_low = " ".join(clean_heading(p).lower() for p in block.section_path)
                if "логика" in sec_low or "правила" in sec_low:
                    break
        if mutation_block is None and scope in url_rows_by_scope:
            url_block = url_rows_by_scope[scope]
            if _URL_MUTATION_RE.search(strip_markdown(url_block.text)):
                mutation_block = url_block

        if mutation_block is not None:
            finding = verifier.make_finding(
                criterion=Criterion.CORRECT,
                severity=Severity.CRITICAL,
                title="Метод GET используется для операции изменения состояния",
                explanation=(
                    "В характеристиках метода указан HTTP-метод `GET`, однако сам метод выполняет сохранение/изменение "
                    "настройки. Согласно семантике HTTP, `GET` должен быть безопасным и не изменять состояние, а передача "
                    "изменяющих параметров в query-строке ведёт к кэшированию, логированию и рискам повторного вызова."
                ),
                evidence_pairs=[(method_block.text, method_block.id), (mutation_block.text, mutation_block.id)],
                question="Какой HTTP-метод должен использоваться для сохранения настройки (`POST`, `PUT` или `PATCH`), и как обновить контракт?",
                addressee=Role.ARCHITECT,
            )
            if finding is not None:
                findings.append(finding)

    # Check Content-Type declared alongside "Тело не используется" in Request body sections.
    body_sections: dict[tuple[str, ...], list[Block]] = {}
    for block in doc.blocks:
        if any("request body" in clean_heading(p).lower() or "тело запроса" in clean_heading(p).lower() for p in block.section_path):
            body_sections.setdefault(endpoint_scope(block.section_path), []).append(block)

    for scope_blocks in body_sections.values():
        ct_block: Block | None = None
        no_body_block: Block | None = None
        for block in scope_blocks:
            cleaned = strip_markdown(block.text)
            if ct_block is None and _CONTENT_TYPE_RE.search(cleaned):
                ct_block = block
            if no_body_block is None and _NO_BODY_RE.search(cleaned):
                no_body_block = block
        if ct_block is not None and no_body_block is not None:
            ev_pairs = (
                [(ct_block.text, ct_block.id)]
                if ct_block.id == no_body_block.id
                else [(ct_block.text, ct_block.id), (no_body_block.text, no_body_block.id)]
            )
            finding = verifier.make_finding(
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.MINOR,
                title="Указан Content-Type для запроса с неиспользуемым телом",
                explanation=(
                    "В разделе описания тела запроса одновременно объявлен заголовок `Content-Type` "
                    "и указано, что тело запроса не используется."
                ),
                evidence_pairs=ev_pairs,
                question="Используется ли тело запроса и заголовок `Content-Type`, или оба пункта нужно привести к единому виду?",
                addressee=Role.ARCHITECT,
            )
            if finding is not None:
                findings.append(finding)

    return findings


# --------------------------------------------------------------------------- 5. HTTP status codes

_STATUS_DEF_LINE_RE = re.compile(r"^(?:[*+-]\s+)?\*{0,2}([1-5]\d{2})\.?\*{0,2}\s*[—–\-.:]?\s+(.+)$")
_STATUS_HEADING_RE = re.compile(r"^([1-5]\d{2})(?:\b|[.\s—–\-:])(.*)$")
_HTTP_CODE_RE = re.compile(
    r"(?<![\d.])(20[0-8]|226|30[0-8]|40\d|41\d|42[0-9]|431|451|50\d|51[01])(?![\d.])"
    r"(?!\s*(?:₸|тенге|тиын|bit|бит|мс|ms|сек|мин|час|дн|знак|%))"
)


@dataclass(frozen=True)
class _StatusMention:
    code: str
    quote: str
    block_id: str
    parent_section_code: str | None


def _is_statuses_section(section_path: list[str]) -> bool:
    return any(
        any(kw in clean_heading(p).lower() for kw in ("статус", "коды ответа", "коды состояний", "statuses"))
        for p in section_path
    )


def _is_response_section(section_path: list[str]) -> bool:
    return any(
        any(kw in clean_heading(p).lower() for kw in ("ответ", "response"))
        for p in section_path
    )


def _parent_status_code(section_path: list[str]) -> str | None:
    if not section_path:
        return None
    last = clean_heading(section_path[-1])
    m = _STATUS_HEADING_RE.match(last)
    return m.group(1) if m else None


def _check_status_codes(
    doc: Document,
    tables: list[Table],
    verifier: DocVerifier,
    collector: FactCollector,
) -> list[RawFinding]:
    listed_by_scope: dict[tuple[str, ...], dict[str, tuple[str, str]]] = {}
    sections_by_scope: dict[tuple[str, ...], dict[str, tuple[Block, str]]] = {}
    mentions_by_scope: dict[tuple[str, ...], list[_StatusMention]] = {}

    table_block_ids = {t.header_block.id for t in tables} | {b.id for t in tables for b, _ in t.rows}

    # 1. Collect status definitions in status tables if present.
    for table in tables:
        if not _is_statuses_section(table.section_path):
            continue
        scope = endpoint_scope(table.section_path)
        for row_block, cells in table.rows:
            if len(cells) >= 2:
                code_cell = strip_markdown(cells[0]).strip()
                if re.fullmatch(r"[1-5]\d{2}", code_cell) and not is_blank_cell(cells[1]):
                    listed_by_scope.setdefault(scope, {}).setdefault(code_cell, (row_block.text, row_block.id))
                    collector.add(
                        name="api.status.listed",
                        value=code_cell,
                        quote=row_block.text,
                        block_id=row_block.id,
                    )

    # 2. Collect status definitions, section headings, and prose mentions across blocks.
    for block in doc.blocks:
        if block.kind == BlockKind.CODE or block.id in table_block_ids:
            continue
        scope = endpoint_scope(block.section_path)

        if block.kind == BlockKind.HEADING and _is_statuses_section(block.section_path):
            clean = clean_heading(block.text)
            m = _STATUS_HEADING_RE.match(clean)
            if m:
                code, title_rest = m.group(1), m.group(2).strip()
                sections_by_scope.setdefault(scope, {}).setdefault(code, (block, title_rest))
                collector.add(
                    name="api.status.section",
                    value=code,
                    quote=block.text,
                    block_id=block.id,
                )
            continue

        if _is_statuses_section(block.section_path):
            parent_code = _parent_status_code(block.section_path)
            for raw_line in block.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                def_m = _STATUS_DEF_LINE_RE.match(line)
                if def_m and re.search(r"[А-ЯЁа-яёA-Za-z]{3,}", def_m.group(2)):
                    code = def_m.group(1)
                    listed_by_scope.setdefault(scope, {}).setdefault(code, (line, block.id))
                    collector.add(
                        name="api.status.listed",
                        value=code,
                        quote=line,
                        block_id=block.id,
                    )
                    continue

                # Non-definition prose line inside Статусы or a status subsection.
                for sentence in split_sentences_with_lines(line):
                    sent_clean = strip_markdown(sentence)
                    for cm in _HTTP_CODE_RE.finditer(sent_clean):
                        code = cm.group(1)
                        mentions_by_scope.setdefault(scope, []).append(
                            _StatusMention(
                                code=code,
                                quote=sentence,
                                block_id=block.id,
                                parent_section_code=parent_code,
                            )
                        )
                        collector.add(
                            name="api.status.mentioned",
                            value=code,
                            quote=sentence,
                            block_id=block.id,
                        )
        elif _is_response_section(block.section_path):
            for sentence in split_sentences_with_lines(block.text):
                sent_clean = strip_markdown(sentence)
                for cm in _HTTP_CODE_RE.finditer(sent_clean):
                    code = cm.group(1)
                    mentions_by_scope.setdefault(scope, []).append(
                        _StatusMention(
                            code=code,
                            quote=sentence,
                            block_id=block.id,
                            parent_section_code=None,
                        )
                    )
                    collector.add(
                        name="api.status.mentioned",
                        value=code,
                        quote=sentence,
                        block_id=block.id,
                    )

    # 3. Compare status sets per endpoint scope.
    findings: list[RawFinding] = []
    all_scopes = set(listed_by_scope) | set(sections_by_scope) | set(mentions_by_scope)
    for scope in sorted(all_scopes):
        listed = listed_by_scope.get(scope, {})
        sections = sections_by_scope.get(scope, {})
        mentions = mentions_by_scope.get(scope, [])

        # A. Status used inside a status subsection (or anywhere in Статусы) but absent from listed.
        reported_unlisted: set[str] = set()
        for mention in mentions:
            if mention.code in listed:
                continue
            if mention.parent_section_code is None:
                continue
            if mention.code in reported_unlisted:
                continue
            reported_unlisted.add(mention.code)

            ev_pairs = [(mention.quote, mention.block_id)]
            parent_code = mention.parent_section_code
            if parent_code in listed:
                ev_pairs.append(listed[parent_code])
                expl = (
                    f"В описании статуса {parent_code} указано, что возвращается статус {mention.code}, "
                    f"однако статус {mention.code} отсутствует в списке кодов ответа, "
                    f"а перечисленный статус {parent_code} фактически не используется."
                )
            else:
                expl = (
                    f"Статус {mention.code} используется в описании раздела статусов, "
                    "но отсутствует в перечне кодов ответа метода."
                )
            finding = verifier.make_finding(
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.CRITICAL,
                title=f"Статус {mention.code} возвращается в описании, но отсутствует в списке статусов",
                explanation=expl,
                evidence_pairs=ev_pairs,
                question=(
                    f"Какой статус должен возвращаться на самом деле — {mention.code} или {parent_code}, "
                    "и нужно ли добавить его в перечень статусов?"
                    if parent_code
                    else f"Нужно ли добавить статус {mention.code} в перечень статусов и описать его отдельно?"
                ),
                addressee=Role.ARCHITECT,
            )
            if finding is not None:
                findings.append(finding)

        # B. Status mentioned in response rules without a definition or section heading.
        reported_undescribed: set[str] = set()
        for mention in mentions:
            if mention.parent_section_code is not None:
                continue
            code = mention.code
            if code in listed or code in sections or code in reported_unlisted or code in reported_undescribed:
                continue
            reported_undescribed.add(code)
            ev_pairs = [(mention.quote, mention.block_id)]
            extra_note = ""
            # If 201 is undocumented and the 200 heading carries "Created", cite that heading too.
            if code == "201" and "200" in sections and "created" in sections["200"][1].lower():
                h200_block, _ = sections["200"]
                ev_pairs.append((h200_block.text, h200_block.id))
                extra_note = f" При этом заголовок раздела 200 («{clean_heading(h200_block.text)}») содержит «Created», что соответствует 201."

            finding = verifier.make_finding(
                criterion=Criterion.SET_CONSISTENT,
                severity=Severity.MAJOR,
                title=f"Статус {code} упомянут в правилах ответа, но не описан в разделе статусов",
                explanation=(
                    f"В разделе ответа упоминается код {code}, однако он отсутствует и в перечне статусов, "
                    f"и в подразделах с примерами ответов.{extra_note}"
                ),
                evidence_pairs=ev_pairs,
                question=f"Может ли метод возвращать статус {code}; если да, то при каких условиях и с каким телом ответа?",
                addressee=Role.ARCHITECT,
            )
            if finding is not None:
                findings.append(finding)

        # C. Listed status missing a dedicated subsection (when other statuses have subsections).
        if listed and sections:
            for code, (def_quote, def_bid) in listed.items():
                if code not in sections:
                    finding = verifier.make_finding(
                        criterion=Criterion.SET_CONSISTENT,
                        severity=Severity.MINOR,
                        title=f"Статус {code} указан в списке, но не имеет отдельного раздела с примером",
                        explanation=(
                            f"Код ответа {code} перечислен в перечне статусов, однако в отличие от остальных "
                            "статусов для него не оформлен отдельный подраздел и пример ответа."
                        ),
                        evidence_pairs=[(def_quote, def_bid)],
                        question=f"Какое тело ответа должно возвращаться для статуса {code}?",
                        addressee=Role.ARCHITECT,
                    )
                    if finding is not None:
                        findings.append(finding)

            for code, (sec_block, _) in sections.items():
                if code not in listed and code not in reported_unlisted:
                    finding = verifier.make_finding(
                        criterion=Criterion.SET_CONSISTENT,
                        severity=Severity.MAJOR,
                        title=f"Подраздел статуса {code} не отражён в общем перечне статусов",
                        explanation=(
                            f"Для статуса {code} заведён отдельный подраздел, но этот код отсутствует "
                            "в списке статусов метода."
                        ),
                        evidence_pairs=[(sec_block.text, sec_block.id)],
                        question=f"Должен ли статус {code} входить в основной перечень статусов метода?",
                        addressee=Role.ARCHITECT,
                    )
                    if finding is not None:
                        findings.append(finding)

    return findings


def extract_and_compare(
    doc: Document,
    tables: list[Table],
    verifier: DocVerifier,
) -> tuple[list[Fact], list[RawFinding]]:
    """Extract comparable facts across the document and run deterministic cross-checks."""
    collector = FactCollector(verifier)
    findings: list[RawFinding] = []
    findings.extend(_check_monetary_units(doc, verifier, collector))
    findings.extend(_extract_and_compare_ranges(doc, tables, verifier, collector))
    findings.extend(_check_api_param_types(tables, verifier, collector))
    findings.extend(_check_http_methods_and_body(doc, tables, verifier, collector))
    findings.extend(_check_status_codes(doc, tables, verifier, collector))
    return collector.facts, findings
