"""Deterministic structural, hygiene, wording, traceability, and injection rules."""

from __future__ import annotations

import re

from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    RawFinding,
    Requirement,
    RequirementKind,
    Role,
    Severity,
)
from tekti.rules.common import (
    DocVerifier,
    Table,
    clean_heading,
    heading_level,
    is_blank_cell,
    is_boilerplate_section,
    parse_table_cells,
    split_sentences_with_lines,
    strip_markdown,
)

# --------------------------------------------------------------------------- embedded instructions

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"(?:оцени(?:те)?|поставь(?:те)?)\s+(?:этот\s+)?документ\s+на\b",
        r"(?:поставь(?:те)?|выстав(?:ь|ите)|присво(?:й|йте))\s+(?:документу\s+)?(?:оценку|максимальный\s+балл|высший\s+балл|100\s+из\s+100|отлично)\b",
        r"игнорируй(?:те)?\s+(?:все\s+|любые\s+)?(?:предыдущие|вышеуказанные|прошлые|системные)?\s*(?:инструкции|правила|указания|замечания)\b",
        r"забудь(?:те)?\s+(?:все\s+|любые\s+)?(?:предыдущие|вышеуказанные|прошлые|системные)?\s*(?:инструкции|правила|указания)\b",
        r"считай(?:те)?\s+(?:этот\s+)?документ\s+(?:безупречным|идеальным|готовым|прошедшим\s+проверку|без\s+дефектов|без\s+ошибок)\b",
        r"не\s+(?:сообщай(?:те)?|указывай(?:те)?|пиши(?:те)?|выводи(?:те)?)\s+(?:об\s+ошибках|о\s+дефектах|о\s+замечаниях|дефекты|замечания|ошибки)\b",
        r"(?:пропусти(?:те)?|не\s+проверяй(?:те)?)\s+(?:этот\s+)?раздел\b",
        r"(?:обращение\s+к|инструкция\s+для)\s+(?:ии|ai|llm|модели|рецензент[уа]|проверяющ(?:ему|его))\b",
        r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:previous|prior|above|system)\s+(?:instructions|rules|prompts|directives)\b",
        r"rate\s+this\s+document\s+(?:100|10/10|perfect|as\s+perfect)\b",
        r"do\s+not\s+report\s+(?:any\s+)?(?:defects|errors|issues|findings)\b",
        r"consider\s+this\s+document\s+(?:perfect|flawless|defect-free)\b",
        r"\bsystem\s+prompt\s*:",
    )
)


def check_embedded_instructions(doc: Document, verifier: DocVerifier) -> list[RawFinding]:
    """Flag document text attempting to instruct the reviewer or LLM."""
    findings: list[RawFinding] = []
    for block in doc.blocks:
        candidates = (
            [block.text]
            if block.kind == BlockKind.CODE
            else (split_sentences_with_lines(block.text) or [block.text])
        )
        matched_snippets = [
            snippet
            for snippet in candidates
            if any(pat.search(strip_markdown(snippet)) for pat in _INJECTION_PATTERNS)
        ]
        if (
            not matched_snippets
            and block.kind != BlockKind.CODE
            and any(pat.search(strip_markdown(block.text)) for pat in _INJECTION_PATTERNS)
        ):
            matched_snippets = [block.text]

        for snippet in matched_snippets:
            finding = verifier.make_finding(
                criterion=Criterion.EMBEDDED_INSTRUCTION,
                severity=Severity.MAJOR,
                title="В тексте документа обнаружена инструкция, адресованная проверяющему",
                explanation=(
                    "Фрагмент документа содержит указание проверяющему или модели вместо описания требований "
                    "к системе. Подобные вставки не должны влиять на ход проверки."
                ),
                evidence_pairs=[(snippet, block.id)],
                question="Почему в текст требований включено обращение к проверяющему; следует ли удалить этот фрагмент?",
                addressee=Role.ANALYST,
                requirement_ids=[],
            )
            if finding is not None:
                findings.append(finding)
    return findings


# --------------------------------------------------------------------------- empty & placeholder sections / cells

_PLACEHOLDER_BODY_RE = re.compile(
    r"^(?:"
    r"заполня(?:ет|ют)ся(?:\s+[а-яёa-z0-9_-]+){1,4}"
    r"|будет\s+(?:заполнен[оаы]?|уточн[её]н[оаы]?|дополнен[оаы]?|добавлен[оаы]?|определ[её]н[оаы]?)(?:\s+позже)?"
    r"|(?:уточнить|определить|добавить|заполнить)\s+позже"
    r"|в\s+(?:разработке|доработке)"
    r"|в\s+процессе\s+(?:наполнения|заполнения|разработки)"
    r"|раздел\s+не\s+заполнен"
    r"|не\s+заполнен[оа]?"
    r"|tbd|todo|tba|fixme|wip"
    r"|…|\.\.\.|—|–|-"
    r")\.?$",
    re.IGNORECASE,
)

_INLINE_PLACEHOLDER_RE = re.compile(
    r"(?:\b(?:TBD|TODO|TBA|FIXME)\b|(?:уточнить|определить|заполнить)\s+позже)",
    re.IGNORECASE,
)

_UNFILLED_METADATA_NOTE_RE = re.compile(
    r"(?:документ\s*\([^)]*(?:верси|дат)[^)]*\)|(?:версия|дата)\s+документа)\s+(?:в\s+источнике\s+)?не\s+заполнен[ао]?",
    re.IGNORECASE,
)

_CHECKLIST_CHOICE_RE = re.compile(
    r"(?:\bда\s*/\s*нет(?:\s*/\s*не\s+требуется)?\b|\byes\s*/\s*no(?:\s*/\s*n/?a)?\b|\[\s*\]\s*да\b)",
    re.IGNORECASE,
)

_SUBSTANTIVE_SECTION_KEYWORDS = (
    "требован",
    "тестирован",
    "приемк",
    "приёмк",
    "логика",
    "данные",
    "данным",
    "интеграц",
    "статус",
    "параметр",
    "ответ",
    "правила",
    "api",
)

_KV_HEADER_KEYS = frozenset({"поле", "атрибут", "параметр", "ключ", "field", "attribute", "property", "key"})
_KV_HEADER_VALS = frozenset({"значение", "описание", "value", "description"})

_MANDATORY_ATTR_KEYS = (
    "swagger",
    "openapi",
    "request url",
    "url",
    "endpoint",
    "rest type",
    "method",
    "authorization",
    "controller",
    "версия",
    "version",
    "дата",
    "date",
    "автор",
    "author",
    "владелец",
    "product owner",
    "business owner",
    "разработчик",
    "developer",
    "аналитик",
    "analyst",
)

_API_ATTR_KEYS = frozenset(
    {"swagger", "openapi", "request url", "url", "endpoint", "rest type", "method", "authorization", "controller"}
)

_ENV_BLANK_RE = re.compile(
    r"\b(?:Prod|Production|Stage|Staging|UAT|QA|Test|Dev)\s*:\s*(?:[—–\-]|TBD|N/A)"
    r"(?=\s*(?:$|[.,;|]|\b(?:Prod|Production|Stage|Staging|UAT|QA|Test|Dev)\b))",
    re.IGNORECASE,
)


def _is_substantive_section(section_path: list[str]) -> bool:
    joined = " ".join(clean_heading(p).lower() for p in section_path)
    return any(kw in joined for kw in _SUBSTANTIVE_SECTION_KEYWORDS)


def _is_kv_table(table: Table) -> bool:
    if len(table.header_cells) != 2:
        return False
    h0 = strip_markdown(table.header_cells[0]).lower()
    h1 = strip_markdown(table.header_cells[1]).lower()
    return h0 in _KV_HEADER_KEYS and h1 in _KV_HEADER_VALS


def _check_empty_headings(doc: Document, verifier: DocVerifier) -> list[RawFinding]:
    findings: list[RawFinding] = []
    blocks = doc.blocks
    for i, block in enumerate(blocks):
        if block.kind != BlockKind.HEADING:
            continue
        is_last = i == len(blocks) - 1
        next_same_or_outer = (
            not is_last
            and blocks[i + 1].kind == BlockKind.HEADING
            and heading_level(blocks[i + 1]) <= heading_level(block)
        )
        if not (is_last or next_same_or_outer):
            continue
        title_clean = clean_heading(block.text)
        sev = (
            Severity.CRITICAL
            if _is_substantive_section([*block.section_path, title_clean])
            else Severity.MAJOR
        )
        finding = verifier.make_finding(
            criterion=Criterion.SET_COMPLETE,
            severity=sev,
            title=f"Пустой раздел «{title_clean}»",
            explanation=(
                f"Заголовок «{title_clean}» не содержит ни текста, ни вложенных подразделов. "
                "Незаполненный раздел не позволяет разработать и проверить требования."
            ),
            evidence_pairs=[(block.text, block.id)],
            question=f"Какое содержание должно быть в разделе «{title_clean}»?",
            addressee=Role.ANALYST,
            requirement_ids=[],
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _check_placeholder_bodies(doc: Document, verifier: DocVerifier) -> list[RawFinding]:
    findings: list[RawFinding] = []
    # Group non-heading blocks by immediate section_path to detect sole-content placeholders.
    blocks_by_section: dict[tuple[str, ...], list[Block]] = {}
    for block in doc.blocks:
        if block.kind != BlockKind.HEADING:
            blocks_by_section.setdefault(tuple(block.section_path), []).append(block)

    reported_blocks: set[str] = set()
    for block in doc.blocks:
        if block.kind in (BlockKind.HEADING, BlockKind.CODE, BlockKind.TABLE_ROW):
            continue
        cleaned = strip_markdown(block.text)
        if _PLACEHOLDER_BODY_RE.match(cleaned):
            sec_blocks = blocks_by_section.get(tuple(block.section_path), [])
            sole = len(sec_blocks) == 1
            substantive = _is_substantive_section(block.section_path)
            sec_name = clean_heading(block.section_path[-1]) if block.section_path else "документе"
            sev = Severity.CRITICAL if (sole and substantive) else Severity.MAJOR
            crit = Criterion.SET_COMPLETE if sole else Criterion.COMPLETE
            is_testing = "тест" in sec_name.lower() or "тестиров" in cleaned.lower()
            finding = verifier.make_finding(
                criterion=crit,
                severity=sev,
                title=f"Раздел «{sec_name}» содержит заглушку вместо требований",
                explanation=(
                    f"В разделе «{sec_name}» вместо проверяемых требований указана заглушка «{cleaned}». "
                    "В таком виде раздел не содержит критериев и условий для работы команды."
                ),
                evidence_pairs=[(block.text, block.id)],
                question=f"Когда и кем будет заполнен раздел «{sec_name}» (сейчас стоит «{cleaned}»)?",
                addressee=Role.TESTER if is_testing else Role.ANALYST,
                requirement_ids=[] if sole else None,
            )
            if finding is not None:
                findings.append(finding)
                reported_blocks.add(block.id)
            continue

        # Check explicit inline TBD/TODO placeholders in non-boilerplate requirement blocks.
        if not is_boilerplate_section(block.section_path) and _INLINE_PLACEHOLDER_RE.search(cleaned):
            finding = verifier.make_finding(
                criterion=Criterion.COMPLETE,
                severity=Severity.MAJOR,
                title="Требование содержит пометку о незавершённом заполнении",
                explanation=(
                    "В тексте требования оставлена пометка (TBD/TODO/«уточнить позже»), "
                    "поэтому условие нельзя считать полным и пригодным к реализации."
                ),
                evidence_pairs=[(block.text, block.id)],
                question="Какое окончательное значение или правило должно заменить пометку о доработке?",
                addressee=Role.PRODUCT_OWNER,
            )
            if finding is not None:
                findings.append(finding)
                reported_blocks.add(block.id)

        # Check explicit prose stating version/date metadata is unfilled.
        if block.id not in reported_blocks:
            for sentence in split_sentences_with_lines(block.text):
                if _UNFILLED_METADATA_NOTE_RE.search(strip_markdown(sentence)):
                    finding = verifier.make_finding(
                        criterion=Criterion.CONFORMING,
                        severity=Severity.MINOR,
                        title="Не заполнены обязательные реквизиты документа (версия и дата)",
                        explanation=(
                            "Документ прямо указывает, что его версия и дата в источнике не заполнены, "
                            "что нарушает требования к оформлению и версионированию."
                        ),
                        evidence_pairs=[(sentence, block.id)],
                        question="Какая версия и дата утверждения должны быть указаны у документа?",
                        addressee=Role.ANALYST,
                        requirement_ids=[],
                    )
                    if finding is not None:
                        findings.append(finding)
                    break
    return findings


def _check_tables_hygiene(tables: list[Table], verifier: DocVerifier) -> list[RawFinding]:
    findings: list[RawFinding] = []
    for table in tables:
        sec_name = clean_heading(table.section_path[-1]) if table.section_path else "Таблица"

        # 1. Empty table: zero data rows, or every data row has all blank cells.
        all_blank = not table.rows or all(all(is_blank_cell(c) for c in cells) for _, cells in table.rows)
        if all_blank:
            pairs = [(table.header_block.text, table.header_block.id)]
            if table.rows:
                pairs.append((table.rows[0][0].text, table.rows[0][0].id))
            finding = verifier.make_finding(
                criterion=Criterion.CONFORMING,
                severity=Severity.MINOR,
                title=f"Пустая таблица в разделе «{sec_name}»",
                explanation=(
                    f"Таблица в разделе «{sec_name}» не содержит заполненных строк. "
                    "Обязательный раздел шаблона оставлен пустым."
                ),
                evidence_pairs=pairs,
                question=f"Кто должен заполнить таблицу в разделе «{sec_name}»?",
                addressee=Role.ANALYST,
                requirement_ids=[],
            )
            if finding is not None:
                findings.append(finding)
            continue

        # 2. Unmarked review checklist: 2+ rows still holding "Да / Нет" template choices.
        checklist_rows = [
            row_block
            for row_block, cells in table.rows
            if any(_CHECKLIST_CHOICE_RE.search(strip_markdown(cell)) for cell in cells)
        ]
        if len(checklist_rows) >= 2:
            first_row = checklist_rows[0]
            finding = verifier.make_finding(
                criterion=Criterion.CONFORMING,
                severity=Severity.MINOR,
                title=f"Не проставлены отметки в чек-листе раздела «{sec_name}»",
                explanation=(
                    f"В таблице раздела «{sec_name}» оставлены невыбранные шаблонные варианты "
                    f"(«Да / Нет» и аналогичные) в {len(checklist_rows)} строках — ревью по чек-листу не проведено."
                ),
                evidence_pairs=[
                    (table.header_block.text, table.header_block.id),
                    (first_row.text, first_row.id),
                ],
                question=f"Когда будут проставлены отметки проверки в чек-листе раздела «{sec_name}»?",
                addressee=Role.ANALYST,
                requirement_ids=[],
            )
            if finding is not None:
                findings.append(finding)
            continue

        # 3. Blank mandatory attributes or environment URL sub-entries in 2-column KV tables.
        if _is_kv_table(table):
            blank_pairs: list[tuple[str, str]] = []
            blank_names: list[str] = []
            has_api_gap = False
            for row_block, cells in table.rows:
                key = strip_markdown(cells[0]).lower()
                val = cells[1]
                key_matches = any(
                    key == attr or key.startswith(attr + " ") or key.endswith(" " + attr)
                    for attr in _MANDATORY_ATTR_KEYS
                )
                is_api_key = any(attr in key for attr in _API_ATTR_KEYS)
                env_match = _ENV_BLANK_RE.search(strip_markdown(val))
                if (key_matches and is_blank_cell(val)) or env_match:
                    blank_pairs.append((row_block.text, row_block.id))
                    blank_names.append(strip_markdown(cells[0]))
                    if is_api_key or env_match:
                        has_api_gap = True

            if blank_pairs:
                joined_names = ", ".join(f"«{n}»" for n in blank_names)
                finding = verifier.make_finding(
                    criterion=Criterion.COMPLETE if has_api_gap else Criterion.CONFORMING,
                    severity=Severity.MINOR,
                    title=f"Не заполнены обязательные атрибуты в разделе «{sec_name}»",
                    explanation=(
                        f"В таблице «{sec_name}» не заполнены значения обязательных полей: {joined_names}. "
                        "Отсутствие адресов окружений или ссылки на спецификацию мешает интеграции."
                    ),
                    evidence_pairs=blank_pairs,
                    question=f"Какие значения должны быть указаны для незаполненных атрибутов ({joined_names})?",
                    addressee=Role.ARCHITECT if has_api_gap else Role.ANALYST,
                )
                if finding is not None:
                    findings.append(finding)

    return findings


def check_empty_and_placeholder_sections(
    doc: Document,
    tables: list[Table],
    verifier: DocVerifier,
) -> list[RawFinding]:
    """Check for empty headings, placeholder bodies, blank metadata, and unmarked checklists."""
    return [
        *_check_empty_headings(doc, verifier),
        *_check_placeholder_bodies(doc, verifier),
        *_check_tables_hygiene(tables, verifier),
    ]


# --------------------------------------------------------------------------- weak / unmeasurable wording

_WEAK_WORDING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bмаксимально\s+быстро\b", re.IGNORECASE), "максимально быстро"),
    (re.compile(r"\bдостаточно\s+быстр(?:о|ый|ая|ое|ые)\b", re.IGNORECASE), "достаточно быстро"),
    (re.compile(r"\bкак\s+можно\s+быстрее\b", re.IGNORECASE), "как можно быстрее"),
    (re.compile(r"\bбыстро\b", re.IGNORECASE), "быстро"),
    (re.compile(r"\bоперативно\b", re.IGNORECASE), "оперативно"),
    (re.compile(r"\bудобно\b", re.IGNORECASE), "удобно"),
    (re.compile(r"\bинтуитивно\s+понятн[а-яё]*\b", re.IGNORECASE), "интуитивно понятно"),
    (re.compile(r"\bпри\s+необходимости\b", re.IGNORECASE), "при необходимости"),
    (re.compile(r"\bпо\s+возможности\b", re.IGNORECASE), "по возможности"),
    (re.compile(r"\bпо\s+усмотрению\b", re.IGNORECASE), "по усмотрению"),
    (re.compile(r"\bкак\s+правило\b", re.IGNORECASE), "как правило"),
    (re.compile(r"\bв\s+разумн(?:ый|ые)\s+срок(?:и)?\b", re.IGNORECASE), "в разумный срок"),
    (re.compile(r"\bсвоевременно\b", re.IGNORECASE), "своевременно"),
    (re.compile(r"\bбез\s+задержек\b", re.IGNORECASE), "без задержек"),
    (re.compile(r"\bи\s+т\.\s*д\.(?=\s|$|[,;:!?)\"»])", re.IGNORECASE), "и т.д."),
    (re.compile(r"\bи\s+т\.\s*п\.(?=\s|$|[,;:!?)\"»])", re.IGNORECASE), "и т.п."),
    (re.compile(r"\bи\s+пр\.(?=\s|$|[,;:!?)\"»])", re.IGNORECASE), "и пр."),
    (
        re.compile(r"(?:,\s*[^\s,]+|\d)\s+(и\s+далее)(?=\s|[—–\-,.;:!?]|$)", re.IGNORECASE),
        "и далее",
    ),
)

_SKIP_ATTR_ROWS = frozenset(
    {"author", "developer", "gitlab project", "framework", "controller", "url type", "задача в jira", "продукт"}
)


def check_weak_wording(doc: Document, verifier: DocVerifier) -> list[RawFinding]:
    """Detect classic vague Russian qualifiers in requirement-bearing blocks."""
    findings: list[RawFinding] = []
    for block in doc.blocks:
        if block.kind in (BlockKind.HEADING, BlockKind.CODE):
            continue
        if is_boilerplate_section(block.section_path):
            continue
        cleaned = strip_markdown(block.text)
        if block.kind == BlockKind.TABLE_ROW:
            cells = [strip_markdown(c) for c in parse_table_cells(block.text)]
            if len(cells) == 2 and cells[0].lower() in _SKIP_ATTR_ROWS:
                continue

        matched_phrases: list[str] = []
        for pattern, label in _WEAK_WORDING_PATTERNS:
            if pattern.search(cleaned) and label not in matched_phrases:
                matched_phrases.append(label)
        if not matched_phrases:
            continue

        # Prefer the specific sentence containing the vague phrase when available.
        quote = block.text
        for sentence in split_sentences_with_lines(block.text):
            sent_clean = strip_markdown(sentence)
            if any(pat.search(sent_clean) for pat, _ in _WEAK_WORDING_PATTERNS):
                quote = sentence
                break

        phrases_text = ", ".join(f"«{p}»" for p in matched_phrases)
        is_api = any("api" in clean_heading(p).lower() or "ответ" in clean_heading(p).lower() for p in block.section_path)
        finding = verifier.make_finding(
            criterion=Criterion.UNAMBIGUOUS,
            severity=Severity.MINOR,
            title=f"Неоднозначная или размытая формулировка ({phrases_text})",
            explanation=(
                f"Формулировка содержит неизмеримое или открытое выражение {phrases_text}, "
                "которое допускает разное прочтение и не задаёт проверяемых границ."
            ),
            evidence_pairs=[(quote, block.id)],
            question=f"Какими конкретными значениями, условиями или закрытым перечнем нужно заменить {phrases_text}?",
            addressee=Role.ARCHITECT if is_api else Role.ANALYST,
        )
        if finding is not None:
            findings.append(finding)
    return findings


# --------------------------------------------------------------------------- traceability (missing requirement IDs)

_REQ_ID_RE = re.compile(
    r"^(?:[*+-]\s+|\d+[.)]\s+)?\*{0,2}(?:Ф|НФ|БТ|Д|И|КП|П|Т|FR|NFR|REQ)-?\d+[.\s*:]",
    re.IGNORECASE,
)

_REQ_SECTION_KEYWORDS = (
    "функциональные требования",
    "нефункциональные требования",
    "требования к данным",
    "требования к интеграции",
    "критерии приемки",
    "критерии приёмки",
)


def check_traceable(
    doc: Document,
    requirements: list[Requirement],
    tables: list[Table],
    verifier: DocVerifier,
) -> list[RawFinding]:
    """Emit one consolidated `TRACEABLE` finding when top-level requirements lack document IDs."""
    if requirements:
        top_non_meta = [r for r in requirements if r.parent_id is None and r.kind != RequirementKind.META]
        missing = [r for r in top_non_meta if not (r.source_id and r.source_id.strip())]
        if not missing:
            return []

        # Pick up to 4 representative quotes across distinct RequirementKinds.
        evidence_pairs: list[tuple[str, str]] = []
        seen_kinds: set[RequirementKind] = set()
        for req in missing:
            if req.kind in seen_kinds:
                continue
            if req.block_ids:
                evidence_pairs.append((req.quote, req.block_ids[0]))
                seen_kinds.add(req.kind)
            if len(evidence_pairs) >= 4:
                break
        if not evidence_pairs:
            for req in missing[:4]:
                if req.block_ids:
                    evidence_pairs.append((req.quote, req.block_ids[0]))

        numbered_count = len(top_non_meta) - len(missing)
        sample_ids = ", ".join(r.id for r in missing[:6])
        more = f" и др. (всего {len(missing)})" if len(missing) > 6 else f" (всего {len(missing)})"
        finding = verifier.make_finding(
            criterion=Criterion.TRACEABLE,
            severity=Severity.MINOR,
            title="Часть требований не имеет собственных идентификаторов в документе",
            explanation=(
                f"Из {len(top_non_meta)} верхнеуровневых требований без идентификатора в документе "
                f"оформлено {len(missing)} (с идентификатором — {numbered_count}): {sample_ids}{more}. "
                "Без уникальных кодов сложнее ссылаться на требования из API, данных и тестов."
            ),
            evidence_pairs=evidence_pairs,
            question="Следует ли присвоить уникальные идентификаторы всем требованиям без номеров (НФ, данным, интеграции, приёмке)?",
            addressee=Role.ANALYST,
            requirement_ids=[r.id for r in missing],
        )
        return [finding] if finding is not None else []

    # Standalone mode (`requirements == []`): inspect requirement sections directly.
    table_headers = {t.header_block.id for t in tables}
    numbered_blocks: list[Block] = []
    unnumbered_blocks: list[Block] = []
    seen_sections: set[str] = set()

    for block in doc.blocks:
        if block.kind in (BlockKind.HEADING, BlockKind.CODE):
            continue
        if block.id in table_headers:
            continue
        if is_boilerplate_section(block.section_path):
            continue
        if _PLACEHOLDER_BODY_RE.match(strip_markdown(block.text)):
            continue
        sec_low = " › ".join(clean_heading(p).lower() for p in block.section_path)
        if not any(kw in sec_low for kw in _REQ_SECTION_KEYWORDS):
            continue
        if _REQ_ID_RE.match(block.text.strip()):
            numbered_blocks.append(block)
        else:
            last_sec = clean_heading(block.section_path[-1]) if block.section_path else ""
            if last_sec not in seen_sections:
                unnumbered_blocks.append(block)
                seen_sections.add(last_sec)

    if not unnumbered_blocks:
        return []

    evidence_pairs = [(b.text, b.id) for b in unnumbered_blocks[:4]]
    has_numbered = bool(numbered_blocks)
    title = (
        "Не все разделы требований имеют собственные идентификаторы"
        if has_numbered
        else "Требования в документе не имеют идентификаторов"
    )
    explanation = (
        "Идентификаторы присвоены только части требований (например, функциональным Ф-*), "
        "тогда как требования других разделов (нефункциональные, к данным, интеграции, приёмке) "
        "оформлены без идентификаторов."
        if has_numbered
        else "В разделах требований отсутствуют уникальные идентификаторы пунктов, что затрудняет трассируемость."
    )
    finding = verifier.make_finding(
        criterion=Criterion.TRACEABLE,
        severity=Severity.MINOR,
        title=title,
        explanation=explanation,
        evidence_pairs=evidence_pairs,
        question="Следует ли присвоить уникальные идентификаторы всем требованиям (НФ, данным, интеграции, приёмке)?",
        addressee=Role.ANALYST,
        requirement_ids=[],
    )
    return [finding] if finding is not None else []
