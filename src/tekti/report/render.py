"""Deterministic Russian GitHub-flavored Markdown report renderer (P7).

Renders a `Run` object strictly from its recorded data without making independent
judgements or leaking rejected findings.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import pairwise

from tekti.contracts import (
    Block,
    Criterion,
    Evidence,
    Finding,
    FindingStatus,
    Lens,
    Mode,
    Question,
    Requirement,
    RequirementKind,
    ReviewerId,
    Role,
    Run,
    Severity,
    normalize_ws,
)
from tekti.criteria import NAME_RU, SOURCE

_IGNORED_MARKUP = frozenset("*`")
_TESTING_BLOCKING_CRITERIA = frozenset({Criterion.VERIFIABLE, Criterion.SET_VALIDATABLE})
_REQ_ID_CYRILLIC_RE = re.compile(r"\b[Рр]\s*[-–—\u2011]\s*(\d{1,4})\b")
_SENTENCE_ABBREVS = frozenset(
    {
        "п",
        "пп",
        "ст",
        "ч",
        "гл",
        "разд",
        "прил",
        "абз",
        "подп",
        "рис",
        "табл",
        "см",
        "напр",
        "ср",
        "др",
        "пр",
        "стр",
        "г",
        "гг",
        "руб",
        "тыс",
        "млн",
        "млрд",
        "мин",
        "сек",
        "etc",
        "vs",
        "fig",
        "sec",
    }
)
_COMPOUND_ABBREV_RE = re.compile(
    r"(?:^|[\s«\"(])(?:и\s+)?(?:т\.\s*д|т\.\s*п|т\.\s*е|т\.\s*к|e\.\s*g|i\.\s*e)$",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r'([.!?…]+(?:[»"\')]*)?)(?=\s+[А-ЯЁA-Z«"(\d])')

MODE_RU: dict[Mode, str] = {
    Mode.FAST: "быстрый (одно семейство моделей, все углы проверки)",
    Mode.STANDARD: "стандартный (два семейства моделей)",
    Mode.BOARD: "полный консилиум (полная матрица моделей)",
}

KIND_RU: dict[RequirementKind, str] = {
    RequirementKind.FUNCTIONAL: "Функциональное",
    RequirementKind.NONFUNCTIONAL: "Нефункциональное",
    RequirementKind.DATA: "Данные",
    RequirementKind.INTEGRATION: "Интеграция",
    RequirementKind.INTERFACE: "Интерфейс (UI)",
    RequirementKind.API: "API",
    RequirementKind.ACCEPTANCE: "Критерий приёмки",
    RequirementKind.META: "Мета / цели",
}

SEVERITY_RU: dict[Severity, str] = {
    Severity.CRITICAL: "Критичное",
    Severity.MAJOR: "Существенное",
    Severity.MINOR: "Незначительное",
}

ROLE_RU: dict[Role, str] = {
    Role.BUSINESS_OWNER: "Владелец бизнеса",
    Role.PRODUCT_OWNER: "Владелец продукта",
    Role.ANALYST: "Аналитик",
    Role.ARCHITECT: "Архитектор",
    Role.DEVELOPER: "Разработчик",
    Role.TESTER: "Тестировщик",
    Role.SECURITY: "Специалист по безопасности",
    Role.CUSTOMER: "Заказчик",
}

LENS_RU: dict[Lens, str] = {
    Lens.FUNCTIONAL: "функциональный анализ",
    Lens.API_CONTRACT: "API-контракт",
    Lens.DATA_INTEGRATION: "данные и интеграция",
    Lens.TESTER: "тестировщик",
    Lens.DEVELOPER: "разработчик",
}
_LENS_VALUE_TO_RU: dict[str, str] = {lens.value: label for lens, label in LENS_RU.items()}


def _strip_markup(text: str) -> str:
    return "".join(ch for ch in text if ch not in _IGNORED_MARKUP)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_req_ids_in_text(text: str) -> str:
    """Normalize Cyrillic 'Р-005' requirement references in model prose to Latin 'R-005'."""
    return _REQ_ID_CYRILLIC_RE.sub(r"R-\1", text)


def _truncate_text(text: str, max_len: int = 120) -> str:
    """Collapse whitespace and truncate cleanly to at most `max_len` characters."""
    cleaned = _collapse_ws(text)
    if len(cleaned) <= max_len:
        return cleaned
    cut = cleaned[: max_len - 1]
    space = cut.rfind(" ")
    if space >= max_len // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + "…"


def _first_two_sentences(text: str) -> str:
    """Return at most the first two sentences without splitting abbreviations or decimals."""
    cleaned = _collapse_ws(text)
    if not cleaned:
        return ""

    guillemet_depth = 0
    paren_depth = 0
    cursor = 0
    sentence_count = 0
    last_cut = 0

    for match in _SENTENCE_END_RE.finditer(cleaned):
        for ch in cleaned[cursor : match.end()]:
            if ch == "«":
                guillemet_depth += 1
            elif ch == "»":
                guillemet_depth = max(0, guillemet_depth - 1)
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
        cursor = match.end()

        if guillemet_depth > 0 or paren_depth > 0:
            continue

        punct = match.group(1)
        if punct.startswith(".") and not punct.startswith(".."):
            before = cleaned[: match.start()]
            if re.fullmatch(r"\s*\d{1,2}", cleaned[last_cut : match.start()]):
                continue
            if _COMPOUND_ABBREV_RE.search(before):
                continue
            token_m = re.search(r"([A-Za-zА-Яа-яЁё]+)$", before)
            if token_m and token_m.group(1).lower() in _SENTENCE_ABBREVS:
                continue

        sentence_count += 1
        last_cut = match.end()
        if sentence_count == 2:
            return cleaned[:last_cut].strip()

    return cleaned


def _format_dropped_reviewer(entry: str) -> str:
    """Format 'lens/model: error' as 'angle / model: <error up to 120 chars>'."""
    head, sep, rest = entry.partition(":")
    if not sep:
        return _truncate_text(entry, 120)
    lens_raw, slash, model_raw = head.partition("/")
    if slash:
        angle = _LENS_VALUE_TO_RU.get(lens_raw.strip(), lens_raw.strip())
        prefix = f"{angle} / {model_raw.strip()}"
    else:
        prefix = head.strip()
    err = _truncate_text(rest, 120)
    return f"{prefix}: {err}" if err else prefix


def _table_text(text: str) -> str:
    """Escape text for a Markdown table cell while collapsing newlines."""
    return _collapse_ws(text).replace("|", r"\|")


def _table_quote(quote: str, *, collapse_newlines: bool) -> str:
    """Prepare a verbatim quote for a Markdown table cell without altering wording."""
    out = quote.replace("|", r"\|")
    if collapse_newlines:
        return re.sub(r"\s*\r?\n\s*", " ", out)
    return re.sub(r"\r?\n", "<br>", out)


def _blockquote_quote(quote: str) -> list[str]:
    """Format an exact quote inside a Markdown blockquote, preserving line breaks."""
    lines = quote.splitlines() or [quote]
    if len(lines) == 1:
        return [f"> «{lines[0]}»"]
    out = [f"> «{lines[0]}"]
    for mid in lines[1:-1]:
        out.append(f"> {mid}" if mid else ">")
    out.append(f"> {lines[-1]}»")
    return out


def _clean_heading(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text).strip()


def _section_label(block: Block) -> str:
    parts = [_clean_heading(p) for p in block.section_path if _clean_heading(p)]
    return " › ".join(parts) if parts else "Начало документа"


def _line_label(block: Block) -> str:
    if block.line_start == block.line_end:
        return f"строка {block.line_start}"
    return f"строки {block.line_start}–{block.line_end}"


def _format_block_location(block: Block) -> str:
    parts = [_section_label(block), _line_label(block)]
    if block.page is not None:
        parts.append(f"стр. {block.page}")
    return f"{', '.join(parts)} [{block.id}]"


def _find_block_for_quote(
    quote: str,
    block_id: str | None,
    blocks: list[Block],
    blocks_by_id: dict[str, Block],
) -> Block | None:
    if block_id and block_id in blocks_by_id:
        return blocks_by_id[block_id]
    q_norm = normalize_ws(quote)
    if not q_norm:
        return None
    for block in blocks:
        if q_norm in normalize_ws(block.text):
            return block
    for prev_block, next_block in pairwise(blocks):
        if q_norm in normalize_ws(f"{prev_block.text} {next_block.text}"):
            return prev_block
    q_stripped = normalize_ws(_strip_markup(quote))
    if len(q_stripped) >= 3:
        for block in blocks:
            if q_stripped in normalize_ws(_strip_markup(block.text)):
                return block
    return None


def _format_evidence_location(
    ev: Evidence,
    blocks: list[Block],
    blocks_by_id: dict[str, Block],
) -> str:
    block = _find_block_for_quote(ev.quote, ev.block_id, blocks, blocks_by_id)
    if block is not None:
        return _format_block_location(block)
    if ev.block_id:
        return f"блок [{ev.block_id}] (не найден в тексте документа)"
    return "место в документе не определено"


def _format_requirement_location(
    req: Requirement,
    blocks: list[Block],
    blocks_by_id: dict[str, Block],
) -> str:
    matched = [blocks_by_id[bid] for bid in req.block_ids if bid in blocks_by_id]
    if not matched:
        fallback = _find_block_for_quote(req.quote, None, blocks, blocks_by_id)
        if fallback is not None:
            matched = [fallback]
    if not matched:
        return "место в документе не определено"
    if len(matched) == 1:
        return _format_block_location(matched[0])
    first, last = matched[0], matched[-1]
    line_start = min(b.line_start for b in matched)
    line_end = max(b.line_end for b in matched)
    line_part = f"строка {line_start}" if line_start == line_end else f"строки {line_start}–{line_end}"
    sec_first = _section_label(first)
    sec_last = _section_label(last)
    sec_part = sec_first if sec_first == sec_last else f"{sec_first} … {sec_last}"
    parts = [sec_part, line_part]
    pages = [b.page for b in matched if b.page is not None]
    if pages:
        p_min, p_max = min(pages), max(pages)
        parts.append(f"стр. {p_min}" if p_min == p_max else f"стр. {p_min}–{p_max}")
    ids_part = ", ".join(b.id for b in matched)
    return f"{', '.join(parts)} [{ids_part}]"


def _order_requirements(requirements: list[Requirement]) -> list[tuple[Requirement, bool]]:
    """Order requirements with split children immediately beneath their parent."""
    by_id = {r.id: r for r in requirements}
    children_map: dict[str, list[Requirement]] = {}
    roots: list[Requirement] = []

    for req in requirements:
        if req.parent_id and req.parent_id in by_id and req.parent_id != req.id:
            children_map.setdefault(req.parent_id, []).append(req)
        else:
            roots.append(req)

    ordered: list[tuple[Requirement, bool]] = []
    seen: set[str] = set()

    def visit(req: Requirement, is_attached_child: bool) -> None:
        if req.id in seen:
            return
        seen.add(req.id)
        ordered.append((req, is_attached_child))
        for child in children_map.get(req.id, []):
            visit(child, True)

    for root in roots:
        visit(root, False)
    for req in requirements:
        if req.id not in seen:
            visit(req, bool(req.parent_id and req.parent_id in by_id and req.parent_id != req.id))
    return ordered


def _format_raiser(reviewer: ReviewerId) -> str:
    if reviewer.model == "rules":
        return "правила"
    if reviewer.lens is not None:
        return f"{reviewer.model} ({LENS_RU[reviewer.lens]})"
    return reviewer.model


def _format_req_ref(req_id: str, reqs_by_id: dict[str, Requirement]) -> str:
    req = reqs_by_id.get(req_id)
    if req and req.source_id and req.source_id.strip():
        return f"{req.id} ({req.source_id.strip()})"
    return req_id


def _first_quote_location(
    finding: Finding,
    blocks: list[Block],
    blocks_by_id: dict[str, Block],
    *,
    brief: bool = False,
) -> str:
    if not finding.evidence:
        return "место не указано"
    ev = finding.evidence[0]
    if not brief:
        return _format_evidence_location(ev, blocks, blocks_by_id)
    block = _find_block_for_quote(ev.quote, ev.block_id, blocks, blocks_by_id)
    if block is not None:
        sec_parts = [_clean_heading(p) for p in block.section_path if _clean_heading(p)]
        sec = " › ".join(sec_parts[-2:]) if len(sec_parts) >= 2 else (sec_parts[0] if sec_parts else "Начало документа")
        loc_parts = [sec, _line_label(block)]
        if block.page is not None:
            loc_parts.append(f"стр. {block.page}")
        return ", ".join(loc_parts)
    if ev.block_id:
        return f"блок [{ev.block_id}] (не найден в тексте документа)"
    return "место в документе не определено"


def _render_finding_details(
    finding: Finding,
    *,
    blocks: list[Block],
    blocks_by_id: dict[str, Block],
    reqs_by_id: dict[str, Requirement],
    side_by_side_table: bool,
    show_requirements: bool,
) -> list[str]:
    title = _normalize_req_ids_in_text(_collapse_ws(finding.title))
    explanation = _normalize_req_ids_in_text(finding.explanation)
    lines: list[str] = [
        f"#### {finding.id}. {title}",
        "",
        f"- **Критерий:** {NAME_RU[finding.criterion]} ({SOURCE[finding.criterion]})",
        f"- **Критичность:** {SEVERITY_RU[finding.severity]}",
    ]
    if show_requirements:
        if finding.requirement_ids:
            req_labels = ", ".join(_format_req_ref(rid, reqs_by_id) for rid in finding.requirement_ids)
        else:
            req_labels = "весь документ"
        lines.append(f"- **Затронутые требования:** {req_labels}")

    lines.append(f"- **Пояснение:** {explanation}")
    raisers = ", ".join(_format_raiser(r) for r in finding.raised_by) if finding.raised_by else "не указаны"
    lines.append(f"- **Подтверждение:** семейств моделей — {finding.families}; источники: {raisers}")

    if finding.challenge is not None:
        ch = finding.challenge
        if ch.upheld:
            if ch.severity is not None:
                verdict_text = f"подтверждено (итоговая критичность: {SEVERITY_RU[ch.severity]})"
            else:
                verdict_text = "подтверждено"
        else:
            verdict_text = "отклонено"
        reasoning_short = _first_two_sentences(_normalize_req_ids_in_text(ch.reasoning))
        if reasoning_short:
            lines.append(f"- **Перекрёстная проверка ({ch.challenger}):** {verdict_text} — {reasoning_short}")
        else:
            lines.append(f"- **Перекрёстная проверка ({ch.challenger}):** {verdict_text}")

    if not finding.evidence:
        lines.append("- **Цитаты и расположение:** не приведены.")
        return lines

    if side_by_side_table and len(finding.evidence) >= 2:
        pair = finding.evidence[:2]
        headers: list[str] = []
        cells: list[str] = []
        for idx, ev in enumerate(pair, 1):
            loc = _format_evidence_location(ev, blocks, blocks_by_id)
            prefix = _format_req_ref(ev.requirement_id, reqs_by_id) if ev.requirement_id else f"Цитата {idx}"
            headers.append(_table_text(f"{prefix} — {loc}"))
            cells.append(f"«{_table_quote(ev.quote, collapse_newlines=False)}»")
        lines.append("")
        lines.append(f"| {headers[0]} | {headers[1]} |")
        lines.append("| :--- | :--- |")
        lines.append(f"| {cells[0]} | {cells[1]} |")
        for extra_idx, ev in enumerate(finding.evidence[2:], 3):
            loc = _format_evidence_location(ev, blocks, blocks_by_id)
            prefix = _format_req_ref(ev.requirement_id, reqs_by_id) if ev.requirement_id else f"Цитата {extra_idx}"
            lines.append("")
            lines.append(f"- **Дополнительная цитата ({prefix} — {loc}):**")
            lines.extend(_blockquote_quote(ev.quote))
        return lines

    lines.append("- **Цитаты и расположение:**")
    for ev in finding.evidence:
        loc = _format_evidence_location(ev, blocks, blocks_by_id)
        req_note = f" ({_format_req_ref(ev.requirement_id, reqs_by_id)})" if ev.requirement_id else ""
        lines.append(f"  - {loc}{req_note}:")
        lines.extend(f"  {bq}" for bq in _blockquote_quote(ev.quote))
    return lines


def _render_first_screen_summary(
    run: Run,
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
) -> list[str]:
    v = run.verdict
    sev_counts = Counter(f.severity for f in confirmed)
    lines = [
        "### Краткий итог на первом экране",
        "",
        f"- **Готовность к разработке:** {'Да' if v.ready_for_development else 'Нет'}",
        f"- **Готовность к тестированию:** {'Да' if v.ready_for_testing else 'Нет'}",
        (
            f"- **Подтверждено замечаний:** {len(confirmed)} "
            f"(критичных — {sev_counts[Severity.CRITICAL]}, "
            f"существенных — {sev_counts[Severity.MAJOR]}, "
            f"незначительных — {sev_counts[Severity.MINOR]})"
        ),
    ]

    confirmed_by_id = {f.id: f for f in confirmed}
    critical_items: list[Finding] = []
    seen_crit: set[str] = set()
    for fid in v.fix_first:
        f = confirmed_by_id.get(fid)
        if f is not None and f.id not in seen_crit:
            critical_items.append(f)
            seen_crit.add(f.id)
    for f in confirmed:
        if f.severity == Severity.CRITICAL and f.id not in seen_crit:
            critical_items.append(f)
            seen_crit.add(f.id)
    critical_items = critical_items[:10]

    blocks = run.document.blocks
    total_crit = sev_counts[Severity.CRITICAL]
    if not critical_items:
        lines.append("- **Критичных дефектов: 0**")
    else:
        crit_header = (
            f"- **Критичные дефекты (показаны {len(critical_items)} из {total_crit}):**"
            if total_crit > len(critical_items)
            else "- **Критичные дефекты:**"
        )
        lines.append(crit_header)
        for f in critical_items:
            title = _normalize_req_ids_in_text(_collapse_ws(f.title))
            loc = _first_quote_location(f, blocks, blocks_by_id, brief=True)
            lines.append(f"  - **{f.id}** — {title} ({loc})")

    ordered_questions = sorted(run.questions, key=lambda q: (q.priority, q.id))
    top_questions = ordered_questions[:5]
    if not top_questions:
        lines.append("- **Уточняющих вопросов: 0**")
    else:
        q_header = (
            f"- **Главные вопросы (первые {len(top_questions)} из {len(ordered_questions)}):**"
            if len(ordered_questions) > len(top_questions)
            else "- **Главные вопросы:**"
        )
        lines.append(q_header)
        for idx, q in enumerate(top_questions, 1):
            q_text = _normalize_req_ids_in_text(_collapse_ws(q.text))
            lines.append(f"  {idx}. **{q.id}** ({ROLE_RU[q.addressee]}): {q_text}")

    return lines


def _render_passport(
    run: Run,
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
) -> list[str]:
    p = run.passport
    models_text = ", ".join(f"`{m}`" for m in p.models) if p.models else "не использовались (детерминированный прогон)"
    lines = [
        "## 1. Паспорт прогона",
        "",
        f"- **Документ:** `{p.document_name}`",
        f"- **Формат:** `{run.document.format}`",
        f"- **SHA-256:** `{p.document_sha256}`",
        f"- **Дата и время запуска:** {p.started_at.isoformat(sep=' ', timespec='seconds')}",
        f"- **Режим проверки:** `{p.mode.value}` — {MODE_RU[p.mode]}",
        f"- **Модели:** {models_text}",
        f"- **Версия агента:** `{p.agent_version}`",
        f"- **Исходных замечаний до сведения:** {len(run.raw_findings)}",
    ]
    if not p.dropped_reviewers:
        lines.append("- **Отпавшие рецензенты:** нет")
    else:
        lines.append(f"- **Отпавшие рецензенты ({len(p.dropped_reviewers)}):**")
        for entry in p.dropped_reviewers:
            lines.append(f"  - {_format_dropped_reviewer(entry)}")

    if not run.segmentation_dropped:
        lines.append("- **Отброшенные кандидаты нарезки:** нет")
    else:
        lines.append(f"- **Отброшенные кандидаты нарезки ({len(run.segmentation_dropped)}):**")
        for item in run.segmentation_dropped:
            lines.append(f"  - {_collapse_ws(item)}")

    lines.append("")
    lines.extend(_render_first_screen_summary(run, confirmed, blocks_by_id))
    return lines


def _render_segmentation(run: Run, ordered_reqs: list[tuple[Requirement, bool]]) -> list[str]:
    lines = ["## 2. Как агент нарезал требования", ""]
    if not ordered_reqs:
        lines.append(
            f"Из документа «{run.document.name}» (проверено блоков: {len(run.document.blocks)}) выделено 0 требований."
        )
        return lines

    lines.append("| Идентификатор в отчёте | Идентификатор в документе | Тип | Краткая формулировка |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for req, is_attached_child in ordered_reqs:
        if is_attached_child and req.parent_id:
            rep_id = f"↳ {req.id} (часть {req.parent_id})"
        elif req.parent_id:
            rep_id = f"{req.id} (часть {req.parent_id})"
        else:
            rep_id = req.id

        has_source_id = bool(req.source_id and req.source_id.strip())
        doc_id = _table_text(req.source_id) if has_source_id and req.source_id else "—"
        kind_label = KIND_RU[req.kind]
        formulation = _table_text(req.summary)
        if not has_source_id:
            quote_cell = _table_quote(req.quote, collapse_newlines=True)
            formulation = f"{formulation} — «{quote_cell}»"
        lines.append(f"| {_table_text(rep_id)} | {doc_id} | {kind_label} | {formulation} |")
    return lines


def _render_per_requirement(
    run: Run,
    ordered_reqs: list[tuple[Requirement, bool]],
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
    reqs_by_id: dict[str, Requirement],
) -> list[str]:
    lines = ["## 3. Проверка каждого требования", ""]
    if not ordered_reqs:
        lines.append(
            f"Поэлементная проверка требований не выполнялась, так как из документа «{run.passport.document_name}» "
            f"(блоков: {len(run.document.blocks)}) выделено 0 требований."
        )
        return lines

    blocks = run.document.blocks
    parent_locations = {r.id: _format_requirement_location(r, blocks, blocks_by_id) for r in run.requirements}
    parent_was_bullet: set[str] = set()
    prev_was_bullet = False

    for req, is_attached_child in ordered_reqs:
        single_findings = [f for f in confirmed if not f.is_set_level and req.id in f.requirement_ids]
        has_source_id = bool(req.source_id and req.source_id.strip())
        source_part = f" ({req.source_id.strip()})" if has_source_id and req.source_id else ""
        parent_part = f" [часть {req.parent_id}]" if req.parent_id else ""

        if not single_findings:
            set_refs = [f.id for f in confirmed if f.is_set_level and req.id in f.requirement_ids]
            set_part = f" (см. замечания к набору: {', '.join(set_refs)})" if set_refs else ""
            short_quote = _truncate_text(req.quote, 120)

            if is_attached_child and req.parent_id:
                core = f"↳ **{req.id}**{source_part}{parent_part}: «{short_quote}» — индивидуальных замечаний нет{set_part}."
                if req.parent_id in parent_was_bullet and prev_was_bullet:
                    lines.append(f"  - {core}")
                else:
                    parent_was_bullet.discard(req.parent_id)
                    if lines and lines[-1] != "":
                        lines.append("")
                    lines.append(f"- {core}")
                prev_was_bullet = True
            else:
                if not prev_was_bullet and lines and lines[-1] != "":
                    lines.append("")
                lines.append(
                    f"- **{req.id}**{source_part}{parent_part}: «{short_quote}» — "
                    f"индивидуальных замечаний нет{set_part}."
                )
                parent_was_bullet.add(req.id)
                prev_was_bullet = True
            continue

        if is_attached_child and req.parent_id:
            parent_was_bullet.discard(req.parent_id)
        prev_was_bullet = False
        if lines and lines[-1] != "":
            lines.append("")

        heading_parts = [req.id]
        if has_source_id and req.source_id:
            heading_parts.append(f"({req.source_id.strip()})")
        if req.parent_id:
            heading_parts.append(f"[часть {req.parent_id}]")
        lines.append(f"### {' '.join(heading_parts)}")
        lines.append("")
        lines.append(f"- **Тип:** {KIND_RU[req.kind]}")
        lines.append(f"- **Краткая формулировка:** {_collapse_ws(req.summary)}")

        loc = parent_locations.get(req.id) or _format_requirement_location(req, blocks, blocks_by_id)
        omit_location = (
            is_attached_child
            and bool(req.parent_id)
            and req.parent_id in parent_locations
            and loc == parent_locations[req.parent_id]
        )
        if not omit_location:
            lines.append(f"- **Расположение:** {loc}")

        lines.append("")
        lines.extend(_blockquote_quote(req.quote))

        for f in single_findings:
            lines.append("")
            lines.extend(
                _render_finding_details(
                    f,
                    blocks=blocks,
                    blocks_by_id=blocks_by_id,
                    reqs_by_id=reqs_by_id,
                    side_by_side_table=False,
                    show_requirements=False,
                )
            )
    return lines


def _render_set_level(
    run: Run,
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
    reqs_by_id: dict[str, Requirement],
) -> list[str]:
    lines = ["## 4. Проверка всего набора требований", ""]
    set_findings = [f for f in confirmed if f.is_set_level]
    if not set_findings:
        lines.append(
            f"На уровне всего набора (проверено требований: {len(run.requirements)}, "
            f"блоков документа: {len(run.document.blocks)}) подтверждённых замечаний не выявлено (0)."
        )
        return lines

    grouped: dict[Criterion, list[Finding]] = {}
    for f in set_findings:
        grouped.setdefault(f.criterion, []).append(f)

    blocks = run.document.blocks
    for g_idx, (criterion, items) in enumerate(grouped.items()):
        if g_idx > 0:
            lines.append("")
        lines.append(f"### {NAME_RU[criterion]} ({SOURCE[criterion]})")
        for f in items:
            lines.append("")
            lines.extend(
                _render_finding_details(
                    f,
                    blocks=blocks,
                    blocks_by_id=blocks_by_id,
                    reqs_by_id=reqs_by_id,
                    side_by_side_table=True,
                    show_requirements=True,
                )
            )
    return lines


KEY_QUESTIONS_MAX = 20  # questions shown in full; the rest go into one compact table


def _render_questions(questions: list[Question], confirmed: list[Finding]) -> list[str]:
    """Section 5. Questions that close a critical finding are shown in full (at most
    KEY_QUESTIONS_MAX, by priority); every other question is one row of a table. Nothing is left out."""
    lines = ["## 5. Уточняющие вопросы", ""]
    if not questions:
        lines.append(
            f"По результатам анализа {len(confirmed)} подтверждённых замечаний уточняющие вопросы не сформированы (0)."
        )
        return lines

    ordered = sorted(questions, key=lambda q: (q.priority, q.id))
    critical_ids = {f.id for f in confirmed if f.severity == Severity.CRITICAL}
    key = [q for q in ordered if critical_ids.intersection(q.finding_ids)][:KEY_QUESTIONS_MAX]
    if not key:  # no critical findings: the highest-priority questions are the key ones
        key = ordered[: min(KEY_QUESTIONS_MAX, len(ordered))]
    key_ids = {q.id for q in key}
    rest = [q for q in ordered if q.id not in key_ids]

    if rest:
        lines.append(
            f"Всего вопросов: {len(ordered)}. Ключевые ({len(key)}) закрывают критичные замечания и приведены "
            f"полностью; остальные ({len(rest)}) собраны в таблицу в конце раздела."
        )
    else:
        lines.append(f"Всего вопросов: {len(ordered)}.")
    lines.append("")
    lines.append("**Кому адресованы:**")
    lines.append("")
    by_role: dict[Role, list[str]] = {}
    for q in ordered:
        by_role.setdefault(q.addressee, []).append(q.id)
    for role, ids in sorted(by_role.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"- {ROLE_RU[role]} ({len(ids)}): {', '.join(ids)}")

    for q in key:
        lines.append("")
        lines.append(f"### {q.id} (приоритет {q.priority})")
        lines.append("")
        lines.append(f"- **Вопрос:** {_normalize_req_ids_in_text(q.text)}")
        lines.append(f"- **Адресат:** {ROLE_RU[q.addressee]}")
        lines.append(f"- **Что блокирует:** {_normalize_req_ids_in_text(q.blocks)}")
        lines.append(f"- **Варианты ответа:** {_question_options(q)}")
        settled = ", ".join(q.finding_ids) if q.finding_ids else "не указаны"
        lines.append(f"- **Закрываемые замечания:** {settled}")

    if rest:
        lines.append("")
        lines.append("### Остальные вопросы")
        lines.append("")
        lines.append("| № | Вопрос | Адресат | Варианты ответа | Замечания |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for q in rest:
            options = _question_options(q) if q.options else "—"
            lines.append(
                f"| {q.id} | {_table_text(_normalize_req_ids_in_text(q.text))} | {ROLE_RU[q.addressee]} "
                f"| {_table_text(options)} | {', '.join(q.finding_ids) or '—'} |"
            )
    return lines


def _question_options(q: Question) -> str:
    if q.options:
        return "; ".join(f"«{_normalize_req_ids_in_text(opt)}»" for opt in q.options)
    return "документ не сужает выбор до фиксированных вариантов"


def _render_summary(
    run: Run,
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
    reqs_by_id: dict[str, Requirement],
) -> list[str]:
    lines = ["## 6. Сводка", ""]
    total_reqs = len(run.requirements)
    child_reqs = sum(1 for r in run.requirements if r.parent_id is not None)
    top_reqs = total_reqs - child_reqs

    if total_reqs == 0:
        lines.append("- **Всего требований:** 0")
    else:
        lines.append(
            f"- **Всего требований:** {total_reqs} "
            f"(верхнеуровневых / самостоятельных — {top_reqs}, дочерних после разбивки — {child_reqs})"
        )
        kind_counts = Counter(r.kind for r in run.requirements)
        kind_parts = [f"{KIND_RU[k]}: {kind_counts[k]}" for k in RequirementKind if kind_counts[k] > 0]
        lines.append(f"- **Распределение по типам требований:** {'; '.join(kind_parts)}")

    lines.append(f"- **Всего подтверждённых замечаний:** {len(confirmed)}")
    lines.append("")

    sev_counts = Counter(f.severity for f in confirmed)
    lines.append("| Критичность | Количество |")
    lines.append("| :--- | ---: |")
    for sev in (Severity.CRITICAL, Severity.MAJOR, Severity.MINOR):
        lines.append(f"| {SEVERITY_RU[sev]} | {sev_counts[sev]} |")

    lines.append("")
    if not confirmed:
        lines.append("**Замечания по критериям качества:** подтверждённых замечаний по критериям — 0.")
    else:
        crit_counts = Counter(f.criterion for f in confirmed)
        lines.append("| Критерий | Стандарт | Количество |")
        lines.append("| :--- | :--- | ---: |")
        for crit in Criterion:
            if crit_counts[crit] > 0:
                lines.append(f"| {NAME_RU[crit]} | {SOURCE[crit]} | {crit_counts[crit]} |")

    lines.append("")
    critical = [f for f in confirmed if f.severity == Severity.CRITICAL]
    lines.append("**Список критичных дефектов:**")
    lines.append("")
    if not critical:
        lines.append(f"Среди {len(confirmed)} подтверждённых замечаний критичных дефектов нет (0).")
    else:
        blocks = run.document.blocks
        for f in critical:
            req_part = (
                ", ".join(_format_req_ref(rid, reqs_by_id) for rid in f.requirement_ids)
                if f.requirement_ids
                else "весь документ"
            )
            loc = _first_quote_location(f, blocks, blocks_by_id)
            title = _normalize_req_ids_in_text(_collapse_ws(f.title))
            lines.append(
                f"- **{f.id}** ({NAME_RU[f.criterion]}): {title} — требования: {req_part}; расположение: {loc}"
            )
    return lines


def _render_verdict(
    run: Run,
    confirmed: list[Finding],
    blocks_by_id: dict[str, Block],
) -> list[str]:
    v = run.verdict
    confirmed_by_id = {f.id: f for f in confirmed}
    blocks = run.document.blocks

    lines = [
        "## 7. Итог",
        "",
        f"- **Можно ли начинать разработку:** {'Да' if v.ready_for_development else 'Нет'}",
        f"- **Можно ли начинать тестирование:** {'Да' if v.ready_for_testing else 'Нет'}",
        f"- **Обоснование:** {v.rationale}",
        "",
        "**Что исправить в первую очередь:**",
        "",
    ]
    if not v.fix_first:
        lines.append("Критичных дефектов для первоочередного исправления нет (0).")
    else:
        for fid in v.fix_first:
            f = confirmed_by_id.get(fid)
            if f is not None:
                loc = _first_quote_location(f, blocks, blocks_by_id)
                title = _normalize_req_ids_in_text(_collapse_ws(f.title))
                lines.append(f"- **{f.id}** — {title} ({loc})")
            else:
                lines.append(f"- **{fid}**")

    if not v.ready_for_testing:
        test_blockers = [
            f for f in confirmed if f.severity == Severity.MAJOR and f.criterion in _TESTING_BLOCKING_CRITERIA
        ]
        if test_blockers:
            lines.append("")
            lines.append(
                "**Существенные замечания, блокирующие тестирование "
                f"(критерии «{NAME_RU[Criterion.VERIFIABLE]}» и «{NAME_RU[Criterion.SET_VALIDATABLE]}»):**"
            )
            lines.append("")
            for f in test_blockers:
                loc = _first_quote_location(f, blocks, blocks_by_id)
                title = _normalize_req_ids_in_text(_collapse_ws(f.title))
                lines.append(f"- **{f.id}** ({NAME_RU[f.criterion]}) — {title} ({loc})")
    return lines


def render(run: Run) -> str:
    """Render a complete Run as a Russian GitHub-flavored Markdown report."""
    confirmed = [f for f in run.findings if f.status == FindingStatus.CONFIRMED]
    blocks_by_id = {b.id: b for b in run.document.blocks}
    reqs_by_id = {r.id: r for r in run.requirements}
    ordered_reqs = _order_requirements(run.requirements)

    doc_title = run.passport.document_name or run.document.name
    sections: list[list[str]] = [
        [f"# Отчёт о проверке требований: {doc_title}"],
        _render_passport(run, confirmed, blocks_by_id),
        _render_segmentation(run, ordered_reqs),
        _render_per_requirement(run, ordered_reqs, confirmed, blocks_by_id, reqs_by_id),
        _render_set_level(run, confirmed, blocks_by_id, reqs_by_id),
        _render_questions(run.questions, confirmed),
        _render_summary(run, confirmed, blocks_by_id, reqs_by_id),
        _render_verdict(run, confirmed, blocks_by_id),
    ]
    return "\n\n".join("\n".join(sec) for sec in sections) + "\n"
