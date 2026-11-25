"""Text PDF ingestion: extract ordered, section-aware blocks from text-based PDFs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from tekti.contracts import Block, BlockKind, Document, normalize_ws
from tekti.ingest import IngestError

_TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
_DICT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES

_LIGATURES: tuple[tuple[str, str], ...] = (
    ("ﬀ", "ff"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
    ("\u00a0", " "),
)

_MONO_TOKENS = (
    "mono",
    "courier",
    "consolas",
    "menlo",
    "monaco",
    "inconsolata",
    "jetbrains",
    "firacode",
    "fira mono",
    "sourcecode",
    "source code",
)
_BOLD_TOKENS = ("bold", "semibold", "demibold", "extrabold", "heavy", "black")

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET_MARKER = re.compile(r"^\s*(?:[•●◦▪▫‣⁃]|[*+-])\s+(.+)$")
_SYMBOL_BULLET = re.compile(r"^\s*[•●◦▪▫‣⁃]\s*(.+)$")
_ORDERED_MARKER = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+)$")
_TABLE_SEP_CELL = re.compile(r"^:?-{2,}:?$")
_WRAP_JOIN_TAIL = re.compile(r"\S[-–/]$")
_SENTENCE_END = re.compile(r"[.!?:;]$")

_MIN_HEADING_DELTA_PT = 0.75
_MIN_HEADING_RATIO = 1.07
_HEADING_CLUSTER_TOLERANCE_PT = 0.4


@dataclass(frozen=True)
class _SpanInfo:
    text: str
    size: float
    font: str
    flags: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class _PageLine:
    block_idx: int
    x0: float
    y0: float
    x1: float
    y1: float
    spans: list[_SpanInfo]

    @property
    def text(self) -> str:
        return _clean_ligatures("".join(s.text for s in self.spans))

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


@dataclass
class _RawTextGroup:
    y0: float
    x0: float
    lines: list[_PageLine]
    is_table: bool = False
    table_text: str = ""


@dataclass
class _ClassifiedItem:
    page: int
    y0: float
    x0: float
    kind: BlockKind
    text: str
    heading_level: int | None = None
    heading_title: str | None = None
    lines_count: int = 1
    y1: float = 0.0


def _clean_ligatures(text: str) -> str:
    for src, dst in _LIGATURES:
        if src in text:
            text = text.replace(src, dst)
    return text


def _is_monospace_font(font_name: str, flags: int) -> bool:
    if flags & pymupdf.TEXT_FONT_MONOSPACED:
        return True
    lower = font_name.lower()
    return any(tok in lower for tok in _MONO_TOKENS)


def _is_bold_font(font_name: str, flags: int) -> bool:
    if flags & pymupdf.TEXT_FONT_BOLD:
        return True
    lower = font_name.lower()
    return any(tok in lower for tok in _BOLD_TOKENS)


def _join_wrapped_lines(lines: list[str]) -> str:
    """Join soft-wrapped lines cleanly, preserving hyphenated words and slashes."""
    parts: list[str] = []
    for raw_line in lines:
        cleaned = normalize_ws(_clean_ligatures(raw_line))
        if not cleaned:
            continue
        if not parts:
            parts.append(cleaned)
        elif _WRAP_JOIN_TAIL.search(parts[-1]):
            parts[-1] = f"{parts[-1]}{cleaned}"
        else:
            parts.append(cleaned)
    return " ".join(parts)


def _is_list_start(text: str) -> bool:
    stripped = text.strip()
    if _BULLET_MARKER.match(stripped) or _SYMBOL_BULLET.match(stripped):
        return True
    ordered = _ORDERED_MARKER.match(stripped)
    if ordered:
        num = int(ordered.group(1))
        return 1 <= num <= 99
    return False


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if not (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return False
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(parsed, (dict, list))


def _line_metrics(line: _PageLine) -> tuple[float, bool, bool]:
    """Return (dominant_font_size, all_monospace, all_bold) for non-whitespace spans."""
    size_weights: Counter[float] = Counter()
    all_mono = True
    all_bold = True
    has_chars = False

    for span in line.spans:
        stripped = _clean_ligatures(span.text).strip()
        if not stripped:
            continue
        has_chars = True
        weight = len(stripped)
        size_weights[round(span.size, 1)] += weight
        if not _is_monospace_font(span.font, span.flags):
            all_mono = False
        if not _is_bold_font(span.font, span.flags):
            all_bold = False

    if not has_chars:
        return (0.0, False, False)
    dominant_size = size_weights.most_common(1)[0][0]
    return (dominant_size, all_mono, all_bold)


def _group_metrics(lines: list[_PageLine]) -> tuple[float, bool, bool]:
    """Return (dominant_font_size, all_monospace, all_bold) across a list of lines."""
    size_weights: Counter[float] = Counter()
    all_mono = True
    all_bold = True
    has_chars = False

    for line in lines:
        for span in line.spans:
            stripped = _clean_ligatures(span.text).strip()
            if not stripped:
                continue
            has_chars = True
            size_weights[round(span.size, 1)] += len(stripped)
            if not _is_monospace_font(span.font, span.flags):
                all_mono = False
            if not _is_bold_font(span.font, span.flags):
                all_bold = False

    if not has_chars:
        return (0.0, False, False)
    dominant_size = size_weights.most_common(1)[0][0]
    return (dominant_size, all_mono, all_bold)


def _find_page_tables(page: pymupdf.Page) -> tuple[list[tuple[float, float, str]], list[pymupdf.Rect]]:
    """Detect tables on a page and extract one row string per valid row."""
    try:
        finder = page.find_tables(use_layout=False)
    except TypeError:
        finder = page.find_tables()

    tables = getattr(finder, "tables", None)
    if tables is None:
        tables = list(finder)

    valid_tables = []
    for table in tables:
        rows = getattr(table, "rows", None)
        row_count = len(rows) if rows is not None else 0
        col_count = getattr(table, "col_count", 0)
        # Skip single-cell bordered callouts or decorative rectangles.
        if col_count < 2 and row_count < 2:
            continue
        valid_tables.append(table)

    table_rects = [pymupdf.Rect(t.bbox) for t in valid_tables]
    extracted_rows: list[tuple[float, float, str]] = []

    for table in valid_tables:
        rows = getattr(table, "rows", None)
        if rows:
            for row in rows:
                cleaned_cells: list[str] = []
                for cell in row.cells:
                    if cell is None:
                        cleaned_cells.append("")
                    else:
                        raw_cell = page.get_text("text", clip=pymupdf.Rect(cell), flags=_TEXT_FLAGS)
                        cleaned_cells.append(_join_wrapped_lines(raw_cell.splitlines()))
                if not cleaned_cells or all(not c or _TABLE_SEP_CELL.match(c) for c in cleaned_cells):
                    continue
                row_text = " | ".join(cleaned_cells).rstrip()
                extracted_rows.append((float(row.bbox[1]), float(row.bbox[0]), row_text))
        else:
            raw_rows = table.extract()
            tx0, ty0, _, ty1 = table.bbox
            step = (ty1 - ty0) / max(1, len(raw_rows))
            for idx, raw_row in enumerate(raw_rows):
                cleaned_cells = [normalize_ws(_clean_ligatures(cell or "")) for cell in raw_row]
                if not cleaned_cells or all(not c or _TABLE_SEP_CELL.match(c) for c in cleaned_cells):
                    continue
                row_text = " | ".join(cleaned_cells).rstrip()
                extracted_rows.append((float(ty0 + idx * step), float(tx0), row_text))

    return extracted_rows, table_rects


def _extract_page_lines(page: pymupdf.Page, table_rects: list[pymupdf.Rect]) -> list[_PageLine]:
    """Extract non-table lines from `page.get_text('dict')` and merge split baseline fragments."""
    text_dict = page.get_text("dict", flags=_DICT_FLAGS)
    lines: list[_PageLine] = []

    for block_idx, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            kept_spans: list[_SpanInfo] = []
            for raw_span in raw_line.get("spans", []):
                sx0, sy0, sx1, sy1 = raw_span["bbox"]
                mid = pymupdf.Point((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0)
                if any(rect.contains(mid) for rect in table_rects):
                    continue
                text = raw_span.get("text", "")
                if not text:
                    continue
                kept_spans.append(
                    _SpanInfo(
                        text=text,
                        size=float(raw_span.get("size", 10.0)),
                        font=str(raw_span.get("font", "")),
                        flags=int(raw_span.get("flags", 0)),
                        x0=float(sx0),
                        y0=float(sy0),
                        x1=float(sx1),
                        y1=float(sy1),
                    )
                )
            if not kept_spans:
                continue
            combined = _clean_ligatures("".join(s.text for s in kept_spans))
            if not combined.strip():
                continue
            x0 = min(s.x0 for s in kept_spans)
            y0 = min(s.y0 for s in kept_spans)
            x1 = max(s.x1 for s in kept_spans)
            y1 = max(s.y1 for s in kept_spans)
            lines.append(_PageLine(block_idx=block_idx, x0=x0, y0=y0, x1=x1, y1=y1, spans=kept_spans))

    if not lines:
        return []

    lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
    merged: list[_PageLine] = [lines[0]]
    for curr in lines[1:]:
        prev = merged[-1]
        baseline_tol = max(2.5, 0.35 * min(prev.height, curr.height))
        same_baseline = abs(curr.y0 - prev.y0) <= baseline_tol
        gap = curr.x0 - prev.x1
        prev_len = len(prev.text.strip())
        # Merge fragments of the same visual line (e.g., a detached bullet or number before text).
        should_merge = same_baseline and -2.0 <= gap <= 24.0 and (
            curr.block_idx == prev.block_idx or prev_len <= 8
        )
        if should_merge:
            joiner = [] if prev.text.endswith(" ") or curr.text.startswith(" ") else [
                _SpanInfo(
                    text=" ",
                    size=curr.spans[0].size,
                    font=curr.spans[0].font,
                    flags=curr.spans[0].flags,
                    x0=prev.x1,
                    y0=curr.y0,
                    x1=curr.x0,
                    y1=curr.y1,
                )
            ]
            prev.spans.extend(joiner)
            prev.spans.extend(curr.spans)
            prev.x1 = max(prev.x1, curr.x1)
            prev.y0 = min(prev.y0, curr.y0)
            prev.y1 = max(prev.y1, curr.y1)
            if prev_len <= 8:
                prev.block_idx = curr.block_idx
        else:
            merged.append(curr)

    return merged


def _calibrate_heading_levels(page_lines: list[list[_PageLine]]) -> tuple[float, list[float]]:
    """Compute document body font size and descending larger-font heading size clusters."""
    size_counts: Counter[float] = Counter()
    for lines in page_lines:
        for line in lines:
            for span in line.spans:
                stripped = _clean_ligatures(span.text).strip()
                if not stripped or _is_monospace_font(span.font, span.flags):
                    continue
                size_counts[round(span.size, 1)] += len(stripped)

    if not size_counts:
        return (10.0, [])

    body_size = size_counts.most_common(1)[0][0]
    heading_threshold = max(body_size + _MIN_HEADING_DELTA_PT, body_size * _MIN_HEADING_RATIO)

    candidate_sizes: set[float] = set()
    for lines in page_lines:
        for line in lines:
            text = normalize_ws(line.text)
            if not text or _is_list_start(text):
                continue
            dom_size, all_mono, _ = _line_metrics(line)
            if not all_mono and dom_size >= heading_threshold and len(text) <= 200:
                candidate_sizes.add(dom_size)

    clusters: list[float] = []
    for sz in sorted(candidate_sizes, reverse=True):
        if not clusters or (clusters[-1] - sz) > _HEADING_CLUSTER_TOLERANCE_PT:
            clusters.append(sz)
    return (body_size, clusters)


def _heading_level_for_size(size: float, clusters: list[float]) -> int | None:
    if not clusters:
        return None
    for idx, cluster_sz in enumerate(clusters):
        if abs(size - cluster_sz) <= _HEADING_CLUSTER_TOLERANCE_PT:
            return idx + 1
    return None


def _is_bold_subheading(lines: list[_PageLine], all_bold: bool, text: str) -> bool:
    """Detect body-sized standalone bold headings without false-positiving on numbers/sentences."""
    if not all_bold or len(lines) > 2 or len(text) > 90 or len(text.split()) > 12:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    return not (_is_list_start(text) or _SENTENCE_END.search(text))


def _split_list_and_paragraphs(lines: list[_PageLine]) -> list[tuple[BlockKind, list[_PageLine]]]:
    """Split a block of body lines into individual LIST_ITEMs and PARAGRAPHs."""
    if not lines:
        return []

    max_x1 = max(ln.x1 for ln in lines)
    min_x0 = min(ln.x0 for ln in lines)
    block_width = max(1.0, max_x1 - min_x0)

    segments: list[tuple[BlockKind, list[_PageLine]]] = []
    current_kind: BlockKind | None = None
    current_lines: list[_PageLine] = []
    current_marker_x0: float | None = None

    def flush() -> None:
        nonlocal current_kind, current_lines, current_marker_x0
        if current_kind is not None and current_lines:
            segments.append((current_kind, current_lines))
        current_kind = None
        current_lines = []
        current_marker_x0 = None

    for idx, line in enumerate(lines):
        stripped = normalize_ws(line.text)
        if not stripped:
            continue

        is_bullet = bool(_BULLET_MARKER.match(stripped) or _SYMBOL_BULLET.match(stripped))
        ordered_match = _ORDERED_MARKER.match(stripped)
        is_ordered = False
        if ordered_match:
            num = int(ordered_match.group(1))
            if 1 <= num <= 99:
                if idx == 0 or current_kind is BlockKind.LIST_ITEM:
                    is_ordered = True
                elif current_lines:
                    prev_text = normalize_ws(current_lines[-1].text)
                    prev_short = (current_lines[-1].x1 - min_x0) < block_width * 0.8
                    if _SENTENCE_END.search(prev_text) or prev_short:
                        is_ordered = True

        if is_bullet or is_ordered:
            flush()
            current_kind = BlockKind.LIST_ITEM
            current_lines = [line]
            current_marker_x0 = line.x0
            continue

        if current_kind is BlockKind.LIST_ITEM and current_marker_x0 is not None:
            # An unindented line to the left of the bullet marker ends the list item.
            if line.x0 < current_marker_x0 - 4.0:
                flush()
                current_kind = BlockKind.PARAGRAPH
                current_lines = [line]
            else:
                current_lines.append(line)
            continue

        if current_kind is None:
            current_kind = BlockKind.PARAGRAPH
        current_lines.append(line)

    flush()
    return segments


def _segment_page_lines(
    page_num: int,
    lines: list[_PageLine],
    body_size: float,
    heading_clusters: list[float],
) -> list[_ClassifiedItem]:
    """Group a page's non-table lines into classified structural items."""
    if not lines:
        return []

    heading_threshold = max(body_size + _MIN_HEADING_DELTA_PT, body_size * _MIN_HEADING_RATIO)

    def line_role(line: _PageLine) -> tuple[str, int | None]:
        text = normalize_ws(line.text)
        md_match = _MD_HEADING.match(text)
        if md_match:
            return ("md_heading", len(md_match.group(1)))
        dom_size, all_mono, _ = _line_metrics(line)
        if all_mono:
            return ("code", None)
        if dom_size >= heading_threshold and not _is_list_start(text) and len(text) <= 200:
            lvl = _heading_level_for_size(dom_size, heading_clusters)
            if lvl is not None:
                return ("heading", lvl)
        return ("body", None)

    # First partition lines by block_idx AND role transitions so mixed blocks split cleanly.
    raw_groups: list[tuple[str, int | None, list[_PageLine]]] = []
    for line in lines:
        role, lvl = line_role(line)
        if not raw_groups:
            raw_groups.append((role, lvl, [line]))
            continue
        prev_role, prev_lvl, prev_lines = raw_groups[-1]
        same_block = line.block_idx == prev_lines[-1].block_idx
        if same_block and role == prev_role and lvl == prev_lvl:
            prev_lines.append(line)
        elif role == "code" and prev_role == "code":
            # Keep consecutive monospace lines in one code block if vertically close and not distinct JSON objects.
            prev_txt = normalize_ws(prev_lines[-1].text)
            curr_txt = normalize_ws(line.text)
            gap = line.y0 - prev_lines[-1].y1
            separate_json = prev_txt.endswith("}") and curr_txt.startswith("{")
            if not separate_json and gap <= 2.2 * prev_lines[-1].height:
                prev_lines.append(line)
            else:
                raw_groups.append((role, lvl, [line]))
        else:
            raw_groups.append((role, lvl, [line]))

    items: list[_ClassifiedItem] = []
    bold_heading_level = len(heading_clusters) + 1

    for role, lvl, grp_lines in raw_groups:
        if role == "md_heading" and lvl is not None:
            for ln in grp_lines:
                txt = normalize_ws(ln.text)
                m = _MD_HEADING.match(txt)
                title = m.group(2).strip() if m else txt
                items.append(
                    _ClassifiedItem(
                        page=page_num,
                        y0=ln.y0,
                        x0=ln.x0,
                        kind=BlockKind.HEADING,
                        text=txt,
                        heading_level=lvl,
                        heading_title=title,
                        y1=ln.y1,
                    )
                )
            continue

        if role == "heading" and lvl is not None:
            txt = _join_wrapped_lines([ln.text for ln in grp_lines])
            if txt:
                items.append(
                    _ClassifiedItem(
                        page=page_num,
                        y0=grp_lines[0].y0,
                        x0=grp_lines[0].x0,
                        kind=BlockKind.HEADING,
                        text=txt,
                        heading_level=lvl,
                        heading_title=txt,
                        y1=grp_lines[-1].y1,
                    )
                )
            continue

        if role == "code":
            code_lines = [_clean_ligatures(ln.text).rstrip() for ln in grp_lines if ln.text.strip()]
            code_text = "\n".join(code_lines).strip()
            if code_text:
                items.append(
                    _ClassifiedItem(
                        page=page_num,
                        y0=grp_lines[0].y0,
                        x0=grp_lines[0].x0,
                        kind=BlockKind.CODE,
                        text=code_text,
                        lines_count=len(code_lines),
                        y1=grp_lines[-1].y1,
                    )
                )
            continue

        # Body lines: check standalone bold subheading or JSON before splitting lists/paragraphs.
        _, _, grp_all_bold = _group_metrics(grp_lines)
        joined_body = _join_wrapped_lines([ln.text for ln in grp_lines])
        if _is_bold_subheading(grp_lines, grp_all_bold, joined_body):
            items.append(
                _ClassifiedItem(
                    page=page_num,
                    y0=grp_lines[0].y0,
                    x0=grp_lines[0].x0,
                    kind=BlockKind.HEADING,
                    text=joined_body,
                    heading_level=bold_heading_level,
                    heading_title=joined_body,
                    y1=grp_lines[-1].y1,
                )
            )
            continue

        if _looks_like_json(joined_body):
            code_lines = [_clean_ligatures(ln.text).strip() for ln in grp_lines if ln.text.strip()]
            code_text = "\n".join(code_lines)
            items.append(
                _ClassifiedItem(
                    page=page_num,
                    y0=grp_lines[0].y0,
                    x0=grp_lines[0].x0,
                    kind=BlockKind.CODE,
                    text=code_text,
                    lines_count=len(code_lines),
                    y1=grp_lines[-1].y1,
                )
            )
            continue

        for seg_kind, seg_lines in _split_list_and_paragraphs(grp_lines):
            seg_text = _join_wrapped_lines([ln.text for ln in seg_lines])
            if not seg_text:
                continue
            items.append(
                _ClassifiedItem(
                    page=page_num,
                    y0=seg_lines[0].y0,
                    x0=seg_lines[0].x0,
                    kind=seg_kind,
                    text=seg_text,
                    y1=seg_lines[-1].y1,
                )
            )

    return items


def load_pdf(path: Path) -> Document:
    """Ingest a text-based PDF into a `Document` of ordered, section-aware `Block`s."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"Файл не найден: {path}")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"Не удалось прочитать файл PDF «{path.name}»: {exc}") from exc

    try:
        pdf_doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestError(
            f"Не удалось открыть PDF «{path.name}»: файл повреждён или имеет некорректный формат."
        ) from exc

    try:
        if pdf_doc.needs_pass:
            raise IngestError(
                f"PDF «{path.name}» защищён паролем: чтение зашифрованных PDF-документов не поддерживается."
            )
        if len(pdf_doc) == 0:
            raise IngestError(
                f"В PDF «{path.name}» нет страниц и не обнаружен текстовый слой; OCR не поддерживается."
            )

        page_tables: list[list[tuple[float, float, str]]] = []
        page_lines: list[list[_PageLine]] = []

        for page in pdf_doc:
            rows, rects = _find_page_tables(page)
            page_tables.append(rows)
            page_lines.append(_extract_page_lines(page, rects))
    finally:
        pdf_doc.close()

    body_size, heading_clusters = _calibrate_heading_levels(page_lines)

    ordered_items: list[_ClassifiedItem] = []
    for page_idx, (rows, lines) in enumerate(zip(page_tables, page_lines, strict=True)):
        page_num = page_idx + 1
        page_items = _segment_page_lines(page_num, lines, body_size, heading_clusters)
        for y0, x0, row_text in rows:
            page_items.append(
                _ClassifiedItem(
                    page=page_num,
                    y0=y0,
                    x0=x0,
                    kind=BlockKind.TABLE_ROW,
                    text=row_text,
                )
            )
        page_items.sort(key=lambda it: (round(it.y0, 1), round(it.x0, 1)))
        ordered_items.extend(page_items)

    if not ordered_items:
        raise IngestError(
            f"В PDF «{path.name}» не обнаружен текстовый слой (возможно, это скан или пустой документ); "
            "OCR не поддерживается."
        )

    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    raw_lines: list[str] = []
    prev_kind: BlockKind | None = None

    for item in ordered_items:
        block_lines = item.text.splitlines()
        if not block_lines or not normalize_ws(item.text):
            continue

        # Add a blank line separator between distinct blocks (keeping table/list groups compact).
        if raw_lines and not (
            (item.kind is BlockKind.TABLE_ROW and prev_kind is BlockKind.TABLE_ROW)
            or (item.kind is BlockKind.LIST_ITEM and prev_kind is BlockKind.LIST_ITEM)
        ):
            raw_lines.append("")

        line_start = len(raw_lines) + 1
        raw_lines.extend(block_lines)
        line_end = len(raw_lines)

        if item.kind is BlockKind.HEADING and item.heading_level is not None and item.heading_title:
            while heading_stack and heading_stack[-1][0] >= item.heading_level:
                heading_stack.pop()
            section_path = [title for _, title in heading_stack]
            heading_stack.append((item.heading_level, item.heading_title))
        else:
            section_path = [title for _, title in heading_stack]

        blocks.append(
            Block(
                id=f"B-{len(blocks) + 1:04d}",
                kind=item.kind,
                text=item.text,
                section_path=section_path,
                line_start=line_start,
                line_end=line_end,
                page=item.page,
            )
        )
        prev_kind = item.kind

    raw_text = "\n".join(raw_lines)
    if not blocks or not normalize_ws(raw_text):
        raise IngestError(
            f"В PDF «{path.name}» не обнаружен текстовый слой (возможно, это скан или пустой документ); "
            "OCR не поддерживается."
        )

    return Document(
        name=path.name,
        source_path=str(path),
        format="pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        raw_text=raw_text,
        blocks=blocks,
    )
