"""Shared helpers, table parser, quote verification, and requirement matching for the rule layer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Evidence,
    Fact,
    RawFinding,
    Requirement,
    ReviewerId,
    Role,
    Severity,
    normalize_ws,
)

RULES_REVIEWER = ReviewerId(lens=None, model="rules")

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
_MARKUP_CHARS = str.maketrans("", "", "*`")

_BOILERPLATE_SECTION_KEYWORDS = (
    "общие сведения",
    "ревью требований",
    "чек-лист",
    "чеклист",
    "чек лист",
    "блок согласующих",
    "согласующие",
    "лист согласования",
    "история изменений",
    "метаданные документа",
)

_API_SUBSECTION_MARKERS = (
    "общие параметры",
    "логика работы",
    "особые правила",
    "схема",
    "параметры запроса",
    "query-параметры",
    "query параметры",
    "path-параметры",
    "path параметры",
    "request body",
    "тело запроса",
    "ответ",
    "параметры ответа",
    "статусы",
    "коды ответа",
    "коды состояний",
    "request",
    "response",
    "statuses",
    "general",
)

_BLANK_CELL_VALUES = frozenset(
    {
        "",
        "-",
        "—",
        "–",
        "―",
        "n/a",
        "na",
        "н/д",
        "tbd",
        "todo",
        "tba",
        "?",
        "не указано",
        "не заполнено",
        "отсутствует",
    }
)


@dataclass(frozen=True)
class Table:
    """A contiguous Markdown or PDF table reconstructed from `TABLE_ROW` blocks."""

    section_path: list[str]
    header_block: Block
    header_cells: list[str]
    rows: list[tuple[Block, list[str]]]


def strip_markdown(text: str) -> str:
    """Remove inline emphasis and code backticks for parsing (never for quote output)."""
    return text.translate(_MARKUP_CHARS).strip()


def clean_heading(text: str) -> str:
    """Strip Markdown heading markers and inline emphasis from a heading string."""
    match = _HEADING_PREFIX_RE.match(text.strip())
    inner = match.group(1) if match else text.strip()
    return strip_markdown(inner)


def heading_level(block: Block) -> int:
    """Return heading depth (1-based), using '#' count when present or section_path depth."""
    stripped = block.text.lstrip()
    if stripped.startswith("#"):
        hashes = len(stripped) - len(stripped.lstrip("#"))
        if 1 <= hashes <= 6:
            return hashes
    return len(block.section_path) + 1


def parse_table_cells(text: str) -> list[str]:
    """Split a Markdown or PDF table row block into trimmed cell strings."""
    s = text.strip().removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in s.split("|")]


def is_blank_cell(cell: str) -> bool:
    """Return True if a table cell is empty or holds a blank/placeholder dash."""
    cleaned = strip_markdown(cell).lower().strip()
    return cleaned in _BLANK_CELL_VALUES


def is_boilerplate_section(section_path: list[str]) -> bool:
    """Return True if the block belongs to template boilerplate (passport, checklist, approvers)."""
    for heading in section_path:
        low = clean_heading(heading).lower()
        if any(kw in low for kw in _BOILERPLATE_SECTION_KEYWORDS):
            return True
    return False


def endpoint_scope(section_path: list[str]) -> tuple[str, ...]:
    """Return the enclosing endpoint/document section tuple for an API block."""
    cleaned = [clean_heading(part) for part in section_path]
    for idx, part in enumerate(cleaned):
        low = part.lower()
        if any(low == marker or low.startswith(marker) for marker in _API_SUBSECTION_MARKERS):
            return tuple(cleaned[:idx])
    return tuple(cleaned)


def extract_tables(doc: Document) -> list[Table]:
    """Group consecutive `TABLE_ROW` blocks into structured `Table` objects."""
    tables: list[Table] = []
    cur_header_block: Block | None = None
    cur_header_cells: list[str] = []
    cur_rows: list[tuple[Block, list[str]]] = []
    prev_block: Block | None = None

    def flush() -> None:
        nonlocal cur_header_block, cur_header_cells, cur_rows
        if cur_header_block is not None:
            tables.append(
                Table(
                    section_path=list(cur_header_block.section_path),
                    header_block=cur_header_block,
                    header_cells=cur_header_cells,
                    rows=cur_rows,
                )
            )
        cur_header_block = None
        cur_header_cells = []
        cur_rows = []

    for block in doc.blocks:
        if block.kind != BlockKind.TABLE_ROW:
            flush()
            prev_block = None
            continue

        cells = parse_table_cells(block.text)
        same_table = (
            cur_header_block is not None
            and prev_block is not None
            and block.section_path == cur_header_block.section_path
            and len(cells) == len(cur_header_cells)
            and (doc.format != "markdown" or block.line_start - prev_block.line_end <= 2)
        )
        if same_table:
            cur_rows.append((block, cells))
        else:
            flush()
            cur_header_block = block
            cur_header_cells = cells
            cur_rows = []
        prev_block = block

    flush()
    return tables


def split_sentences_with_lines(text: str) -> list[str]:
    """Split block text into individual non-empty lines and sentences (keeping exact substrings)."""
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on sentence boundaries ". " / "? " / "! " only when followed by uppercase/digit/bullet.
        chunks = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9*«\"`])", line)
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
    return parts


class DocVerifier:
    """Verifies quotes against `Document.raw_text` and resolves `Requirement` IDs."""

    def __init__(self, doc: Document, requirements: list[Requirement]) -> None:
        self.doc = doc
        self.norm_raw = normalize_ws(doc.raw_text)
        self.requirements = requirements
        self._by_block: dict[str, list[Requirement]] = {}
        for req in requirements:
            for bid in req.block_ids:
                self._by_block.setdefault(bid, []).append(req)

    def verify_quote(self, quote: str) -> str | None:
        """Return the quote if it exists in `normalize_ws(doc.raw_text)`, else None."""
        q = quote.strip()
        if not q:
            return None
        if normalize_ws(q) in self.norm_raw:
            return q
        return None

    def find_requirement_id(self, block_id: str, quote: str) -> str | None:
        """Find the best matching `Requirement.id` for `(block_id, quote)`."""
        if not self.requirements:
            return None
        q_norm = normalize_ws(quote)
        if not q_norm:
            return None

        cands = self._by_block.get(block_id, [])
        if not cands:
            # Fallback: quote overlap across all requirements when block_ids were omitted.
            cands = [
                r
                for r in self.requirements
                if q_norm in normalize_ws(r.quote) or normalize_ws(r.quote) in q_norm
            ]
        if not cands:
            return None

        containing = [r for r in cands if q_norm in normalize_ws(r.quote)]
        if containing:
            # Pick the most specific (shortest) requirement containing the quote (e.g. split child).
            return min(containing, key=lambda r: (len(normalize_ws(r.quote)), r.parent_id is None)).id

        contained = [r for r in cands if normalize_ws(r.quote) in q_norm]
        if contained:
            # Pick the broadest requirement covered by the quote (e.g. parent over child).
            return max(contained, key=lambda r: (len(normalize_ws(r.quote)), r.parent_id is None)).id

        if len(cands) == 1:
            req = cands[0]
            if req.parent_id is None:
                return req.id
        return None

    def make_evidence(self, quote: str, block_id: str) -> Evidence | None:
        """Build an `Evidence` object only if `quote` is verified against `doc.raw_text`."""
        verified = self.verify_quote(quote)
        if verified is None:
            return None
        req_id = self.find_requirement_id(block_id, verified)
        return Evidence(quote=verified, block_id=block_id, requirement_id=req_id)

    def make_finding(
        self,
        *,
        criterion: Criterion,
        severity: Severity,
        title: str,
        explanation: str,
        evidence_pairs: list[tuple[str, str]],
        question: str | None = None,
        addressee: Role | None = None,
        requirement_ids: list[str] | None = None,
    ) -> RawFinding | None:
        """Construct a `RawFinding` after verifying every evidence quote against `doc.raw_text`."""
        evidence: list[Evidence] = []
        seen_quotes: set[str] = set()
        for quote, block_id in evidence_pairs:
            ev = self.make_evidence(quote, block_id)
            if ev is None:
                continue
            q_norm = normalize_ws(ev.quote)
            if q_norm in seen_quotes:
                continue
            seen_quotes.add(q_norm)
            evidence.append(ev)

        if not evidence:
            return None

        if requirement_ids is None:
            req_ids = list(dict.fromkeys(ev.requirement_id for ev in evidence if ev.requirement_id))
        else:
            req_ids = list(dict.fromkeys(requirement_ids))

        return RawFinding(
            reviewer=RULES_REVIEWER,
            requirement_ids=req_ids,
            criterion=criterion,
            severity=severity,
            title=title,
            explanation=explanation,
            evidence=evidence,
            question=question,
            addressee=addressee,
        )


class FactCollector:
    """Assigns sequential `FACT-NNN` identifiers and verifies fact quotes."""

    def __init__(self, verifier: DocVerifier) -> None:
        self._verifier = verifier
        self._facts: list[Fact] = []

    def add(
        self,
        *,
        name: str,
        value: str,
        quote: str,
        block_id: str,
        unit: str | None = None,
    ) -> Fact | None:
        verified = self._verifier.verify_quote(quote)
        if verified is None:
            return None
        fact = Fact(
            id=f"FACT-{len(self._facts) + 1:03d}",
            name=name,
            value=value,
            unit=unit,
            quote=verified,
            block_id=block_id,
            requirement_id=self._verifier.find_requirement_id(block_id, verified),
        )
        self._facts.append(fact)
        return fact

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts)
