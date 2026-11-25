"""Markdown (and plain text) ingestion: a line-based splitter that keeps line numbers exact.

Block text is kept as the raw source content so that any quote copied from a block is a
substring of the document after `normalize_ws`:

* headings: the whole line, '#' marks included ("### Цели"); `section_path` holds the bare titles;
* paragraphs: the source lines joined with newlines (surrounding spaces trimmed);
* list items: the whole item, bullet or number included ("* **Ф-1.** ...", "1. Проверить ...");
* table rows: the original row text, pipes included, trimmed. Cells are NOT re-joined with
  " | ", because re-joining would change spacing and break substring matching for quotes that
  span cells. The separator row (| :--- |) is skipped; the header row is its own block;
* fenced code: one block holding all lines including the fences.

Markup is kept everywhere because the gold quotes (evals/gold.yaml) cite it, e.g.
"### Требования к интеграции" or "* **Content-Type:** `application/json` .".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tekti.contracts import Block, BlockKind, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_BULLET = re.compile(r"^(\s*)([*+-])\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _starts_block(line: str) -> bool:
    return bool(
        _HEADING.match(line)
        or _FENCE.match(line)
        or _HR.match(line)
        or _is_table_line(line)
        or _BULLET.match(line)
        or _ORDERED.match(line)
    )


def split_blocks(text: str) -> list[Block]:
    """Split Markdown text into blocks. Pure function, used by `load_markdown` and tests."""
    lines = text.splitlines()
    blocks: list[Block] = []
    headings: list[tuple[int, str]] = []  # (level, text)

    def add(kind: BlockKind, body: str, start: int, end: int) -> None:
        blocks.append(
            Block(
                id=f"B-{len(blocks) + 1:04d}",
                kind=kind,
                text=body,
                section_path=[h for _, h in headings],
                line_start=start,
                line_end=end,
            )
        )

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        lineno = i + 1
        if not line.strip() or _HR.match(line):
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            j = i + 1
            while j < n and not lines[j].strip().startswith(marker):
                j += 1
            end = min(j, n - 1)
            add(BlockKind.CODE, "\n".join(lines[i : end + 1]).strip(), lineno, end + 1)
            i = end + 1
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            add(BlockKind.HEADING, line.strip(), lineno, lineno)
            headings.append((level, title))
            i += 1
            continue

        if _is_table_line(line):
            if not _TABLE_SEP.match(line):
                add(BlockKind.TABLE_ROW, line.strip(), lineno, lineno)
            i += 1
            continue

        item = _BULLET.match(line) or _ORDERED.match(line)
        if item:
            indent = item.group(1)
            body_lines = [line.strip()]
            j = i + 1
            # Continuation lines: non-blank, indented deeper than the bullet, not a new block.
            while j < n and lines[j].strip() and not _starts_block(lines[j]) and (
                len(lines[j]) - len(lines[j].lstrip()) > len(indent)
            ):
                body_lines.append(lines[j].strip())
                j += 1
            add(BlockKind.LIST_ITEM, "\n".join(body_lines), lineno, j)
            i = j
            continue

        # Paragraph: consecutive non-blank lines that do not start another block.
        j = i + 1
        while j < n and lines[j].strip() and not _starts_block(lines[j]):
            j += 1
        body = "\n".join(x.strip() for x in lines[i:j])
        add(BlockKind.PARAGRAPH, body, lineno, j)
        i = j

    return blocks


def load_markdown(path: Path) -> Document:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Файл {path} не в кодировке UTF-8: {exc}") from exc
    return Document(
        name=path.name,
        source_path=str(path),
        format="markdown",
        sha256=hashlib.sha256(data).hexdigest(),
        raw_text=text,
        blocks=split_blocks(text),
    )
