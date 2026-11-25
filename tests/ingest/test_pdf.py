from __future__ import annotations

import html as htmllib
import itertools
import re
from pathlib import Path

import pymupdf
import pytest
import yaml

from tekti.contracts import BlockKind, normalize_ws
from tekti.ingest import IngestError, load
from tekti.ingest.pdf import load_pdf

ROOT = Path(__file__).resolve().parents[2]
REAL_DOC = ROOT / "local" / "requirements.md"
GOLD = ROOT / "evals" / "gold.yaml"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_BULLET_RE = re.compile(r"^\s*[*+-]\s+(.*)$")
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")

_STORY_CSS = """
<style>
  body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
  h1 { font-size: 18pt; margin: 8px 0 4px 0; }
  h2 { font-size: 15pt; margin: 8px 0 4px 0; }
  h3 { font-size: 12.5pt; margin: 7px 0 3px 0; }
  h4 { font-size: 11pt; margin: 6px 0 3px 0; }
  h5, h6 { font-size: 10.5pt; margin: 5px 0 2px 0; }
  p { margin: 4px 0; }
  ul, ol { margin: 4px 0 4px 18px; padding-left: 12px; }
  li { margin: 2px 0; }
  code, pre { font-family: monospace; font-size: 9pt; }
  pre { background: #f5f5f5; padding: 6px; margin: 5px 0; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0; }
  th, td { border: 1px solid #444; padding: 4px; text-align: left; vertical-align: top; }
</style>
"""


def _inline_html(text: str) -> str:
    escaped = htmllib.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or _HR_RE.match(line):
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline_html(heading.group(2).strip())}</h{level}>")
            i += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith(marker):
                code_lines.append(htmllib.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(code_lines) + "</code></pre>")
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                if not _TABLE_SEP_RE.match(row_line):
                    inner = row_line.strip("|")
                    cells = [c.strip() for c in inner.split("|")]
                    rows.append(cells)
                i += 1
            if rows:
                out.append('<table border="1" cellpadding="4" cellspacing="0">')
                for r_idx, cells in enumerate(rows):
                    tag = "th" if r_idx == 0 else "td"
                    cells_html = "".join(f"<{tag}>{_inline_html(c)}</{tag}>" for c in cells)
                    out.append(f"<tr>{cells_html}</tr>")
                out.append("</table>")
            continue

        if _BULLET_RE.match(line):
            out.append("<ul>")
            while i < n and _BULLET_RE.match(lines[i]):
                item_lines = [_BULLET_RE.match(lines[i]).group(1).strip()]  # type: ignore[union-attr]
                i += 1
                while (
                    i < n
                    and lines[i].strip()
                    and not (
                        _HEADING_RE.match(lines[i].strip())
                        or _FENCE_RE.match(lines[i])
                        or _HR_RE.match(lines[i])
                        or lines[i].strip().startswith("|")
                        or _BULLET_RE.match(lines[i])
                        or _ORDERED_RE.match(lines[i])
                    )
                ):
                    item_lines.append(lines[i].strip())
                    i += 1
                out.append(f"<li>{_inline_html(' '.join(item_lines))}</li>")
            out.append("</ul>")
            continue

        if _ORDERED_RE.match(line):
            out.append("<ol>")
            while i < n and _ORDERED_RE.match(lines[i]):
                item_lines = [_ORDERED_RE.match(lines[i]).group(1).strip()]  # type: ignore[union-attr]
                i += 1
                while (
                    i < n
                    and lines[i].strip()
                    and not (
                        _HEADING_RE.match(lines[i].strip())
                        or _FENCE_RE.match(lines[i])
                        or _HR_RE.match(lines[i])
                        or lines[i].strip().startswith("|")
                        or _BULLET_RE.match(lines[i])
                        or _ORDERED_RE.match(lines[i])
                    )
                ):
                    item_lines.append(lines[i].strip())
                    i += 1
                out.append(f"<li>{_inline_html(' '.join(item_lines))}</li>")
            out.append("</ol>")
            continue

        para_lines: list[str] = []
        while (
            i < n
            and lines[i].strip()
            and not (
                _HEADING_RE.match(lines[i].strip())
                or _FENCE_RE.match(lines[i])
                or _HR_RE.match(lines[i])
                or lines[i].strip().startswith("|")
                or _BULLET_RE.match(lines[i])
                or _ORDERED_RE.match(lines[i])
            )
        ):
            para_lines.append(lines[i].strip())
            i += 1
        out.append("<p>" + "<br/>".join(_inline_html(p) for p in para_lines) + "</p>")

    return _STORY_CSS + "\n".join(out)


def markdown_to_pdf(markdown_text: str, pdf_path: Path) -> Path:
    """Render Markdown offline into a multi-page text PDF using `pymupdf.Story`."""
    html_body = _markdown_to_html(markdown_text)
    story = pymupdf.Story(html=html_body)
    writer = pymupdf.DocumentWriter(str(pdf_path))
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (36, 36, -36, -36)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device, None)
        writer.end_page()
    writer.close()
    return pdf_path


def _strip_md_quote(quote: str) -> str:
    """Strip Markdown-only syntax (`**`, `` ` ``, `# `, `* `, outer `|`) for PDF comparison."""
    s = quote.strip()
    if s.startswith("|") and s.endswith("|"):
        cells = [c.strip() for c in s.strip("|").split("|")]
        s = " | ".join(cells).rstrip()
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"^\*\s+", "", s)
    s = s.replace("**", "").replace("`", "")
    return normalize_ws(s)


def test_load_pdf_missing_and_corrupted(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(IngestError, match="Файл не найден"):
        load_pdf(missing)

    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_bytes(b"%PDF-1.4\nnot valid pdf data")
    with pytest.raises(IngestError, match="Не удалось открыть PDF"):
        load(corrupted)

    empty_bytes = tmp_path / "zero.pdf"
    empty_bytes.write_bytes(b"")
    with pytest.raises(IngestError, match="Не удалось открыть PDF|текстовый слой"):
        load_pdf(empty_bytes)


def test_load_pdf_detached_bullets_and_bold_subheading(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 70), "Main Title", fontsize=16, fontname="helv")
    page.insert_text((50, 105), "Subsection Heading", fontsize=11, fontname="hebo")
    page.insert_text((50, 130), "First body paragraph with enough words to set body font size clearly.", fontsize=11, fontname="helv")
    page.insert_text((50, 143), "More regular body text on a second line for calibration.", fontsize=11, fontname="helv")
    page.insert_text((55, 175), "-", fontsize=11, fontname="helv")
    page.insert_text((68, 175), "Detached bullet item", fontsize=11, fontname="helv")
    page.insert_text((50, 205), '{"ok": true}', fontsize=10, fontname="cour")
    doc.save(str(pdf_path))
    doc.close()

    loaded = load_pdf(pdf_path)
    kinds = [b.kind for b in loaded.blocks]
    assert kinds == [
        BlockKind.HEADING,
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST_ITEM,
        BlockKind.CODE,
    ]
    assert loaded.blocks[0].section_path == []
    assert loaded.blocks[1].section_path == ["Main Title"]
    assert loaded.blocks[2].text == (
        "First body paragraph with enough words to set body font size clearly. "
        "More regular body text on a second line for calibration."
    )
    assert loaded.blocks[3].text == "- Detached bullet item"
    assert loaded.blocks[3].section_path == ["Main Title", "Subsection Heading"]
    assert loaded.blocks[4].section_path == ["Main Title", "Subsection Heading"]


def test_load_pdf_encrypted_rejected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret content")
    doc.save(
        str(pdf_path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user123",
        owner_pw="owner123",
    )
    doc.close()

    with pytest.raises(IngestError, match="защищён паролем"):
        load_pdf(pdf_path)


def test_load_pdf_blank_and_image_only_raise_ingest_error(tmp_path: Path) -> None:
    blank_pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 200, 150), color=(0, 0, 0), width=1.5)
    doc.save(str(blank_pdf))
    doc.close()

    with pytest.raises(IngestError, match="OCR не поддерживается"):
        load_pdf(blank_pdf)

    image_only_pdf = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pix.clear_with(240)
    page.insert_image(pymupdf.Rect(72, 72, 200, 200), pixmap=pix)
    doc.save(str(image_only_pdf))
    doc.close()

    with pytest.raises(IngestError, match="не обнаружен текстовый слой"):
        load(image_only_pdf)


def test_load_pdf_synthetic_blocks_unicode_tables_and_lines(tmp_path: Path) -> None:
    sample_md = """# Заголовок документа

Вводный абзац с суммой 4 000 ₸ и переходом → к деталям.

## Раздел А

### Требования

* **Ф-1.** Порог от 1 000–1 000 000 ₸ включительно.
* **Ф-2.** Второе требование, которое переносится на следующую строку для проверки склейки.

1. Первый шаг алгоритма.
2. Второй шаг алгоритма.

| Параметр | Значение | Комментарий |
| :--- | :--- | :--- |
| REST type | GET | Тест таблицы |
| Лимит | 4 000 ₸ | В тенге |

```json
{"result": "OK", "amount": "50000"}
```

Заключительный текст после кода.
"""
    pdf_path = markdown_to_pdf(sample_md, tmp_path / "synthetic.pdf")
    doc = load(pdf_path)

    assert doc.format == "pdf"
    assert doc.name == "synthetic.pdf"
    assert doc.source_path == str(pdf_path)
    assert len(doc.sha256) == 64

    kinds = [b.kind for b in doc.blocks]
    assert kinds == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.HEADING,
        BlockKind.HEADING,
        BlockKind.LIST_ITEM,
        BlockKind.LIST_ITEM,
        BlockKind.LIST_ITEM,
        BlockKind.LIST_ITEM,
        BlockKind.TABLE_ROW,
        BlockKind.TABLE_ROW,
        BlockKind.TABLE_ROW,
        BlockKind.CODE,
        BlockKind.PARAGRAPH,
    ]
    assert [b.id for b in doc.blocks[:3]] == ["B-0001", "B-0002", "B-0003"]

    raw_lines = doc.raw_text.splitlines()
    raw_norm = normalize_ws(doc.raw_text)
    for b in doc.blocks:
        assert b.page is not None and b.page >= 1
        assert 1 <= b.line_start <= b.line_end <= len(raw_lines)
        assert "\n".join(raw_lines[b.line_start - 1 : b.line_end]) == b.text
        assert b.text in doc.raw_text
        assert normalize_ws(b.text) in raw_norm

    table_rows = [b for b in doc.blocks if b.kind is BlockKind.TABLE_ROW]
    assert [r.text for r in table_rows] == [
        "Параметр | Значение | Комментарий",
        "REST type | GET | Тест таблицы",
        "Лимит | 4 000 ₸ | В тенге",
    ]
    assert table_rows[0].section_path == ["Заголовок документа", "Раздел А", "Требования"]

    code_block = next(b for b in doc.blocks if b.kind is BlockKind.CODE)
    assert code_block.text == '{"result": "OK", "amount": "50000"}'
    assert "4 000 ₸" in doc.raw_text
    assert "1 000–1 000 000 ₸" in doc.raw_text
    assert "→" in doc.raw_text


@pytest.mark.skipif(not REAL_DOC.exists(), reason="organisers' document not present (local/ is git-ignored)")
def test_real_requirements_pdf_blocks_and_gold_quotes(tmp_path: Path) -> None:
    pdf_path = markdown_to_pdf(REAL_DOC.read_text(encoding="utf-8"), tmp_path / "requirements.pdf")
    doc = load(pdf_path)

    assert doc.format == "pdf"
    assert len(doc.blocks) > 40
    assert any(b.kind is BlockKind.TABLE_ROW for b in doc.blocks)
    assert all(b.page is not None and b.page >= 1 for b in doc.blocks)

    raw_lines = doc.raw_text.splitlines()
    raw_norm = normalize_ws(doc.raw_text)
    for b in doc.blocks:
        assert 1 <= b.line_start <= b.line_end <= len(raw_lines)
        assert "\n".join(raw_lines[b.line_start - 1 : b.line_end]) == b.text
        assert b.text in doc.raw_text
        assert normalize_ws(b.text) in raw_norm

    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in gold["items"]}
    norm_blocks = [normalize_ws(b.text) for b in doc.blocks]
    block_pairs = [f"{a} {b}" for a, b in itertools.pairwise(norm_blocks)]

    # Explicitly assert all anchor and related quotes for G-01, G-22, G-36, G-39, and G-40.
    for gid in ("G-01", "G-22", "G-36", "G-39", "G-40"):
        item = by_id[gid]
        for key in ("anchor", "related"):
            quote = item.get(key)
            if not quote:
                continue
            stripped_q = _strip_md_quote(quote)
            assert any(stripped_q in bt for bt in norm_blocks), f"{gid}.{key} not found in blocks: {stripped_q}"

    table_texts = [normalize_ws(b.text) for b in doc.blocks if b.kind is BlockKind.TABLE_ROW]
    assert any("Февральский счётчик вырос на 4 000 ₸" in t for t in table_texts)
    assert any("REST type | GET" in t for t in table_texts)
    assert any("amount | integer | 32 bit | Да | 1 000–1 000 000 | Порог в тенге | limit_kzt" in t for t in table_texts)
    assert any("amount | string | При успехе | Порог в тенге | limit_kzt" in t for t in table_texts)

    # Check recall across all 48 gold items.
    missing_all: list[tuple[str, str, str]] = []
    for item in gold["items"]:
        for key in ("anchor", "related"):
            quote = item.get(key)
            if not quote:
                continue
            stripped_q = _strip_md_quote(quote)
            if not any(stripped_q in bt for bt in norm_blocks) and not any(stripped_q in bp for bp in block_pairs):
                missing_all.append((item["id"], key, stripped_q))
    assert not missing_all, missing_all
