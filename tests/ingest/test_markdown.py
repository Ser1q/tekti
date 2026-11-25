from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from tekti.contracts import BlockKind, normalize_ws
from tekti.ingest import IngestError, load
from tekti.ingest.markdown import load_markdown, split_blocks

ROOT = Path(__file__).resolve().parents[2]
REAL_DOC = ROOT / "local" / "requirements.md"
GOLD = ROOT / "evals" / "gold.yaml"

SAMPLE = """# Title

Intro line one,
intro line two.

## Part A

### Rules
* **Ф-1.** First rule.
* Second rule
  continues here.

1. Step one.
2. Step two.

| Col | Val |
| :--- | :--- |
| a | 1 000 ₸ |
| b | two |

---

## Part B

```json
{"x": 1}
```
Tail paragraph.
"""


def test_split_blocks_kinds_lines_and_sections():
    blocks = split_blocks(SAMPLE)
    kinds = [b.kind for b in blocks]
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
        BlockKind.HEADING,
        BlockKind.CODE,
        BlockKind.PARAGRAPH,
    ]
    assert [b.id for b in blocks[:3]] == ["B-0001", "B-0002", "B-0003"]
    by_text = {b.text: b for b in blocks}
    intro = blocks[1]
    assert (intro.line_start, intro.line_end) == (3, 4)
    assert intro.section_path == ["Title"]
    rule = by_text["* **Ф-1.** First rule."]
    assert rule.section_path == ["Title", "Part A", "Rules"]
    assert rule.line_start == 9
    cont = by_text["* Second rule\ncontinues here."]
    assert (cont.line_start, cont.line_end) == (10, 11)
    assert "1. Step one." in by_text
    header = by_text["| Col | Val |"]
    assert header.kind is BlockKind.TABLE_ROW and header.line_start == 16
    assert "| a | 1 000 ₸ |" in by_text
    assert not any(":---" in b.text for b in blocks)
    assert by_text["### Rules"].kind is BlockKind.HEADING
    part_b = by_text["## Part B"]
    assert part_b.section_path == ["Title"]
    code = by_text['```json\n{"x": 1}\n```']
    assert (code.line_start, code.line_end) == (25, 27)
    assert code.section_path == ["Title", "Part B"]
    assert by_text["Tail paragraph."].line_start == 28


def test_every_block_is_substring_of_normalized_raw():
    raw = normalize_ws(SAMPLE)
    for b in split_blocks(SAMPLE):
        assert normalize_ws(b.text) in raw, b


def test_load_markdown_hash_and_raw(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(SAMPLE, encoding="utf-8")
    doc = load_markdown(p)
    assert doc.raw_text == SAMPLE
    assert doc.format == "markdown"
    assert len(doc.sha256) == 64
    assert doc.name == "doc.md"


def test_load_dispatch_errors(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_bytes(b"x")
    with pytest.raises(IngestError, match="Неподдерживаемый формат"):
        load(p)
    with pytest.raises(IngestError, match="Файл не найден"):
        load(tmp_path / "missing.md")
    t = tmp_path / "doc.txt"
    t.write_text("Просто текст.", encoding="utf-8")
    assert load(t).blocks[0].text == "Просто текст."


def test_load_pdf_dispatch(tmp_path, monkeypatch):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    try:
        import tekti.ingest.pdf  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(IngestError, match="Поддержка PDF не установлена"):
            load(p)
    else:
        pytest.skip("tekti.ingest.pdf is installed; its own tests cover it")


@pytest.mark.skipif(not REAL_DOC.exists(), reason="organisers' document not present (local/ is git-ignored)")
def test_real_document_blocks_and_gold_quotes():
    doc = load(REAL_DOC)
    raw = normalize_ws(doc.raw_text)
    norm_blocks = [normalize_ws(b.text) for b in doc.blocks]
    for b, nt in zip(doc.blocks, norm_blocks, strict=True):
        assert nt in raw, b.id
    lines = doc.raw_text.splitlines()
    for b in doc.blocks:
        assert 1 <= b.line_start <= b.line_end <= len(lines)
        assert normalize_ws(b.text).split(" ")[-1] in normalize_ws(lines[b.line_end - 1]) or b.kind is BlockKind.HEADING

    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    quotes = []
    for item in gold["items"]:
        for key in ("anchor", "related"):
            if item.get(key):
                quotes.append((item["id"], key, item[key]))
    assert quotes
    pairs = [f"{a} {b}" for a, b in itertools.pairwise(norm_blocks)]
    missing = []
    for gid, key, q in quotes:
        nq = normalize_ws(q)
        if not any(nq in t for t in norm_blocks) and not any(nq in t for t in pairs):
            missing.append((gid, key, q))
    assert not missing, missing
