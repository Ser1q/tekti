"""Document ingestion: turn a file into a `Document` of numbered blocks."""

from __future__ import annotations

import importlib
from pathlib import Path

from tekti.contracts import Document

MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}


class IngestError(ValueError):
    """Raised with a Russian message when a file cannot be ingested."""


def load(path: Path) -> Document:
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"Файл не найден: {path}")
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        from tekti.ingest.markdown import load_markdown

        return load_markdown(path)
    if suffix == ".pdf":
        try:
            pdf = importlib.import_module("tekti.ingest.pdf")
        except ModuleNotFoundError as exc:
            raise IngestError(
                "Поддержка PDF не установлена: модуль tekti.ingest.pdf недоступен "
                f"({exc}). Сохраните документ в Markdown (.md) и запустите снова."
            ) from exc
        return pdf.load_pdf(path)
    raise IngestError(
        f"Неподдерживаемый формат файла «{path.suffix or 'без расширения'}»: {path.name}. "
        "Поддерживаются .md, .markdown, .txt и .pdf."
    )
