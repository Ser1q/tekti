"""Command line: tekti <requirements file> <report file>."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from importlib import import_module
from pathlib import Path

from tekti.contracts import Mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tekti",
        description="Проверка документа с требованиями: дефекты с привязкой к тексту и уточняющие вопросы.",
    )
    parser.add_argument("input", type=Path, help="файл требований (.md, .txt, .pdf)")
    parser.add_argument("report", type=Path, help="куда записать отчёт (.md)")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.BOARD.value,
        help="fast: одно семейство моделей; standard: два; board: три (по умолчанию)",
    )
    parser.add_argument("--run-json", type=Path, help="куда сохранить данные прогона (по умолчанию рядом с отчётом)")
    parser.add_argument("--no-rules", action="store_true", help="не запускать проверку правилами")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 2

    from tekti import pipeline

    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f} с] {message}", file=sys.stderr, flush=True)

    try:
        run = asyncio.run(pipeline.run(args.input, args.mode, use_rules=not args.no_rules, progress=progress))
    except Exception as exc:  # noqa: BLE001 - the CLI reports any failure in one line
        print(f"Ошибка прогона: {exc}", file=sys.stderr)
        return 1

    run_json = args.run_json or args.report.with_suffix(".run.json")
    run_json.parent.mkdir(parents=True, exist_ok=True)
    run_json.write_text(run.model_dump_json(indent=1), encoding="utf-8")

    try:
        report = import_module("tekti.report")
    except ModuleNotFoundError:
        print(f"Модуль отчёта не установлен. Данные прогона: {run_json}", file=sys.stderr)
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report.render(run), encoding="utf-8")

    critical = sum(1 for f in run.findings if f.severity.value == "critical")
    progress(
        f"Готово: требований {len(run.requirements)}, замечаний {len(run.findings)} "
        f"(критичных {critical}), вопросов {len(run.questions)}"
    )
    print(f"Отчёт: {args.report}\nДанные прогона: {run_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
