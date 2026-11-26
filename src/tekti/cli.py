"""Command line.

tekti <requirements file> <report file> [--mode ...]   review a document (the organisers' single command)
tekti login | logout | whoami                          sign in to the Tekti gateway with GitHub
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit

from tekti.contracts import Mode

ACCOUNT_COMMANDS = ("login", "logout", "whoami")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ACCOUNT_COMMANDS:
        return account_main(argv)
    return review_main(argv)


def build_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tekti",
        description="Проверка документа с требованиями: дефекты с привязкой к тексту и уточняющие вопросы.",
        epilog="Вход в шлюз Tekti через GitHub: tekti login | tekti whoami | tekti logout",
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
    return parser


def review_main(argv: list[str]) -> int:
    args = build_review_parser().parse_args(argv)

    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 2

    from tekti import pipeline
    from tekti.config import load_config
    from tekti.llm_backends import GatewayRefused

    try:
        cfg = load_config(args.mode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from tekti.config import choose_responsive_models, probe_timeout

    try:
        cfg, notes = asyncio.run(choose_responsive_models(cfg, probe_timeout()))
    except GatewayRefused as exc:
        print(f"Проверка остановлена: {exc}", file=sys.stderr)
        return 3
    for note in notes:
        print(note, file=sys.stderr)
    if cfg.gateway_expires_at is not None:
        left_min = (cfg.gateway_expires_at - datetime.now(UTC)).total_seconds() / 60
        if left_min < 30:
            print(
                f"Сессия шлюза истекает через {left_min:.0f} мин — при необходимости выполните `uv run tekti login`.",
                file=sys.stderr,
            )

    started = time.monotonic()

    def progress(message: str) -> None:
        print(f"[{time.monotonic() - started:6.1f} с] {message}", file=sys.stderr, flush=True)

    try:
        run = asyncio.run(pipeline.run(args.input, args.mode, use_rules=not args.no_rules, cfg=cfg, progress=progress))
    except GatewayRefused as exc:
        print(f"Проверка остановлена: {exc}", file=sys.stderr)
        return 3
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


# --------------------------------------------------------------------------- login / logout / whoami


def account_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tekti", description="Вход в шлюз Tekti через GitHub.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="войти через GitHub (device flow) и сохранить сессию шлюза")
    sub.add_parser("logout", help="удалить сохранённую сессию для текущего шлюза")
    sub.add_parser("whoami", help="показать, под каким логином GitHub вы вошли и до когда")
    args = parser.parse_args(argv)

    from tekti.config import env_settings, gateway_url_from, github_client_id
    from tekti.llm_backends import gateway_auth

    try:
        env = env_settings()
        gateway = gateway_url_from(env)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not gateway:
        print("Не задан адрес шлюза: укажите TEKTI_GATEWAY_URL в .env или в окружении.", file=sys.stderr)
        return 2
    host = urlsplit(gateway).hostname

    if args.command == "whoami":
        if env.get("TEKTI_GATEWAY_TOKEN", "").strip():
            print(f"Шлюз {host}: используется токен владельца (TEKTI_GATEWAY_TOKEN).")
        session = gateway_auth.load_session(gateway, env)
        if session is None:
            print(f"Шлюз {host}: вход не выполнен. Выполните `uv run tekti login`.")
            return 1
        state = "истекла" if session.expired() else "действует до"
        print(f"Шлюз {host}: вы вошли как {session.login}; сессия {state} {_local(session.expires_at)}.")
        return 1 if session.expired() else 0

    if args.command == "logout":
        removed = gateway_auth.delete_session(gateway, env)
        print(f"Сессия для шлюза {host} удалена." if removed else f"Для шлюза {host} сохранённой сессии нет.")
        return 0

    client_id = github_client_id(env)
    if not client_id:
        print(
            "Вход через GitHub ещё не настроен автором (нет client id OAuth-приложения). "
            "Пока можно запускать проверку со своими ключами (см. README).",
            file=sys.stderr,
        )
        return 2
    try:
        return _login(gateway, client_id, env, host or gateway)
    except gateway_auth.LoginError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nВход прерван.", file=sys.stderr)
        return 130


def _login(gateway: str, client_id: str, env, host: str) -> int:
    from tekti.llm_backends import gateway_auth

    device = gateway_auth.start_device_flow(client_id)
    print(f"Откройте {device.verification_uri} и введите код: {device.user_code}")
    print(f"Код действует {max(1, device.expires_in // 60)} мин. Жду подтверждения в браузере...")
    import contextlib
    import webbrowser

    with contextlib.suppress(Exception):  # no browser is fine: the address is printed
        webbrowser.open(device.verification_uri)
    github_token = gateway_auth.poll_for_token(client_id, device)
    session_token, login, expires_at = gateway_auth.exchange_for_session(gateway, github_token)
    del github_token  # used once for the exchange, never stored
    session = gateway_auth.save_session(gateway, session_token, login, expires_at, env)
    print(f"Вход выполнен: {session.login} (шлюз {host}); сессия действует до {_local(session.expires_at)}.")
    return 0


def _local(moment: datetime) -> str:
    return moment.astimezone().strftime("%d.%m.%Y %H:%M %Z").strip()


if __name__ == "__main__":
    sys.exit(main())
