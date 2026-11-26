"""`tekti login/logout/whoami`, session storage, gateway credential precedence and early stop on refusals.

All offline: GitHub's device flow and the gateway's /auth/github are played by fake_github.py, model
requests by tests/llm_backends/fake_vendor.py.
"""

import asyncio
import json
import os
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fake_github import GITHUB_TOKEN, SESSION, FakeGitHub
from pydantic import BaseModel, SecretStr

from tekti import cli, llm
from tekti.config import GATEWAY_SESSION_EXPIRED, NO_GATEWAY_LOGIN, load_config
from tekti.llm import ModelSpec
from tekti.llm_backends import GatewayRefused, StrandsBackend, gateway_auth, gateway_endpoint
from tekti.llm_backends.gateway_auth import (
    DeviceCode,
    LoginError,
    delete_session,
    exchange_for_session,
    load_session,
    poll_for_token,
    save_session,
    session_file,
    start_device_flow,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_backends"))
from fake_vendor import FakeVendor

GW = "https://gw.example.on.aws"
FUTURE = (datetime.now(UTC) + timedelta(hours=20)).isoformat()
PAST = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated config dir and working dir (so no real .env or session is read)."""
    cfg_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))
    monkeypatch.chdir(tmp_path)
    for var in ("TEKTI_GATEWAY_URL", "TEKTI_GATEWAY_TOKEN", "TEKTI_GITHUB_CLIENT_ID", "TEKTI_PROVIDER", "TEKTI_PANEL"):
        monkeypatch.delenv(var, raising=False)
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return {"XDG_CONFIG_HOME": str(cfg_home)}


# ----------------------------------------------------------------------------- device flow


def test_device_flow_pending_slow_down_success():
    slept = []
    with FakeGitHub(poll_script=["authorization_pending", "slow_down", "ok"], interval=1) as gh:
        device = start_device_flow("client-123", github_base=gh.url)
        token = poll_for_token("client-123", device, github_base=gh.url, sleep=slept.append)
    assert token.get_secret_value() == GITHUB_TOKEN
    assert slept == [1, 1, 6]  # slow_down: the new interval (old + 5) from GitHub's answer
    (code_path, code_body, code_headers), *polls = gh.seen
    assert code_path == "/login/device/code" and code_body == {"client_id": "client-123"}  # no scope
    assert code_headers["accept"] == "application/json"
    assert [p for p, _, _ in polls] == ["/login/oauth/access_token"] * 3
    assert polls[0][1] == {
        "client_id": "client-123",
        "device_code": "dev-code-1",
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    assert all("client_secret" not in body for _, body, _ in gh.seen)
    assert device.user_code == "WDJB-MJHT" and device.verification_uri == "https://github.com/login/device"


@pytest.mark.parametrize(
    ("step", "words"),
    [
        ("access_denied", "отменён"),
        ("expired_token", "истёк"),
        ("device_flow_disabled", "Device Flow"),
        ("incorrect_client_credentials", "client id"),
    ],
)
def test_device_flow_failures(step, words):
    with FakeGitHub(poll_script=["authorization_pending", step]) as gh:
        device = start_device_flow("c", github_base=gh.url)
        with pytest.raises(LoginError, match=words):
            poll_for_token("c", device, github_base=gh.url, sleep=lambda s: None)


def test_device_flow_gives_up_when_the_code_expires():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    device = DeviceCode("d", "U-1", "https://github.com/login/device", expires_in=10, interval=5)
    with FakeGitHub(poll_script=["authorization_pending"] * 10) as gh, pytest.raises(LoginError, match="истёк"):
        poll_for_token("c", device, github_base=gh.url, sleep=sleep, clock=lambda: now[0])
    assert len(gh.seen) == 2  # polls at t=5 and t=10, gives up at t=15


# ----------------------------------------------------------------------------- gateway exchange


def test_exchange_sends_the_github_token_once():
    with FakeGitHub() as gh:
        session, login, expires = exchange_for_session(gh.url + "/", SecretStr(GITHUB_TOKEN))
    assert (session, login) == (SESSION, "octo") and expires.startswith("2099")
    [(path, body, headers)] = gh.seen
    assert path == "/auth/github" and body == {"github_token": GITHUB_TOKEN}
    assert headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    ("status", "body", "words"),
    [
        (403, {"error": {"type": "tekti_gateway", "code": "not_on_allowlist", "message": "x"}}, "списке допуска"),
        (429, {"error": {"type": "tekti_gateway", "code": "auth_rate_limited", "message": "x"}}, "Слишком много"),
        (500, {}, "HTTP 500"),
    ],
)
def test_exchange_refusals(status, body, words):
    with FakeGitHub(auth_status=status, auth_body=body) as gh, pytest.raises(LoginError, match=words) as info:
        exchange_for_session(gh.url, SecretStr(GITHUB_TOKEN))
    assert GITHUB_TOKEN not in str(info.value)


# ----------------------------------------------------------------------------- storage


def test_session_file_location(monkeypatch):
    assert session_file({"XDG_CONFIG_HOME": "/x"}) == Path("/x/tekti/session.json")
    assert session_file({"HOME": "/home/u"}) == Path("/home/u/.config/tekti/session.json")
    monkeypatch.setattr(gateway_auth.sys, "platform", "win32")
    assert session_file({"APPDATA": "C:/AppData"}) == Path("C:/AppData") / "tekti" / "session.json"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_storage_is_owner_only_and_keyed_by_gateway(home):
    save_session(GW + "/", SESSION, "octo", FUTURE, home)
    save_session("https://OTHER.example", "tks1.other.mac", "cat", FUTURE, home)
    path = session_file(home)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    data = json.loads(path.read_text())
    assert set(data) == {GW, "https://other.example"}
    assert GITHUB_TOKEN not in path.read_text()
    assert load_session(GW, home).login == "octo"
    assert load_session("https://other.example/", home).login == "cat"
    assert delete_session(GW, home) and not delete_session(GW, home)
    assert load_session(GW, home) is None and load_session("https://other.example", home) is not None


def test_expiry_falls_back_to_the_token_payload(home):
    save_session(GW, SESSION, "octo", "", home)  # payload exp = 4102444800 (2100-01-01)
    assert load_session(GW, home).expires_at.year == 2100
    assert SESSION not in repr(load_session(GW, home))


# ----------------------------------------------------------------------------- CLI login / whoami / logout


def test_login_whoami_logout(home, monkeypatch, capsys):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    monkeypatch.setattr(gateway_auth, "sleep_between_polls", lambda s: None)
    with FakeGitHub(poll_script=["authorization_pending", "ok"]) as gh:
        monkeypatch.setattr(gateway_auth, "GITHUB_BASE", gh.url)
        monkeypatch.setenv("TEKTI_GATEWAY_URL", gh.url)  # the fake also plays the gateway's /auth/github
        monkeypatch.setenv("TEKTI_GITHUB_CLIENT_ID", "client-123")
        assert cli.main(["login"]) == 0
        out = capsys.readouterr()
        assert "WDJB-MJHT" in out.out and "https://github.com/login/device" in out.out and "octo" in out.out
        assert opened == ["https://github.com/login/device"]
        stored = session_file(home).read_text()
        assert SESSION in stored and GITHUB_TOKEN not in stored

        assert cli.main(["whoami"]) == 0
        out = capsys.readouterr().out
        assert "octo" in out and "127.0.0.1" in out and SESSION not in out

        assert cli.main(["logout"]) == 0
        assert "удалена" in capsys.readouterr().out
        assert cli.main(["whoami"]) == 1
        assert "tekti login" in capsys.readouterr().out
    for text in (out,):
        assert GITHUB_TOKEN not in text


def test_login_without_client_id_or_gateway(home, monkeypatch, capsys):
    assert cli.main(["login"]) == 2
    assert "TEKTI_GATEWAY_URL" in capsys.readouterr().err
    monkeypatch.setenv("TEKTI_GATEWAY_URL", GW)
    assert cli.main(["login"]) == 2
    assert "не настроен автором" in capsys.readouterr().err


def test_positional_review_form_still_parses(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "review_main", lambda argv: seen.append(("review", argv)) or 0)
    monkeypatch.setattr(cli, "account_main", lambda argv: seen.append(("account", argv)) or 0)
    cli.main(["doc.md", "report.md", "--mode", "fast"])
    cli.main(["whoami"])
    assert seen == [("review", ["doc.md", "report.md", "--mode", "fast"]), ("account", ["whoami"])]
    args = cli.build_review_parser().parse_args(["in.pdf", "out.md", "--run-json", "x.json", "--no-rules"])
    assert (args.input, args.report, args.mode, args.run_json, args.no_rules) == (
        Path("in.pdf"),
        Path("out.md"),
        "board",
        Path("x.json"),
        True,
    )


# ----------------------------------------------------------------------------- config precedence


def test_owner_token_beats_session_beats_error(home):
    with pytest.raises(ValueError) as info:
        load_config(env={**home, "TEKTI_GATEWAY_URL": GW})
    assert str(info.value) == NO_GATEWAY_LOGIN
    save_session(GW, SESSION, "octo", FUTURE, home)
    cfg = load_config(env={**home, "TEKTI_GATEWAY_URL": GW + "/"})
    assert cfg.gateway_auth == "github_session" and cfg.gateway_login == "octo"
    assert cfg.gateway_token.get_secret_value() == SESSION and cfg.provider_mode == "vendors"
    assert SESSION not in repr(cfg) and SESSION not in cfg.model_dump_json()
    owner = load_config(env={**home, "TEKTI_GATEWAY_URL": GW, "TEKTI_GATEWAY_TOKEN": "owner-tok"})
    assert owner.gateway_auth == "owner_token" and owner.gateway_token.get_secret_value() == "owner-tok"
    # an explicit Bedrock run does not need the gateway credential
    assert load_config(env={**home, "TEKTI_GATEWAY_URL": GW, "TEKTI_PROVIDER": "bedrock"}).gateway_token is None


def test_expired_session_asks_for_login(home):
    save_session(GW, SESSION, "octo", PAST, home)
    with pytest.raises(ValueError) as info:
        load_config(env={**home, "TEKTI_GATEWAY_URL": GW})
    assert str(info.value) == GATEWAY_SESSION_EXPIRED


# ----------------------------------------------------------------------------- early stop on gateway refusals


class Out(BaseModel):
    x: int


SPEC = ModelSpec(family="openai", model_id="gpt-6.1-terra", provider="openai")


@pytest.mark.parametrize(
    ("status", "code", "words"),
    [
        (401, "session_expired", "tekti login"),
        (401, "invalid_token", "tekti login"),
        (403, "not_on_allowlist", "списке допуска"),
        (429, "user_cap_reached", "Лимит запросов"),
    ],
)
async def test_refusal_stops_every_pending_call(status, code, words):
    with FakeVendor(answer={"x": 1}, refuse_with=status, refuse_code=code) as fake:
        llm.use_backend(StrandsBackend(vendors={"openai": gateway_endpoint("openai", fake.url, SecretStr("s"))}))
        calls = [llm.ask(SPEC, system="s", user="u", schema=Out, stage="panel") for _ in range(15)]
        with pytest.raises(GatewayRefused) as info:
            await asyncio.gather(*calls)
        first_wave = len(fake.seen)
        with pytest.raises(GatewayRefused):
            await llm.ask(SPEC, system="s", user="u", schema=Out, stage="panel")
    assert words in str(info.value) and "\n" not in str(info.value)
    assert (info.value.status, info.value.code) == (status, code)
    assert len(fake.seen) == first_wave  # later calls stop without a request
    assert not isinstance(info.value, Exception)  # passes the stages' `except LLMError` / `except Exception`


def test_cli_stops_the_run_with_one_line(home, monkeypatch, capsys):
    doc = Path("req.md")
    doc.write_text("# Требования\n\n1. Система должна хранить заказы.\n", encoding="utf-8")
    with FakeVendor(answer={}, refuse_with=401, refuse_code="session_expired") as fake:
        monkeypatch.setenv("TEKTI_GATEWAY_URL", fake.url)
        monkeypatch.setenv("TEKTI_GATEWAY_TOKEN", "owner-tok")
        assert cli.main([str(doc), "report.md"]) == 3
    err = capsys.readouterr().err.strip().splitlines()
    assert err[-1].startswith("Проверка остановлена: Сессия шлюза истекла")
    assert len(fake.seen) == 1 and not Path("report.md").exists()


def test_cli_without_login_fails_at_start(home, monkeypatch, capsys):
    Path("req.md").write_text("# T\n\n1. Система должна работать.\n", encoding="utf-8")
    monkeypatch.setenv("TEKTI_GATEWAY_URL", GW)
    assert cli.main(["req.md", "report.md"]) == 2
    assert capsys.readouterr().err.strip() == NO_GATEWAY_LOGIN
