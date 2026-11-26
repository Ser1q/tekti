"""GitHub sign-in: POST /auth/github, tks1 sessions, login allowlist, caps, expiry (CONTRACT.md)."""

from __future__ import annotations

import base64
import datetime as dt
import json
import time

import pytest
from conftest import GITHUB_BASE


def error_code(resp):
    body = json.loads(resp["body"])
    assert body["error"]["type"] == "tekti_gateway"
    return body["error"]["code"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def log_lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def model_call(gw, token):
    return gw.call(headers=bearer(token))


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


# ---------------------------------------------------------------------------- sign-in


def test_sign_in_issues_a_session(gw, capsys):
    capsys.readouterr()
    gw.upstream.add_github_user("gho_secret_github_token", "Octo-Cat", 42)
    before = int(time.time())
    resp = gw.sign_in("gho_secret_github_token")
    assert resp["statusCode"] == 200
    assert resp["headers"]["cache-control"] == "no-store"
    body = json.loads(resp["body"])
    assert set(body) == {"session", "login", "expires_at"}
    assert body["login"] == "Octo-Cat"
    prefix, payload_b64, sig_b64 = body["session"].split(".")
    assert prefix == "tks1"
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    assert payload["login"] == "octo-cat"
    assert payload["id"] == 42
    assert before + 24 * 3600 <= payload["exp"] <= int(time.time()) + 24 * 3600
    expires = dt.datetime.fromisoformat(body["expires_at"])
    assert int(expires.timestamp()) == payload["exp"]
    assert len(base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))) == 32  # HMAC-SHA256

    [gh] = gw.upstream.github_requests
    assert gh["method"] == "GET"
    assert gh["url"] == f"{GITHUB_BASE}/user"
    assert gh["body"] is None
    assert gh["headers"]["authorization"] == "Bearer gho_secret_github_token"
    assert gh["headers"]["accept"] == "application/vnd.github+json"
    assert gh["headers"]["x-github-api-version"] == "2026-03-10"
    assert gh["headers"]["user-agent"]
    assert gw.upstream.requests == []  # no vendor call

    [line] = log_lines(capsys)
    assert line["token"] == "gh:octo-cat"
    assert line["route"] == "/auth/github"
    assert line["status"] == 200
    raw = json.dumps(line)
    assert "gho_secret_github_token" not in raw
    assert body["session"] not in raw
    assert sig_b64 not in raw


def test_session_works_for_model_calls_and_is_stripped(gw, capsys):
    session = gw.session_for("Octo-Cat")
    capsys.readouterr()
    resp = gw.call(
        path="/anthropic/v1/messages",
        body={"model": "claude-test-1"},
        headers={"Authorization": f"Bearer {session}", "anthropic-version": "2023-06-01"},
    )
    assert resp["statusCode"] == 200
    [sent] = gw.upstream.requests
    assert sent["headers"]["x-api-key"] == gw.vendor_keys["anthropic"]
    assert "authorization" not in sent["headers"]
    assert session not in json.dumps(sent["headers"]) + sent["url"]
    assert session.encode() not in sent["body"]
    assert gw.dynamo.counts["gh:octo-cat"] == 1
    assert gw.dynamo.counts["gh-total"] == 1
    [line] = log_lines(capsys)
    assert line["token"] == "gh:octo-cat"
    assert session not in json.dumps(line)


def test_sign_in_does_not_count_against_model_caps(gw):
    gw.session_for("Octo-Cat")
    assert "gh:octo-cat" not in gw.dynamo.counts
    assert "gh-total" not in gw.dynamo.counts


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        {"github_token": ""},
        {"github_token": 5},
        {"token": "x"},
        {"github_token": "x" * 2000},
        {"github_token": "gho_a\r\nx-evil: 1"},
    ],
)
def test_bad_sign_in_body_is_400(gw, body):
    resp = gw.call(path="/auth/github", body=body)
    assert resp["statusCode"] == 400
    assert error_code(resp) == "invalid_body"
    assert gw.upstream.github_requests == []


def test_sign_in_needs_post(gw):
    resp = gw.call(method="GET", path="/auth/github", body=b"")
    assert resp["statusCode"] == 405


def test_sign_in_body_too_large(gw):
    resp = gw.call(path="/auth/github", body=json.dumps({"github_token": "x", "pad": "y" * 20000}))
    assert resp["statusCode"] == 413


# ---------------------------------------------------------------------------- GitHub answers


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_github_rejection_is_401(gw, status):
    gw.upstream.github_users["gho_x"] = (status, b'{"message":"nope"}')
    resp = gw.sign_in("gho_x")
    assert resp["statusCode"] == 401
    assert error_code(resp) == "github_token_rejected"
    assert resp["headers"]["x-should-retry"] == "false"


def test_unknown_github_token_is_401(gw):
    resp = gw.sign_in("gho_not_a_real_one")
    assert resp["statusCode"] == 401
    assert error_code(resp) == "github_token_rejected"


@pytest.mark.parametrize(
    "answer",
    [(500, b"{}"), (503, b"down"), (200, b"not json"), (200, b'{"id": 1}'), (200, b'{"login": "a b", "id": 1}')],
)
def test_github_outage_or_garbage_is_502(gw, answer):
    gw.upstream.github_users["gho_x"] = answer
    resp = gw.sign_in("gho_x")
    assert resp["statusCode"] == 502
    assert error_code(resp) == "github_unavailable"
    assert "x-should-retry" not in resp["headers"]


@pytest.mark.parametrize("exc", [TimeoutError("slow"), ConnectionRefusedError("refused"), OSError("dns")])
def test_github_unreachable_is_502(gw, exc):
    gw.upstream.github_exc = exc
    resp = gw.sign_in("gho_x")
    assert resp["statusCode"] == 502
    assert error_code(resp) == "github_unavailable"


# ---------------------------------------------------------------------------- session verification


def test_tampered_payload_is_401(gw):
    session = gw.session_for("Octo-Cat")
    prefix, payload_b64, sig = session.split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["login"] = "admin"
    forged = f"{prefix}.{b64url(json.dumps(payload).encode())}.{sig}"
    resp = model_call(gw, forged)
    assert resp["statusCode"] == 401
    assert error_code(resp) == "invalid_token"
    assert gw.upstream.requests == []


def test_tampered_signature_is_401(gw):
    session = gw.session_for("Octo-Cat")
    flipped = session[:-2] + ("A" if session[-2] != "A" else "B") + session[-1]
    resp = model_call(gw, flipped)
    assert resp["statusCode"] == 401
    assert error_code(resp) == "invalid_token"


def test_session_signed_with_another_key_is_401(gw):
    session = gw.session_for("Octo-Cat")
    gw.module._SECRET_CACHE.clear()
    gw.secrets.values["arn:test:signing"] = "a-completely-different-signing-key-0123456789abcdef"
    resp = model_call(gw, session)
    assert resp["statusCode"] == 401
    assert error_code(resp) == "invalid_token"


@pytest.mark.parametrize(
    "token",
    [
        "tks1.",
        "tks1.abc",
        "tks1.abc.def.ghi",
        "tks1.!!!.abc",
        "tks1..abc",
        "tks2.eyJsb2dpbiI6ImEifQ.abc",
    ],
)
def test_malformed_session_is_401(gw, token):
    resp = model_call(gw, token)
    assert resp["statusCode"] == 401
    assert error_code(resp) == "invalid_token"
    assert gw.upstream.requests == []


def test_signed_payload_with_wrong_shape_is_401(gw):
    m = gw.module
    for payload in ([1, 2], {"login": "a", "id": "7", "exp": 9999999999}, {"login": "", "id": 1, "exp": 9999999999}):
        signing_input = f"tks1.{m._b64url(json.dumps(payload).encode())}"
        token = f"{signing_input}.{m._b64url(m._sign(signing_input))}"
        resp = model_call(gw, token)
        assert resp["statusCode"] == 401
        assert error_code(resp) == "invalid_token"


def test_expired_session_is_401_session_expired(gw):
    session = gw.module._issue_session("octo-cat", 42, int(time.time()) - 1)
    resp = model_call(gw, session)
    assert resp["statusCode"] == 401
    assert error_code(resp) == "session_expired"
    assert "tekti login" in json.loads(resp["body"])["error"]["message"]
    assert gw.upstream.requests == []


# ---------------------------------------------------------------------------- allowlist


def test_allowlist_at_sign_in_is_case_insensitive(gw, monkeypatch):
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", " alice , BOB ")
    gw.upstream.add_github_user("gho_bob", "Bob", 2)
    gw.upstream.add_github_user("gho_carol", "carol", 3)
    assert gw.sign_in("gho_bob")["statusCode"] == 200
    resp = gw.sign_in("gho_carol")
    assert resp["statusCode"] == 403
    assert error_code(resp) == "not_on_allowlist"
    assert "session" not in json.loads(resp["body"])
    assert not any(k.startswith("signin:carol") for k in gw.dynamo.counts)


def test_allowlist_checked_on_every_request(gw, monkeypatch):
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "bob")
    session = gw.session_for("BOB")
    assert model_call(gw, session)["statusCode"] == 200
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "alice")  # what a redeploy does
    resp = model_call(gw, session)
    assert resp["statusCode"] == 403
    assert error_code(resp) == "not_on_allowlist"
    assert len(gw.upstream.requests) == 1
    assert gw.dynamo.counts["gh:bob"] == 1


def test_empty_allowlist_means_anyone(gw, monkeypatch):
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "")
    session = gw.session_for("random-stranger")
    assert model_call(gw, session)["statusCode"] == 200


# ---------------------------------------------------------------------------- caps


def test_per_user_cap(gw, monkeypatch):
    monkeypatch.setenv("GITHUB_USER_CAP", "2")
    a = gw.session_for("alice", 1)
    b = gw.session_for("bob", 2)
    assert model_call(gw, a)["statusCode"] == 200
    assert model_call(gw, a)["statusCode"] == 200
    resp = model_call(gw, a)
    assert resp["statusCode"] == 429
    assert error_code(resp) == "request_cap_reached"
    assert resp["headers"]["x-should-retry"] == "false"
    assert model_call(gw, b)["statusCode"] == 200
    assert len(gw.upstream.requests) == 3
    # the refused request consumed neither counter
    assert gw.dynamo.counts == {
        **{k: v for k, v in gw.dynamo.counts.items() if k.startswith("signin:")},
        "gh:alice": 2,
        "gh:bob": 1,
        "gh-total": 3,
    }


def test_total_cap_and_both_or_neither(gw, monkeypatch):
    monkeypatch.setenv("GITHUB_TOTAL_CAP", "3")
    a = gw.session_for("alice", 1)
    b = gw.session_for("bob", 2)
    assert model_call(gw, a)["statusCode"] == 200
    assert model_call(gw, a)["statusCode"] == 200
    assert model_call(gw, b)["statusCode"] == 200
    resp = model_call(gw, b)
    assert resp["statusCode"] == 429
    assert error_code(resp) == "total_cap_reached"
    # bob's own counter was not consumed by the refused request
    assert gw.dynamo.counts["gh:bob"] == 1
    assert gw.dynamo.counts["gh-total"] == 3
    assert len(gw.upstream.requests) == 3
    # the owner is unaffected by GitHub caps
    assert model_call(gw, gw.tokens["owner"])["statusCode"] == 200


def test_user_at_cap_does_not_consume_total(gw, monkeypatch):
    monkeypatch.setenv("GITHUB_USER_CAP", "1")
    a = gw.session_for("alice", 1)
    assert model_call(gw, a)["statusCode"] == 200
    assert model_call(gw, a)["statusCode"] == 429
    assert gw.dynamo.counts["gh-total"] == 1


def test_cancellation_reasons_from_message_only(gw, monkeypatch):
    monkeypatch.setenv("GITHUB_TOTAL_CAP", "1")
    gw.dynamo.reasons_in_message_only = True
    a = gw.session_for("alice", 1)
    assert model_call(gw, a)["statusCode"] == 200
    resp = model_call(gw, a)
    assert resp["statusCode"] == 429
    assert error_code(resp) == "total_cap_reached"


def test_transaction_conflict_is_retried(gw):
    a = gw.session_for("alice", 1)
    gw.dynamo.conflicts_to_inject = 2
    assert model_call(gw, a)["statusCode"] == 200
    assert gw.dynamo.counts["gh:alice"] == 1
    assert gw.dynamo.counts["gh-total"] == 1


def test_persistent_conflict_is_503_and_nothing_forwarded(gw):
    a = gw.session_for("alice", 1)
    gw.dynamo.conflicts_to_inject = 100
    resp = model_call(gw, a)
    assert resp["statusCode"] == 503
    assert error_code(resp) == "gateway_busy"
    assert "x-should-retry" not in resp["headers"]
    assert gw.upstream.requests == []
    assert "gh:alice" not in gw.dynamo.counts


@pytest.mark.parametrize("var", ["GITHUB_USER_CAP", "GITHUB_TOTAL_CAP"])
def test_zero_cap_blocks_without_touching_table(gw, monkeypatch, var):
    a = gw.session_for("alice", 1)
    monkeypatch.setenv(var, "0")
    calls = gw.dynamo.calls
    resp = model_call(gw, a)
    assert resp["statusCode"] == 429
    assert gw.dynamo.calls == calls


def test_sign_in_daily_limit(gw, monkeypatch):
    monkeypatch.setenv("GITHUB_SIGNIN_DAILY_LIMIT", "3")
    gw.upstream.add_github_user("gho_a", "Alice", 1)
    for _ in range(3):
        assert gw.sign_in("gho_a")["statusCode"] == 200
    resp = gw.sign_in("gho_a")
    assert resp["statusCode"] == 429
    assert error_code(resp) == "signin_limit_reached"
    assert resp["headers"]["x-should-retry"] == "false"
    day = dt.datetime.now(dt.UTC).date().isoformat()
    assert gw.dynamo.counts[f"signin:alice:{day}"] == 3
    # another login has its own limit
    gw.upstream.add_github_user("gho_b", "bob", 2)
    assert gw.sign_in("gho_b")["statusCode"] == 200


# ---------------------------------------------------------------------------- expiry date


def test_access_expiry_date(gw, monkeypatch):
    today = dt.datetime.now(dt.UTC).date()
    session = gw.session_for("alice", 1)
    monkeypatch.setenv("GITHUB_EXPIRES", today.isoformat())  # last day still valid
    assert model_call(gw, session)["statusCode"] == 200
    gw.upstream.add_github_user("gho_a", "alice", 1)
    body = json.loads(gw.sign_in("gho_a")["body"])
    end_of_today = dt.datetime.combine(today + dt.timedelta(days=1), dt.time(), dt.UTC)
    assert dt.datetime.fromisoformat(body["expires_at"]) <= end_of_today

    monkeypatch.setenv("GITHUB_EXPIRES", (today - dt.timedelta(days=1)).isoformat())
    resp = model_call(gw, session)
    assert resp["statusCode"] == 403
    assert error_code(resp) == "token_expired"
    github_calls = len(gw.upstream.github_requests)
    resp = gw.sign_in("gho_a")
    assert resp["statusCode"] == 403
    assert error_code(resp) == "token_expired"
    assert len(gw.upstream.github_requests) == github_calls  # GitHub not even asked
    # the owner token is not affected
    assert model_call(gw, gw.tokens["owner"])["statusCode"] == 200


# ---------------------------------------------------------------------------- owner and headers


def test_owner_token_still_works(gw, capsys):
    capsys.readouterr()
    resp = model_call(gw, gw.tokens["owner"])
    assert resp["statusCode"] == 200
    assert gw.dynamo.counts == {"owner": 1}
    assert gw.dynamo.transactions == 0
    [line] = log_lines(capsys)
    assert line["token"] == "owner"


def test_should_retry_false_on_every_gateway_4xx(gw):
    owner = gw.tokens["owner"]
    responses = [
        gw.call(path="/nope", headers=bearer(owner)),
        gw.call(method="GET", headers=bearer(owner)),
        gw.call(headers={}),
        gw.call(body={"model": "not-allowed"}, headers=bearer(owner)),
        gw.call(body=b"{bad", headers=bearer(owner)),
        gw.call(body=json.dumps({"model": "gpt-test-1", "pad": "x" * 1100000}), headers=bearer(owner)),
    ]
    assert [r["statusCode"] for r in responses] == [404, 405, 401, 403, 400, 413]
    for r in responses:
        assert r["headers"]["x-should-retry"] == "false"
    ok = model_call(gw, owner)
    assert "x-should-retry" not in ok["headers"]


def test_vendor_4xx_does_not_get_should_retry_header(gw):
    gw.upstream.status = 429
    gw.upstream.headers = [("content-type", "application/json")]
    resp = model_call(gw, gw.tokens["owner"])
    assert resp["statusCode"] == 429
    assert "x-should-retry" not in resp["headers"]


def test_no_github_token_or_session_in_any_log_line(gw, capsys):
    capsys.readouterr()
    gw.upstream.add_github_user("gho_TOPSECRET", "alice", 1)
    resp = gw.sign_in("gho_TOPSECRET")
    session = json.loads(resp["body"])["session"]
    model_call(gw, session)
    model_call(gw, session[:-3] + "xyz")
    gw.sign_in("gho_TOPSECRET_BUT_WRONG")
    gw.upstream.github_exc = OSError("gho_TOPSECRET")
    gw.sign_in("gho_TOPSECRET")
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 5
    assert "gho_TOPSECRET" not in out
    assert session not in out
    assert session.split(".")[2] not in out
