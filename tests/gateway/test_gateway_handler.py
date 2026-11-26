"""Unit tests for gateway/handler.py against the contract in gateway/CONTRACT.md."""

from __future__ import annotations

import base64
import json

import pytest

TOKEN_HEADERS = ("authorization", "x-api-key", "x-goog-api-key")


def error_code(resp):
    body = json.loads(resp["body"])
    assert body["error"]["type"] == "tekti_gateway"
    return body["error"]["code"]


def log_lines(capsys):
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------- health and routing


def test_health_needs_no_token(gw):
    resp = gw.call(method="GET", path="/health", body=b"")
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"]) == {"ok": True}
    assert gw.upstream.requests == []


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/openai/v1/models",
        "/openai/v1/embeddings",
        "/openai/v1/chat/completions/extra",
        "/anthropic/v1/messages/count_tokens",
        "/google/v1beta/models/gemini-test-1:embedContent",
        "/google/v1/models/gemini-test-1:generateContent",
        "/azure/v1/chat/completions",
        "/openai/../anthropic/v1/messages",
    ],
)
def test_unknown_paths_are_404(gw, path):
    resp = gw.call(path=path, headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 404
    assert error_code(resp) == "not_found"
    assert gw.upstream.requests == []


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
def test_wrong_method_is_405(gw, method):
    resp = gw.call(method=method, headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 405
    assert error_code(resp) == "method_not_allowed"
    assert gw.upstream.requests == []


# ---------------------------------------------------------------------------- authentication


@pytest.mark.parametrize("kind", ["owner", "session"])
@pytest.mark.parametrize("place", ["authorization", "x-api-key", "x-goog-api-key", "query"])
def test_token_accepted_from_each_place_and_stripped(gw, place, kind):
    token = gw.tokens["owner"] if kind == "owner" else gw.session_for()
    headers = {"content-type": "application/json"}
    query = "alt=sse"
    if place == "authorization":
        headers["Authorization"] = f"Bearer {token}"
    elif place == "query":
        query = f"key={token}&alt=sse"
    else:
        headers[place] = token
    resp = gw.call(
        path="/google/v1beta/models/gemini-test-1:streamGenerateContent",
        headers=headers,
        query=query,
        body={"contents": []},
    )
    assert resp["statusCode"] == 200
    [sent] = gw.upstream.requests
    assert token not in json.dumps(sent["headers"])
    assert token not in sent["url"]
    assert "key=" not in sent["url"]
    assert sent["url"].endswith("?alt=sse")
    assert sent["headers"]["x-goog-api-key"] == gw.vendor_keys["google"]
    assert "authorization" not in sent["headers"]
    assert "x-api-key" not in sent["headers"]


@pytest.mark.parametrize(
    "headers,query",
    [
        ({}, ""),
        ({"Authorization": "Bearer wrong-token-wrong-token-wrong"}, ""),
        ({"x-api-key": "nope"}, ""),
        ({"x-goog-api-key": ""}, ""),
        ({}, "key=nope"),
        ({"Authorization": "Basic Zm9vOmJhcg=="}, ""),
    ],
)
def test_missing_or_wrong_token_is_401(gw, headers, query):
    resp = gw.call(headers=headers, query=query)
    assert resp["statusCode"] == 401
    assert error_code(resp) in {"missing_token", "invalid_token"}
    assert gw.upstream.requests == []
    assert gw.dynamo.calls == 0


def test_conflicting_tokens_are_401(gw):
    headers = {"Authorization": f"Bearer {gw.tokens['owner']}", "x-api-key": gw.session_for()}
    resp = gw.call(headers=headers)
    assert resp["statusCode"] == 401
    assert gw.upstream.requests == []


def test_same_token_in_two_places_is_fine(gw):
    t = gw.tokens["owner"]
    resp = gw.call(headers={"Authorization": f"Bearer {t}", "x-api-key": t})
    assert resp["statusCode"] == 200


def test_short_configured_token_is_never_accepted(gw):
    gw.secrets.values["arn:test:owner"] = "short"
    resp = gw.call(headers=bearer("short"))
    assert resp["statusCode"] == 401
    assert gw.upstream.requests == []


# ---------------------------------------------------------------------------- vendor mapping


@pytest.mark.parametrize(
    "path,body,expected_url,cred_header,cred_value_fmt",
    [
        (
            "/openai/v1/chat/completions",
            {"model": "gpt-test-1"},
            "https://api.openai.com/v1/chat/completions",
            "authorization",
            "Bearer {openai}",
        ),
        (
            "/openai/v1/responses",
            {"model": "gpt-test-1"},
            "https://api.openai.com/v1/responses",
            "authorization",
            "Bearer {openai}",
        ),
        (
            "/anthropic/v1/messages",
            {"model": "claude-test-1"},
            "https://api.anthropic.com/v1/messages",
            "x-api-key",
            "{anthropic}",
        ),
        (
            "/google/v1beta/models/gemini-test-1:generateContent",
            {"contents": []},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test-1:generateContent",
            "x-goog-api-key",
            "{google}",
        ),
        (
            "/google/v1beta/models/gemini-test-1:streamGenerateContent",
            {"contents": []},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test-1:streamGenerateContent",
            "x-goog-api-key",
            "{google}",
        ),
        (
            "/google/v1beta/openai/chat/completions",
            {"model": "gemini-test-1"},
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "authorization",
            "Bearer {google}",
        ),
    ],
)
def test_routes_map_to_vendor_with_vendor_credential(gw, path, body, expected_url, cred_header, cred_value_fmt):
    resp = gw.call(path=path, body=body, headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 200
    [sent] = gw.upstream.requests
    assert sent["method"] == "POST"
    assert sent["url"] == expected_url
    assert sent["headers"][cred_header] == cred_value_fmt.format(**gw.vendor_keys)
    assert sent["body"] == json.dumps(body).encode()
    other_vendor_keys = [v for k, v in gw.vendor_keys.items() if k != path.split("/")[1]]
    assert not any(k in json.dumps(sent["headers"]) for k in other_vendor_keys)


def test_upstream_base_overridable_by_env(gw, monkeypatch):
    monkeypatch.setenv("UPSTREAM_ANTHROPIC_URL", "http://127.0.0.1:9999/")
    gw.call(path="/anthropic/v1/messages", body={"model": "claude-test-1"}, headers={"x-api-key": gw.tokens["owner"]})
    assert gw.upstream.requests[0]["url"] == "http://127.0.0.1:9999/v1/messages"


def test_headers_forwarded_and_dropped(gw):
    incoming = {
        "x-api-key": gw.tokens["owner"],
        "content-type": "application/json",
        "accept": "text/event-stream",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "some-beta-2025",
        "user-agent": "Anthropic/Python 1.0",
        "x-stainless-retry-count": "0",
        "openai-organization": "org-should-not-pass",
        "openai-project": "proj-should-not-pass",
        "connection": "keep-alive",
        "keep-alive": "timeout=5",
        "transfer-encoding": "chunked",
        "te": "trailers",
        "upgrade": "h2c",
        "host": "abc.lambda-url.eu-central-1.on.aws",
        "x-forwarded-for": "1.2.3.4",
        "x-forwarded-proto": "https",
        "x-forwarded-port": "443",
        "x-amzn-trace-id": "Root=1-abc",
        "x-amzn-tls-version": "TLSv1.3",
        "forwarded": "for=1.2.3.4",
        "cookie": "a=b",
        "accept-encoding": "gzip, br",
    }
    gw.call(path="/anthropic/v1/messages", body={"model": "claude-test-1"}, headers=incoming)
    [sent] = gw.upstream.requests
    h = sent["headers"]
    for keep in (
        "content-type",
        "accept",
        "anthropic-version",
        "anthropic-beta",
        "user-agent",
        "x-stainless-retry-count",
    ):
        assert h[keep] == incoming[keep], keep
    for drop in (
        "openai-organization",
        "openai-project",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "upgrade",
        "host",
        "x-forwarded-for",
        "x-forwarded-proto",
        "x-forwarded-port",
        "x-amzn-trace-id",
        "x-amzn-tls-version",
        "forwarded",
        "cookie",
        "x-goog-api-key",
        "authorization",
    ):
        assert drop not in h, drop
    assert h["accept-encoding"] == "identity"
    assert h["x-api-key"] == gw.vendor_keys["anthropic"]
    assert h["content-length"] == str(len(sent["body"]))


def test_openai_organization_header_not_forwarded(gw):
    gw.call(headers={**bearer(gw.tokens["owner"]), "OpenAI-Organization": "org-x"})
    assert "openai-organization" not in gw.upstream.requests[0]["headers"]


def test_query_string_preserved_except_key(gw):
    t = gw.tokens["owner"]
    gw.call(
        path="/google/v1beta/models/gemini-test-1:streamGenerateContent",
        body={"contents": []},
        query=f"alt=sse&key={t}&foo=a%20b&key=other",
        headers={"x-goog-api-key": t},
    )
    # two different `key` values -> refused
    assert gw.upstream.requests == []
    gw.call(
        path="/google/v1beta/models/gemini-test-1:streamGenerateContent",
        body={"contents": []},
        query=f"alt=sse&key={t}&foo=a%20b&$fields=x",
    )
    assert gw.upstream.requests[0]["url"].endswith("?alt=sse&foo=a%20b&$fields=x")


def test_encoded_colon_in_google_path(gw):
    resp = gw.call(
        path="/google/v1beta/models/gemini-test-1%3AgenerateContent",
        body={"contents": []},
        headers={"x-goog-api-key": gw.tokens["owner"]},
    )
    assert resp["statusCode"] == 200
    assert gw.upstream.requests[0]["url"].endswith("/v1beta/models/gemini-test-1:generateContent")


# ---------------------------------------------------------------------------- limits


@pytest.mark.parametrize(
    "path,body",
    [
        ("/openai/v1/chat/completions", {"model": "gpt-4o"}),
        ("/openai/v1/responses", {"model": "o3-pro"}),
        ("/anthropic/v1/messages", {"model": "claude-opus-everything"}),
        ("/google/v1beta/openai/chat/completions", {"model": "gemini-ultra"}),
        ("/google/v1beta/models/gemini-ultra:generateContent", {"contents": []}),
        ("/google/v1beta/models/gemini-ultra:streamGenerateContent", {"contents": []}),
        ("/openai/v1/chat/completions", {"messages": []}),
    ],
)
def test_model_not_on_allowlist_is_403(gw, path, body):
    resp = gw.call(path=path, body=body, headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 403
    assert error_code(resp) == "model_not_allowed"
    assert gw.upstream.requests == []
    assert gw.dynamo.calls == 0


def test_google_models_prefix_in_body_is_ignored(gw):
    resp = gw.call(
        path="/google/v1beta/openai/chat/completions",
        body={"model": "models/gemini-test-1"},
        headers=bearer(gw.tokens["owner"]),
    )
    assert resp["statusCode"] == 200


def test_invalid_json_body_is_400(gw):
    resp = gw.call(body=b"not json", headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 400
    assert gw.upstream.requests == []


def test_owner_cap_reached_is_429_and_not_forwarded(gw, monkeypatch):
    monkeypatch.setenv("OWNER_CAP", "2")
    h = bearer(gw.tokens["owner"])
    assert gw.call(headers=h)["statusCode"] == 200
    assert gw.call(headers=h)["statusCode"] == 200
    resp = gw.call(headers=h)
    assert resp["statusCode"] == 429
    assert error_code(resp) == "request_cap_reached"
    assert resp["headers"]["x-should-retry"] == "false"
    assert "2" in json.loads(resp["body"])["error"]["message"]
    assert len(gw.upstream.requests) == 2
    # signed-in users have their own counters
    assert gw.call(headers=bearer(gw.session_for()))["statusCode"] == 200
    assert gw.dynamo.counts["owner"] == 2


def test_cap_zero_blocks_without_touching_table(gw, monkeypatch):
    monkeypatch.setenv("OWNER_CAP", "0")
    resp = gw.call(headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 429
    assert gw.dynamo.calls == 0
    assert gw.upstream.requests == []


def test_body_too_large_is_413(gw):
    big = json.dumps({"model": "gpt-test-1", "pad": "x" * (1024 * 1024)})
    resp = gw.call(body=big, headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 413
    assert error_code(resp) == "body_too_large"
    assert gw.upstream.requests == []
    assert gw.dynamo.calls == 0


def test_body_just_under_limit_is_forwarded(gw):
    prefix = '{"model":"gpt-test-1","pad":"'
    body = prefix + "x" * (1024 * 1024 - len(prefix) - 2) + '"}'
    assert len(body) == 1024 * 1024
    assert gw.call(body=body, headers=bearer(gw.tokens["owner"]))["statusCode"] == 200


def test_base64_body_is_decoded_and_forwarded(gw):
    body = {"model": "claude-test-1", "messages": [{"role": "user", "content": "héllo"}]}
    raw = json.dumps(body, ensure_ascii=False).encode()
    resp = gw.call(path="/anthropic/v1/messages", body=raw, b64=True, headers={"x-api-key": gw.tokens["owner"]})
    assert resp["statusCode"] == 200
    assert gw.upstream.requests[0]["body"] == raw


def test_vendor_key_placeholder_is_503(gw):
    keys = dict(gw.vendor_keys, anthropic="REPLACE_ME")
    gw.secrets.values["arn:test:vendor"] = json.dumps(keys)
    resp = gw.call(
        path="/anthropic/v1/messages", body={"model": "claude-test-1"}, headers={"x-api-key": gw.tokens["owner"]}
    )
    assert resp["statusCode"] == 503
    assert error_code(resp) == "vendor_not_configured"
    assert gw.upstream.requests == []
    assert gw.dynamo.calls == 0
    # other vendors still work
    assert gw.call(headers=bearer(gw.tokens["owner"]))["statusCode"] == 200


def test_secrets_are_cached(gw):
    for _ in range(3):
        gw.call(headers=bearer(gw.tokens["owner"]))
    assert gw.secrets.calls == 2  # owner token, vendor keys: once each


def test_secrets_refetched_after_ttl(gw, monkeypatch):
    monkeypatch.setenv("SECRET_CACHE_SECONDS", "0")
    old = gw.tokens["owner"]
    assert gw.call(headers=bearer(old))["statusCode"] == 200
    gw.secrets.values["arn:test:owner"] = "a-brand-new-rotated-token-value-1234567890"
    assert gw.call(headers=bearer(old))["statusCode"] == 401


# ---------------------------------------------------------------------------- pass-through


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 529])
def test_vendor_error_passes_through_unchanged(gw, status):
    vendor_body = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}).encode()
    gw.upstream.status = status
    gw.upstream.body = vendor_body
    gw.upstream.headers = [("content-type", "application/json"), ("retry-after", "3"), ("request-id", "r1")]
    resp = gw.call(
        path="/anthropic/v1/messages", body={"model": "claude-test-1"}, headers={"x-api-key": gw.tokens["owner"]}
    )
    assert resp["statusCode"] == status
    assert resp["body"].encode() == vendor_body
    assert resp["headers"]["retry-after"] == "3"
    assert resp["headers"]["request-id"] == "r1"


def test_sse_body_passes_through_with_content_type(gw):
    sse = (
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":25}}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":15}}\n\n'
    )
    gw.upstream.body = sse
    gw.upstream.headers = [
        ("Content-Type", "text/event-stream; charset=utf-8"),
        ("Transfer-Encoding", "chunked"),
        ("Connection", "keep-alive"),
        ("Content-Encoding", "gzip"),
        ("Set-Cookie", "x=y"),
    ]
    resp = gw.call(
        path="/anthropic/v1/messages",
        body={"model": "claude-test-1", "stream": True},
        headers={"x-api-key": gw.tokens["owner"]},
    )
    assert resp["statusCode"] == 200
    assert resp["body"].encode() == sse
    assert resp["isBase64Encoded"] is False
    assert resp["headers"] == {"content-type": "text/event-stream; charset=utf-8"}


def test_binary_response_is_base64(gw):
    gw.upstream.body = b"\xff\xfe\x00binary"
    gw.upstream.headers = [("content-type", "application/octet-stream")]
    resp = gw.call(headers=bearer(gw.tokens["owner"]))
    assert resp["isBase64Encoded"] is True
    assert base64.b64decode(resp["body"]) == b"\xff\xfe\x00binary"


def test_oversized_vendor_response_is_502(gw):
    gw.upstream.body = b'{"x":"' + b"y" * (6 * 1024 * 1024) + b'"}'
    resp = gw.call(headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 502
    assert error_code(resp) == "upstream_response_too_large"


def test_upstream_timeout_is_504(gw):
    gw.upstream.raise_exc = TimeoutError("timed out")
    resp = gw.call(headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 504
    assert error_code(resp) == "upstream_timeout"


def test_upstream_unreachable_is_502(gw):
    gw.upstream.raise_exc = ConnectionRefusedError("refused")
    resp = gw.call(headers=bearer(gw.tokens["owner"]))
    assert resp["statusCode"] == 502
    assert error_code(resp) == "upstream_unreachable"


# ---------------------------------------------------------------------------- logging


def test_log_line_has_fields_and_no_secrets_or_body(gw, capsys):
    capsys.readouterr()
    token = gw.session_for(login="Jury-Member")
    capsys.readouterr()
    gw.upstream.body = json.dumps(
        {
            "choices": [{"message": {"content": "SECRET-RESPONSE-TEXT"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
    ).encode()
    gw.call(headers={**bearer(token), "x-custom-header": "SECRET-HEADER-VALUE"}, query="trace=SECRET-QUERY")
    [line] = log_lines(capsys)
    assert line["token"] == "gh:jury-member"
    assert line["vendor"] == "openai"
    assert line["model"] == "gpt-test-1"
    assert line["status"] == 200
    assert isinstance(line["duration_ms"], int)
    assert line["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    assert "ts" in line
    raw = json.dumps(line)
    for secret in (
        token,
        gw.tokens["owner"],
        *gw.vendor_keys.values(),
        "SECRET-PROMPT-TEXT",
        "SECRET-RESPONSE-TEXT",
        "SECRET-HEADER-VALUE",
        "SECRET-QUERY",
    ):
        assert secret not in raw


def test_log_line_usage_from_sse(gw, capsys):
    capsys.readouterr()
    gw.upstream.headers = [("content-type", "text/event-stream")]
    gw.upstream.body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"SECRET-RESPONSE"}]}}],'
        b'"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}\n\n'
        b'data: {"candidates":[],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":9,"totalTokenCount":12}}'
        b"\n\n"
    )
    gw.call(
        path="/google/v1beta/models/gemini-test-1:streamGenerateContent",
        query="alt=sse",
        body={"contents": []},
        headers={"x-goog-api-key": gw.tokens["owner"]},
    )
    [line] = log_lines(capsys)
    assert line["model"] == "gemini-test-1"
    assert line["vendor"] == "google"
    assert line["usage"] == {"promptTokenCount": 3, "candidatesTokenCount": 9, "totalTokenCount": 12}
    assert "SECRET-RESPONSE" not in json.dumps(line)


def test_log_line_for_refused_request(gw, capsys):
    capsys.readouterr()
    gw.call(headers=bearer("not-a-real-token-at-all-xxxxxxxx"))
    [line] = log_lines(capsys)
    assert line["status"] == 401
    assert line["token"] is None
    assert line["error"] == "invalid_token"
    assert "not-a-real-token" not in json.dumps(line)


def test_log_line_never_contains_secrets_on_any_path(gw, capsys):
    capsys.readouterr()
    owner = gw.tokens["owner"]
    gw.call(headers=bearer(owner))
    gw.call(path="/google/v1beta/models/gemini-test-1:generateContent", query=f"key={owner}", body={"contents": []})
    gw.call(body=b"{bad", headers=bearer(owner))
    gw.upstream.raise_exc = ConnectionRefusedError(f"refused {gw.vendor_keys['openai']}")
    gw.call(headers=bearer(owner))
    out = json.dumps(log_lines(capsys))
    for secret in (owner, *gw.vendor_keys.values()):
        assert secret not in out
