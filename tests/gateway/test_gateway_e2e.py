"""End to end: the handler's real HTTP client against a tiny local server standing in for the vendors."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest


class FakeVendor(BaseHTTPRequestHandler):
    """Plays OpenAI/Anthropic/Google and, for GET /user, GitHub."""

    seen: ClassVar[list[dict]] = []
    github_seen: ClassVar[list[dict]] = []

    def do_GET(self):
        headers = {k.lower(): v for k, v in self.headers.items()}
        FakeVendor.github_seen.append({"path": self.path, "headers": headers})
        if self.path != "/user":
            self._send(404, "application/json", b'{"message":"Not Found"}')
        elif not headers.get("user-agent") or headers.get("accept") != "application/vnd.github+json":
            self._send(403, "application/json", b'{"message":"missing user agent"}')
        elif headers.get("authorization") != "Bearer gho_e2e_token":
            self._send(401, "application/json", b'{"message":"Bad credentials"}')
        else:
            self._send(200, "application/json", json.dumps({"login": "Jury-Member", "id": 7, "type": "User"}).encode())

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        FakeVendor.seen.append(
            {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
        )
        if self.path.startswith("/v1/messages"):
            if self.headers.get("x-api-key") != "vendor-anthropic-key":
                self._send(401, "application/json", b'{"type":"error","error":{"type":"authentication_error"}}')
                return
            payload = json.dumps(
                {
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "pong"}],
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                }
            ).encode()
            self._send(200, "application/json", payload, chunked=True)
        elif ":streamGenerateContent" in self.path:
            sse = b'data: {"usageMetadata":{"promptTokenCount":2,"totalTokenCount":5}}\r\n\r\n'
            self._send(200, "text/event-stream", sse)
        else:
            self._send(404, "application/json", b'{"error":"nope"}')

    def _send(self, status, ctype, body, chunked=False):
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("x-vendor", "fake")
        if chunked:
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for i in range(0, len(body), 7):
                piece = body[i : i + 7]
                self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def vendor_server():
    FakeVendor.seen = []
    FakeVendor.github_seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVendor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_end_to_end_against_local_vendor(gw, vendor_server, monkeypatch, capsys):
    monkeypatch.setattr(gw.module, "_send_upstream", gw.real_send)
    monkeypatch.setenv("UPSTREAM_ANTHROPIC_URL", vendor_server)
    monkeypatch.setenv("UPSTREAM_GOOGLE_URL", vendor_server)
    monkeypatch.setenv("GITHUB_API_URL", vendor_server)
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "someone-else,JURY-member")
    gw.secrets.values["arn:test:vendor"] = json.dumps(
        {"openai": "REPLACE_ME", "anthropic": "vendor-anthropic-key", "google": "vendor-google-key"}
    )
    capsys.readouterr()

    # sign in: the gateway asks the fake GitHub who owns the token
    assert gw.sign_in("gho_wrong")["statusCode"] == 401
    signed = gw.sign_in("gho_e2e_token")
    assert signed["statusCode"] == 200, signed
    session = json.loads(signed["body"])["session"]
    assert [x["path"] for x in FakeVendor.github_seen] == ["/user", "/user"]
    gh_headers = FakeVendor.github_seen[1]["headers"]
    assert gh_headers["x-github-api-version"] == "2026-03-10"
    assert gh_headers["user-agent"] == "tekti-gateway"

    # the Anthropic SDK sends the key as a bearer token
    resp = gw.call(
        path="/anthropic/v1/messages",
        body={"model": "claude-test-1", "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]},
        headers={
            "Authorization": f"Bearer {session}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    assert resp["statusCode"] == 200, resp
    assert json.loads(resp["body"])["content"][0]["text"] == "pong"
    assert resp["headers"]["x-vendor"] == "fake"
    assert "transfer-encoding" not in resp["headers"]
    [seen] = FakeVendor.seen
    assert seen["path"] == "/v1/messages"
    assert seen["headers"]["x-api-key"] == "vendor-anthropic-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert session not in json.dumps(seen["headers"])
    assert "authorization" not in seen["headers"]

    resp = gw.call(
        path="/google/v1beta/models/gemini-test-1:streamGenerateContent",
        query=f"alt=sse&key={gw.tokens['owner']}",
        body={"contents": [{"parts": [{"text": "hi"}]}]},
    )
    assert resp["statusCode"] == 200
    assert resp["headers"]["content-type"] == "text/event-stream"
    assert FakeVendor.seen[1]["path"] == "/v1beta/models/gemini-test-1:streamGenerateContent?alt=sse"
    assert FakeVendor.seen[1]["headers"]["x-goog-api-key"] == "vendor-google-key"

    # OpenAI key is still the placeholder: refused before anything is sent
    resp = gw.call(headers={"Authorization": f"Bearer {gw.tokens['owner']}"})
    assert resp["statusCode"] == 503
    assert len(FakeVendor.seen) == 2

    out = capsys.readouterr().out
    assert session not in out
    assert "gho_e2e_token" not in out
    lines = [json.loads(x) for x in out.splitlines()][2:]  # skip the two sign-in lines
    assert [(x["token"], x["vendor"], x["status"]) for x in lines] == [
        ("gh:jury-member", "anthropic", 200),
        ("owner", "google", 200),
        ("owner", "openai", 503),
    ]
    assert lines[0]["usage"] == {"input_tokens": 4, "output_tokens": 1}
    assert lines[1]["usage"] == {"promptTokenCount": 2, "totalTokenCount": 5}
    assert gw.dynamo.counts["gh:jury-member"] == 1
    assert gw.dynamo.counts["gh-total"] == 1
    assert gw.dynamo.counts["owner"] == 1


def test_vendor_error_passes_through_end_to_end(gw, vendor_server, monkeypatch):
    monkeypatch.setattr(gw.module, "_send_upstream", gw.real_send)
    monkeypatch.setenv("UPSTREAM_ANTHROPIC_URL", vendor_server)  # vendor key in fixture is not the one it expects
    resp = gw.call(
        path="/anthropic/v1/messages", body={"model": "claude-test-1"}, headers={"x-api-key": gw.tokens["owner"]}
    )
    assert resp["statusCode"] == 401
    assert json.loads(resp["body"]) == {"type": "error", "error": {"type": "authentication_error"}}


def test_unreachable_vendor_is_502(gw, monkeypatch):
    monkeypatch.setattr(gw.module, "_send_upstream", gw.real_send)
    monkeypatch.setenv("UPSTREAM_OPENAI_URL", "http://127.0.0.1:9")  # discard port, nothing listens
    resp = gw.call(headers={"Authorization": f"Bearer {gw.tokens['owner']}"})
    assert resp["statusCode"] == 502


def test_github_outage_end_to_end(gw, monkeypatch):
    monkeypatch.setattr(gw.module, "_send_upstream", gw.real_send)
    monkeypatch.setenv("GITHUB_API_URL", "http://127.0.0.1:9")  # nothing listens
    resp = gw.sign_in("gho_anything")
    assert resp["statusCode"] == 502
    assert json.loads(resp["body"])["error"]["code"] == "github_unavailable"
