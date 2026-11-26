"""A local server playing GitHub's device flow endpoints and the gateway's POST /auth/github.

`poll_script` is the list of answers to successive token polls, e.g.
["authorization_pending", "slow_down", "ok"]; "ok" returns an access token. Every request is recorded
(path and parsed body) so tests can check the parameters the client sent.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

GITHUB_TOKEN = "gho_fake_github_token_123"
SESSION = "tks1.eyJsb2dpbiI6Im9jdG8iLCJpZCI6MSwiZXhwIjo0MTAyNDQ0ODAwfQ.c2ln"


@dataclass
class FakeGitHub:
    poll_script: list[str] = field(default_factory=lambda: ["ok"])
    interval: int = 1
    expires_in: int = 900
    auth_status: int = 200
    auth_body: dict[str, Any] | None = None
    seen: list[tuple[str, dict[str, Any], dict[str, str]]] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)).decode()
                if self.headers.get("Content-Type", "").startswith("application/json"):
                    body = json.loads(raw or "{}")
                else:
                    body = {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}
                fake.seen.append((self.path, body, {k.lower(): v for k, v in self.headers.items()}))
                if self.path == "/login/device/code":
                    self._send(
                        200,
                        {
                            "device_code": "dev-code-1",
                            "user_code": "WDJB-MJHT",
                            "verification_uri": "https://github.com/login/device",
                            "expires_in": fake.expires_in,
                            "interval": fake.interval,
                        },
                    )
                elif self.path == "/login/oauth/access_token":
                    step = fake.poll_script.pop(0) if fake.poll_script else "authorization_pending"
                    if step == "ok":
                        self._send(200, {"access_token": GITHUB_TOKEN, "token_type": "bearer", "scope": ""})
                    elif step == "slow_down":
                        self._send(200, {"error": "slow_down", "interval": fake.interval + 5})
                    else:
                        self._send(200, {"error": step, "error_description": step})
                elif self.path == "/auth/github":
                    default = {"session": SESSION, "login": "octo", "expires_at": "2099-12-31T00:00:00Z"}
                    self._send(fake.auth_status, fake.auth_body if fake.auth_body is not None else default)
                else:
                    self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()
