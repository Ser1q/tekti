"""Sign-in to the Tekti gateway with GitHub (gateway/CONTRACT.md, "Sign-in with GitHub").

GitHub OAuth device flow, checked against GitHub's docs on 26 Nov 2026
(https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow):

1. `POST https://github.com/login/device/code` with `client_id` (no `scope`: the token then has read-only
   access to public information, which is enough for the gateway's `GET /user`). With
   `Accept: application/json` the answer is JSON: device_code, user_code, verification_uri
   (https://github.com/login/device), expires_in (default 900 s), interval.
2. The user enters user_code at verification_uri.
3. Poll `POST https://github.com/login/oauth/access_token` with client_id, device_code and
   grant_type=urn:ietf:params:oauth:grant-type:device_code, never faster than `interval`.
   Errors: authorization_pending (keep polling), slow_down (add 5 s; the answer carries the new interval),
   expired_token (start again), access_denied (cancelled; the code cannot be reused),
   device_flow_disabled (the OAuth app must have "Enable Device Flow" switched on),
   incorrect_client_credentials, incorrect_device_code, unsupported_grant_type.
   No client secret is used in this flow.

The GitHub token is sent once to `POST <gateway>/auth/github` and then dropped; only the gateway session
is stored, in the user's config directory with owner-only permissions, keyed by gateway URL.
Only the standard library is used for HTTP, always with timeouts. Nothing here prints a token.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

GITHUB_BASE = "https://github.com"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
HTTP_TIMEOUT = 20.0
USER_AGENT = "tekti-cli"


class LoginError(Exception):
    """A sign-in step failed; the message is one Russian sentence for the user."""


@dataclass(frozen=True)
class Session:
    gateway_url: str
    login: str
    expires_at: datetime
    token: SecretStr

    def expired(self, now: datetime | None = None, margin_s: float = 0) -> bool:
        now = now or datetime.now(UTC)
        return (self.expires_at - now).total_seconds() <= margin_s

    def __repr__(self) -> str:
        return (
            f"Session(gateway_url={self.gateway_url!r}, login={self.login!r}, expires_at={self.expires_at.isoformat()})"
        )


# --------------------------------------------------------------------------- storage


def session_file(env: Mapping[str, str] | None = None) -> Path:
    """`$XDG_CONFIG_HOME/tekti/session.json`, else `~/.config/tekti/session.json`; `%APPDATA%\\tekti` on Windows."""
    e = os.environ if env is None else env
    if sys.platform == "win32" and e.get("APPDATA"):
        return Path(e["APPDATA"]) / "tekti" / "session.json"
    base = e.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path(e.get("HOME") or Path.home()) / ".config"
    return root / "tekti" / "session.json"


def gateway_key(url: str) -> str:
    """Sessions are kept per gateway: scheme and host lower-cased, no trailing slash."""
    parts = urllib.parse.urlsplit(url.strip())
    path = parts.path.rstrip("/")
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"


def _read_all(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_all(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    if os.name == "posix":
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _expiry(record: Mapping[str, Any]) -> datetime | None:
    text = record.get("expires_at")
    if isinstance(text, str):
        try:
            value = datetime.fromisoformat(text)
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        except ValueError:
            pass
    token = record.get("session")
    if isinstance(token, str) and token.count(".") == 2:  # tks1.<payload>.<mac>: read exp without verifying
        try:
            payload = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            return datetime.fromtimestamp(int(claims["exp"]), UTC)
        except (ValueError, KeyError, TypeError):
            return None
    return None


def load_session(gateway_url: str, env: Mapping[str, str] | None = None) -> Session | None:
    record = _read_all(session_file(env)).get(gateway_key(gateway_url))
    if not isinstance(record, dict) or not isinstance(record.get("session"), str):
        return None
    expires = _expiry(record)
    if expires is None:
        return None
    return Session(gateway_key(gateway_url), str(record.get("login", "?")), expires, SecretStr(record["session"]))


def save_session(
    gateway_url: str, session: str, login: str, expires_at: str, env: Mapping[str, str] | None = None
) -> Session:
    path = session_file(env)
    data = _read_all(path)
    data[gateway_key(gateway_url)] = {"session": session, "login": login, "expires_at": expires_at}
    _write_all(path, data)
    stored = load_session(gateway_url, env)
    if stored is None:
        raise LoginError("Шлюз вернул сессию без срока действия — сообщите автору.")
    return stored


def delete_session(gateway_url: str, env: Mapping[str, str] | None = None) -> bool:
    path = session_file(env)
    data = _read_all(path)
    if data.pop(gateway_key(gateway_url), None) is None:
        return False
    _write_all(path, data)
    return True


# --------------------------------------------------------------------------- HTTP (standard library only)


def _post(
    url: str,
    *,
    form: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> tuple[int, dict[str, Any]]:
    """POST and parse the answer (JSON, or form-encoded as GitHub sends without Accept). HTTP errors are
    returned, not raised; network errors raise URLError/TimeoutError."""
    if body is not None:
        data, ctype = json.dumps(body).encode(), "application/json"
    else:
        data, ctype = urllib.parse.urlencode(form or {}).encode(), "application/x-www-form-urlencoded"
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Accept": "application/json", "Content-Type": ctype, "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read()
    text = raw.decode("utf-8", "replace")
    try:
        parsed = json.loads(text) if text.strip() else {}
    except ValueError:
        parsed = {k: v[0] for k, v in urllib.parse.parse_qs(text).items()}
    return status, parsed if isinstance(parsed, dict) else {}


# --------------------------------------------------------------------------- device flow


def sleep_between_polls(seconds: float) -> None:
    time.sleep(seconds)


@dataclass(frozen=True)
class DeviceCode:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


def start_device_flow(client_id: str, github_base: str | None = None) -> DeviceCode:
    github_base = github_base or GITHUB_BASE
    try:
        status, data = _post(f"{github_base}/login/device/code", form={"client_id": client_id})
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LoginError(f"GitHub недоступен ({exc}) — проверьте подключение к интернету.") from exc
    if status != 200 or "device_code" not in data:
        raise LoginError(_github_error_text(data.get("error"), status))
    return DeviceCode(
        device_code=str(data["device_code"]),
        user_code=str(data["user_code"]),
        verification_uri=str(data.get("verification_uri") or "https://github.com/login/device"),
        expires_in=int(data.get("expires_in") or 900),
        interval=int(data.get("interval") or 5),
    )


def _github_error_text(code: Any, status: int) -> str:
    return {
        "expired_token": "Код подтверждения истёк — запустите `tekti login` ещё раз.",
        "access_denied": "Вход отменён в браузере — запустите `tekti login` ещё раз, если это было случайно.",
        "device_flow_disabled": "У OAuth-приложения автора не включён Device Flow — сообщите автору.",
        "incorrect_client_credentials": "GitHub не узнал client id приложения — проверьте TEKTI_GITHUB_CLIENT_ID.",
        "incorrect_device_code": "GitHub не принял код устройства — запустите `tekti login` ещё раз.",
        "unsupported_grant_type": "GitHub отклонил запрос входа (unsupported_grant_type) — сообщите автору.",
    }.get(str(code), f"GitHub отклонил вход (HTTP {status}, {code or 'без кода'}).")


def poll_for_token(
    client_id: str,
    device: DeviceCode,
    *,
    github_base: str | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> SecretStr:
    """Poll until the user approves; returns the GitHub access token. Raises LoginError otherwise."""
    github_base = github_base or GITHUB_BASE
    sleep = sleep or sleep_between_polls
    interval = max(1, device.interval)
    deadline = clock() + device.expires_in
    while True:
        sleep(interval)
        if clock() > deadline:
            raise LoginError(_github_error_text("expired_token", 0))
        try:
            status, data = _post(
                f"{github_base}/login/oauth/access_token",
                form={"client_id": client_id, "device_code": device.device_code, "grant_type": DEVICE_GRANT},
            )
        except (urllib.error.URLError, TimeoutError):
            continue  # a network blip while waiting: try again at the next interval
        if data.get("access_token"):
            return SecretStr(str(data["access_token"]))
        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval = int(data.get("interval") or interval + 5)
            continue
        raise LoginError(_github_error_text(error, status))


# --------------------------------------------------------------------------- gateway exchange


def exchange_for_session(gateway_url: str, github_token: SecretStr) -> tuple[str, str, str]:
    """`POST <gateway>/auth/github` -> (session, login, expires_at). The GitHub token is sent once, only here."""
    url = gateway_url.rstrip("/") + "/auth/github"
    try:
        status, data = _post(url, body={"github_token": github_token.get_secret_value()})
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LoginError(f"Шлюз недоступен ({exc}) — проверьте TEKTI_GATEWAY_URL и подключение.") from exc
    if status == 200 and isinstance(data.get("session"), str):
        return data["session"], str(data.get("login", "?")), str(data.get("expires_at", ""))
    err = data.get("error") if isinstance(data.get("error"), dict) else {}
    code = err.get("code")
    if code == "not_on_allowlist":
        raise LoginError("Вашего логина GitHub нет в списке допуска шлюза — попросите автора добавить его.")
    if status == 429:
        raise LoginError("Слишком много входов за сутки — попробуйте позже.")
    if status == 404:
        raise LoginError("По адресу TEKTI_GATEWAY_URL нет входа через GitHub — проверьте адрес шлюза.")
    detail = f"{code}: {err.get('message')}" if code else f"HTTP {status}"
    raise LoginError(f"Шлюз не выдал сессию ({detail}) — сообщите автору.")
