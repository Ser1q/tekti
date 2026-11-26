"""Tekti gateway: a thin HTTPS pass-through to OpenAI, Anthropic and Google.

Runs as an AWS Lambda function behind a function URL (AuthType NONE, buffered
invoke mode). Two kinds of caller:

- the owner, with the `owner` gateway token;
- signed-in GitHub users, with a session (`tks1.<payload>.<hmac>`) that
  POST /auth/github issues after checking a GitHub access token.

It checks the token or session, enforces the GitHub login allowlist, the
access expiry date, the model allowlist, the body size and the request caps,
swaps the credential for the vendor's real key and forwards the request
unchanged. The vendor's response is returned unchanged. See gateway/CONTRACT.md.

Standard library plus the boto3 that the Lambda Python runtime ships; no other
packages. Secrets are read at runtime and cached in memory; they are never
logged. The single log line per request carries no prompt, response, header,
query string, token, session or GitHub token.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import http.client
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.parse

# --------------------------------------------------------------------------- limits

MAX_REQUEST_BODY_BYTES = 1024 * 1024  # contract: requests over 1 MB are refused (413)
# Function URL buffered responses are capped at 6 MB (Lambda counts 1 MB as 1,024 KB).
# The cap applies to the whole response object Lambda returns, so keep room for
# the JSON envelope and headers.
LAMBDA_RESPONSE_LIMIT_BYTES = 6 * 1024 * 1024
RESPONSE_ENVELOPE_RESERVE_BYTES = 32 * 1024
MIN_TOKEN_LENGTH = 20  # refuse to authenticate against a suspiciously short configured token
MIN_SIGNING_KEY_LENGTH = 32
PLACEHOLDER = "REPLACE_ME"

# --------------------------------------------------------------------------- GitHub sign-in

SESSION_PREFIX = "tks1"
SESSION_LIFETIME_SECONDS = 24 * 3600
MAX_AUTH_BODY_BYTES = 16 * 1024
DEFAULT_GITHUB_API = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_TIMEOUT_SECONDS = 10.0
GITHUB_TOTAL_KEY = "gh-total"  # counter item for all signed-in users; GitHub logins never contain ':'
DEFAULT_CAPS = {"owner": 3000, "github_user": 400, "github_total": 4000}
DEFAULT_SIGNIN_DAILY_LIMIT = 10
TRANSACTION_ATTEMPTS = 5

# --------------------------------------------------------------------------- routes

DEFAULT_UPSTREAMS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com",
}
UPSTREAM_ENV = {
    "openai": "UPSTREAM_OPENAI_URL",
    "anthropic": "UPSTREAM_ANTHROPIC_URL",
    "google": "UPSTREAM_GOOGLE_URL",
}

# vendor -> list of (path regex, model source, credential placement)
# model source: "body" (JSON field `model`) or "path" (first regex group).
# credential placement: "bearer" (Authorization), "x-api-key", "x-goog-api-key".
_GOOGLE_MODEL = r"([A-Za-z0-9][A-Za-z0-9._-]*)"
ROUTES: dict[str, list[tuple[re.Pattern[str], str, str]]] = {
    "openai": [
        (re.compile(r"^/v1/chat/completions$"), "body", "bearer"),
        (re.compile(r"^/v1/responses$"), "body", "bearer"),
    ],
    "anthropic": [
        (re.compile(r"^/v1/messages$"), "body", "x-api-key"),
    ],
    "google": [
        (re.compile(r"^/v1beta/models/" + _GOOGLE_MODEL + r":generateContent$"), "path", "x-goog-api-key"),
        (re.compile(r"^/v1beta/models/" + _GOOGLE_MODEL + r":streamGenerateContent$"), "path", "x-goog-api-key"),
        # Google's OpenAI-compatible endpoint takes the key as a bearer token.
        (re.compile(r"^/v1beta/openai/chat/completions$"), "body", "bearer"),
    ],
}

# Request headers copied to the vendor. Everything else (hop-by-hop, host,
# x-forwarded-*, x-amzn-*, cookies, the four token places, openai-organization,
# openai-project, accept-encoding, ...) is dropped.
FORWARD_REQUEST_HEADERS = frozenset(
    {
        "content-type",
        "accept",
        "user-agent",
        "anthropic-version",
        "anthropic-beta",
        "openai-beta",
        "idempotency-key",
        "x-goog-api-client",
    }
)
FORWARD_REQUEST_HEADER_PREFIXES = ("x-stainless-",)  # SDK metadata (language, retry count)

# Response headers never copied back to the client.
DROP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
        "set-cookie",
        "alt-svc",
        "openai-organization",
        "openai-project",
    }
)

# --------------------------------------------------------------------------- state

_SECRET_CACHE: dict[str, tuple[float, str]] = {}
_CLIENTS: dict[str, object] = {}


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _client(service: str):
    client = _CLIENTS.get(service)
    if client is None:
        import boto3  # shipped with the Lambda Python runtime

        client = boto3.client(service)
        _CLIENTS[service] = client
    return client


def _secret(arn_env: str) -> str:
    """Return a secret's string value, cached for SECRET_CACHE_SECONDS."""
    arn = _env(arn_env)
    if not arn:
        raise GatewayError(500, "gateway_misconfigured", "The gateway is not configured.")
    ttl = float(_env("SECRET_CACHE_SECONDS", "120"))
    now = time.monotonic()
    cached = _SECRET_CACHE.get(arn)
    if cached and now - cached[0] < ttl:
        return cached[1]
    try:
        value = _client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]
    except Exception as exc:  # never surface the SDK's message
        raise GatewayError(500, "gateway_misconfigured", "The gateway could not read its configuration.") from exc
    _SECRET_CACHE[arn] = (now, value)
    return value


def _owner_token() -> str:
    return _secret("OWNER_TOKEN_SECRET_ARN").strip()


def _signing_key() -> bytes:
    key = _secret("SESSION_SIGNING_SECRET_ARN").strip()
    if len(key) < MIN_SIGNING_KEY_LENGTH:
        raise GatewayError(500, "gateway_misconfigured", "The gateway's session key is not configured.")
    return key.encode("utf-8")


def _vendor_key(vendor: str) -> str:
    raw = _secret("VENDOR_KEYS_SECRET_ARN")
    try:
        keys = json.loads(raw)
    except ValueError:
        keys = None
    if not isinstance(keys, dict):
        raise GatewayError(503, "vendor_not_configured", f"No key is configured for {vendor} on this gateway.")
    key = keys.get(vendor)
    if not isinstance(key, str) or not key.strip() or key.strip() == PLACEHOLDER:
        raise GatewayError(503, "vendor_not_configured", f"No key is configured for {vendor} on this gateway.")
    return key.strip()


# --------------------------------------------------------------------------- responses


def _error(status: int, code: str, message: str) -> dict:
    body = {"error": {"type": "tekti_gateway", "code": code, "message": message}}
    headers = {"content-type": "application/json"}
    if 400 <= status < 500:
        headers["x-should-retry"] = "false"  # tells the vendor SDKs not to resend a refused request
    return {"statusCode": status, "headers": headers, "body": json.dumps(body), "isBase64Encoded": False}


def _json_response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", "cache-control": "no-store"},
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }


def _log(record: dict) -> None:
    sys.stdout.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- request parsing


def _headers(event: dict) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}


def _body(event: dict) -> bytes:
    raw = event.get("body")
    if raw is None:
        return b""
    if event.get("isBase64Encoded"):
        try:
            return base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise GatewayError(400, "invalid_body", "The request body could not be decoded.") from exc
    return raw.encode("utf-8")


def _split_query(raw_query: str) -> tuple[str, list[str]]:
    """Return (query without `key`, values of `key`)."""
    kept: list[str] = []
    keys: list[str] = []
    for part in raw_query.split("&") if raw_query else []:
        if not part:
            continue
        name, _, value = part.partition("=")
        if urllib.parse.unquote_plus(name) == "key":
            keys.append(urllib.parse.unquote_plus(value))
        else:
            kept.append(part)
    return "&".join(kept), keys


def _presented_tokens(headers: dict[str, str], query_keys: list[str]) -> list[str]:
    presented: list[str] = []
    auth = headers.get("authorization")
    if auth is not None:
        scheme, _, value = auth.strip().partition(" ")
        presented.append(value.strip() if scheme.lower() == "bearer" else "")
    for name in ("x-api-key", "x-goog-api-key"):
        if name in headers:
            presented.append(headers[name].strip())
    presented.extend(k.strip() for k in query_keys)
    return presented


def _authenticate(presented: list[str]) -> dict:
    """Return the caller as {"kind", "name", "login"}, or raise 401.

    A value starting with "tks1." is a GitHub session, verified with the signing key;
    anything else is compared in constant time with the owner token.
    """
    if not presented or not any(presented):
        raise GatewayError(401, "missing_token", "A gateway token is required.")
    candidate = presented[0]
    # If the token appears in several places it must be the same everywhere.
    consistent = all(hmac.compare_digest(p.encode(), candidate.encode()) for p in presented)
    if not consistent:
        raise GatewayError(401, "invalid_token", "The gateway token is not valid.")
    if candidate.startswith(SESSION_PREFIX + "."):
        login = _verify_session(candidate)
        return {"kind": "github", "name": f"gh:{login}", "login": login}
    configured = _owner_token()
    ok = hmac.compare_digest(candidate.encode(), configured.encode())
    if not ok or len(configured) < MIN_TOKEN_LENGTH:
        raise GatewayError(401, "invalid_token", "The gateway token is not valid.")
    return {"kind": "owner", "name": "owner", "login": None}


# --------------------------------------------------------------------------- sessions


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    if not text or not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(signing_input: str) -> bytes:
    return hmac.new(_signing_key(), signing_input.encode("ascii"), hashlib.sha256).digest()


def _issue_session(login: str, user_id: int, exp: int) -> str:
    payload = json.dumps({"login": login, "id": user_id, "exp": exp}, separators=(",", ":"), sort_keys=True)
    signing_input = f"{SESSION_PREFIX}.{_b64url(payload.encode('utf-8'))}"
    return f"{signing_input}.{_b64url(_sign(signing_input))}"


def _verify_session(session: str) -> str:
    """Return the session's lower-cased login, or raise 401 invalid_token / session_expired."""
    invalid = GatewayError(401, "invalid_token", "The session is not valid. Run `tekti login` again.")
    parts = session.split(".")
    if len(parts) != 3 or parts[0] != SESSION_PREFIX:
        raise invalid
    try:
        signature = _b64url_decode(parts[2])
    except (ValueError, binascii.Error) as exc:
        raise invalid from exc
    expected = _sign(f"{SESSION_PREFIX}.{parts[1]}")
    if not hmac.compare_digest(signature, expected):
        raise invalid
    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except (ValueError, binascii.Error) as exc:
        raise invalid from exc
    if not isinstance(payload, dict):
        raise invalid
    login, user_id, exp = payload.get("login"), payload.get("id"), payload.get("exp")
    if not isinstance(login, str) or not login or not isinstance(user_id, int) or not isinstance(exp, int):
        raise invalid
    if exp <= int(time.time()):
        raise GatewayError(401, "session_expired", "The session has expired. Run `tekti login` again.")
    return login.lower()


def _allowed_logins() -> set[str]:
    return {x.strip().lower() for x in _env("ALLOWED_GITHUB_LOGINS").split(",") if x.strip()}


def _check_login_allowed(login: str) -> None:
    allowed = _allowed_logins()
    if allowed and login.lower() not in allowed:
        raise GatewayError(403, "not_on_allowlist", f"GitHub user '{login}' is not allowed on this gateway.")


def _access_end() -> dt.datetime | None:
    """End of GitHub access (exclusive) from GITHUB_EXPIRES, or None when unset."""
    raw = _env("GITHUB_EXPIRES").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            return dt.datetime.combine(dt.date.fromisoformat(raw) + dt.timedelta(days=1), dt.time(), dt.UTC)
        moment = dt.datetime.fromisoformat(raw)
        return moment if moment.tzinfo else moment.replace(tzinfo=dt.UTC)
    except ValueError as exc:
        raise GatewayError(500, "gateway_misconfigured", "The gateway's expiry setting is invalid.") from exc


def _check_access_period() -> None:
    end = _access_end()
    if end is not None and dt.datetime.now(dt.UTC) >= end:
        raise GatewayError(403, "token_expired", "Access through GitHub sign-in has ended on this gateway.")


# --------------------------------------------------------------------------- GitHub


def _github_user(github_token: str) -> tuple[str, int]:
    """Ask GitHub who owns the token. Returns (login, id)."""
    base = (_env("GITHUB_API_URL") or DEFAULT_GITHUB_API).rstrip("/")
    headers = {
        "authorization": f"Bearer {github_token}",
        "accept": "application/vnd.github+json",
        "x-github-api-version": GITHUB_API_VERSION,
        "user-agent": "tekti-gateway",
    }
    unavailable = GatewayError(502, "github_unavailable", "GitHub could not be reached. Try again later.")
    try:
        status, _, body = _send_upstream("GET", f"{base}/user", headers, None, GITHUB_TIMEOUT_SECONDS, 1024 * 1024)
    except (OSError, http.client.HTTPException, ValueError) as exc:  # TimeoutError is an OSError
        raise unavailable from exc
    if status >= 500:
        raise unavailable
    if status != 200:
        raise GatewayError(401, "github_token_rejected", "GitHub did not accept the token. Run `tekti login` again.")
    try:
        user = json.loads(body)
    except ValueError as exc:
        raise unavailable from exc
    login = user.get("login") if isinstance(user, dict) else None
    user_id = user.get("id") if isinstance(user, dict) else None
    if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,39}", login) or not isinstance(user_id, int):
        raise unavailable
    return login, user_id


# --------------------------------------------------------------------------- counters


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError as exc:
        raise GatewayError(500, "gateway_misconfigured", "The gateway's limits are invalid.") from exc


def _table() -> str:
    table = _env("COUNTER_TABLE")
    if not table:
        raise GatewayError(500, "gateway_misconfigured", "The gateway is not configured.")
    return table


def _error_code(exc: Exception) -> str | None:
    return ((getattr(exc, "response", None) or {}).get("Error") or {}).get("Code")


def _increment(key: str, cap: int, code: str, message: str) -> None:
    """Atomically add one to counter `key`; raise 429 `code` when it is already at `cap`."""
    if cap <= 0:
        raise GatewayError(429, code, message)
    try:
        _client("dynamodb").update_item(
            TableName=_table(),
            Key={"token": {"S": key}},
            UpdateExpression="ADD #c :one",
            ConditionExpression="attribute_not_exists(#c) OR #c < :cap",
            ExpressionAttributeNames={"#c": "count"},
            ExpressionAttributeValues={":one": {"N": "1"}, ":cap": {"N": str(cap)}},
        )
    except GatewayError:
        raise
    except Exception as exc:
        if _error_code(exc) == "ConditionalCheckFailedException":
            raise GatewayError(429, code, message) from exc
        raise GatewayError(500, "gateway_unavailable", "The gateway could not record the request.") from exc


def _count_owner() -> None:
    cap = _int_env("OWNER_CAP", DEFAULT_CAPS["owner"])
    _increment("owner", cap, "request_cap_reached", f"The request cap of {cap} for the owner token is reached.")


def _cancellation_codes(exc: Exception) -> list[str]:
    """Per-item codes of a TransactionCanceledException, in request order."""
    response = getattr(exc, "response", None) or {}
    reasons = response.get("CancellationReasons")
    if isinstance(reasons, list) and reasons:
        return [str((r or {}).get("Code") or "None") for r in reasons]
    # Fallback: botocore also puts them in the message, e.g. "... reasons [ConditionalCheckFailed, None]".
    message = str((response.get("Error") or {}).get("Message") or "")
    match = re.search(r"\[([^\]]*)\]\s*$", message)
    return [c.strip() for c in match.group(1).split(",")] if match else []


def _count_github(login: str) -> None:
    """Add one to the user's counter and to the all-users counter, both or neither."""
    user_cap = _int_env("GITHUB_USER_CAP", DEFAULT_CAPS["github_user"])
    total_cap = _int_env("GITHUB_TOTAL_CAP", DEFAULT_CAPS["github_total"])
    user_msg = f"The request cap of {user_cap} for GitHub user '{login}' is reached. Ask the gateway owner."
    total_msg = f"The request cap of {total_cap} for all signed-in users is reached. Ask the gateway owner."
    if user_cap <= 0:
        raise GatewayError(429, "request_cap_reached", user_msg)
    if total_cap <= 0:
        raise GatewayError(429, "total_cap_reached", total_msg)
    table = _table()

    def update(key: str, cap: int) -> dict:
        return {
            "Update": {
                "TableName": table,
                "Key": {"token": {"S": key}},
                "UpdateExpression": "ADD #c :one",
                "ConditionExpression": "attribute_not_exists(#c) OR #c < :cap",
                "ExpressionAttributeNames": {"#c": "count"},
                "ExpressionAttributeValues": {":one": {"N": "1"}, ":cap": {"N": str(cap)}},
            }
        }

    items = [update(f"gh:{login}", user_cap), update(GITHUB_TOTAL_KEY, total_cap)]
    for attempt in range(TRANSACTION_ATTEMPTS):
        try:
            _client("dynamodb").transact_write_items(TransactItems=items)
            return
        except Exception as exc:
            if _error_code(exc) != "TransactionCanceledException":
                raise GatewayError(500, "gateway_unavailable", "The gateway could not record the request.") from exc
            codes = _cancellation_codes(exc)
            if codes[:1] == ["ConditionalCheckFailed"]:
                raise GatewayError(429, "request_cap_reached", user_msg) from exc
            if codes[1:2] == ["ConditionalCheckFailed"]:
                raise GatewayError(429, "total_cap_reached", total_msg) from exc
            if not any(c in ("TransactionConflict", "ThrottlingError") for c in codes):
                raise GatewayError(500, "gateway_unavailable", "The gateway could not record the request.") from exc
            # Another request is updating the shared total right now: back off and try again.
            if attempt < TRANSACTION_ATTEMPTS - 1:
                time.sleep(0.03 * (2**attempt) + random.uniform(0, 0.03))
    raise GatewayError(503, "gateway_busy", "The gateway is busy. Try again in a moment.")


def _count_signin(login: str) -> None:
    limit = _int_env("GITHUB_SIGNIN_DAILY_LIMIT", DEFAULT_SIGNIN_DAILY_LIMIT)
    day = dt.datetime.now(dt.UTC).date().isoformat()
    _increment(
        f"signin:{login}:{day}",
        limit,
        "signin_limit_reached",
        f"Too many sign-ins for '{login}' today (limit {limit}). Try again tomorrow (UTC).",
    )


def _route(path: str) -> tuple[str, str, str, str, str | None]:
    """Return (vendor, upstream path, model source, credential place, path model)."""
    for vendor, routes in ROUTES.items():
        prefix = f"/{vendor}"
        if not path.startswith(prefix + "/"):
            continue
        upstream_path = path[len(prefix) :]
        for pattern, source, place in routes:
            match = pattern.match(upstream_path)
            if match:
                return vendor, upstream_path, source, place, (match.group(1) if source == "path" else None)
    raise GatewayError(404, "not_found", "No such route on this gateway.")


def _allowed_models() -> set[str]:
    return {m.strip() for m in _env("ALLOWED_MODELS").split(",") if m.strip()}


def _body_model(body: bytes) -> str:
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise GatewayError(400, "invalid_body", "The request body must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise GatewayError(400, "invalid_body", "The request body must be a JSON object.")
    model = parsed.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GatewayError(403, "model_not_allowed", "The request names no model.")
    return model.strip()


def _check_model(model: str) -> None:
    name = model.removeprefix("models/")
    if name not in _allowed_models():
        raise GatewayError(403, "model_not_allowed", f"Model '{name}' is not allowed through this gateway.")


# --------------------------------------------------------------------------- upstream


def _send_upstream(
    method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float, max_bytes: int
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Send one HTTP request and return (status, headers, body). Never follows redirects.

    Reads at most max_bytes + 1 bytes of the body so an oversized answer is detected
    without holding all of it in memory.
    """
    parts = urllib.parse.urlsplit(url)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname, parts.port, timeout=timeout, context=_tls_context()
        )
    elif parts.scheme == "http":
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    else:
        raise ValueError("unsupported scheme")
    try:
        conn.request(method, target, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read(max_bytes + 1)
        return resp.status, resp.getheaders(), data
    finally:
        conn.close()


def _tls_context() -> ssl.SSLContext:
    ctx = _CLIENTS.get("tls")
    if ctx is None:
        ctx = ssl.create_default_context()
        _CLIENTS["tls"] = ctx
    return ctx  # type: ignore[return-value]


def _upstream_base(vendor: str) -> str:
    return (_env(UPSTREAM_ENV[vendor]) or DEFAULT_UPSTREAMS[vendor]).rstrip("/")


def _forward_headers(incoming: dict[str, str], vendor_key: str, place: str, body: bytes) -> dict[str, str]:
    out = {
        name: value
        for name, value in incoming.items()
        if name in FORWARD_REQUEST_HEADERS or name.startswith(FORWARD_REQUEST_HEADER_PREFIXES)
    }
    out["accept-encoding"] = "identity"
    out["content-length"] = str(len(body))
    if place == "bearer":
        out["authorization"] = f"Bearer {vendor_key}"
    else:
        out[place] = vendor_key
    return out


def _response_headers(upstream_headers: list[tuple[str, str]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for name, value in upstream_headers:
        lower = name.lower()
        if lower in DROP_RESPONSE_HEADERS:
            continue
        merged[lower] = f"{merged[lower]}, {value}" if lower in merged else value
    return merged


def _usage(content_type: str, body: bytes) -> dict[str, int] | None:
    """Pull integer token counts out of a JSON or server-sent-event response. Never content."""
    found: dict[str, int] = {}

    def take(obj: object) -> None:
        if not isinstance(obj, dict):
            return
        candidates = [obj.get("usage"), obj.get("usageMetadata")]
        for nested in ("response", "message"):
            inner = obj.get(nested)
            if isinstance(inner, dict):
                candidates.append(inner.get("usage"))
        for usage in candidates:
            if isinstance(usage, dict):
                for k, v in usage.items():
                    if isinstance(v, int) and not isinstance(v, bool):
                        found[str(k)] = v

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        take(json.loads(payload))
                    except ValueError:
                        continue
    else:
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        for item in parsed if isinstance(parsed, list) else [parsed]:
            take(item)
    return found or None


def _client_response(status: int, headers: dict[str, str], body: bytes) -> dict:
    try:
        text = body.decode("utf-8")
        payload, is_b64 = text, False
    except UnicodeDecodeError:
        payload, is_b64 = base64.b64encode(body).decode("ascii"), True
    response = {"statusCode": status, "headers": headers, "body": payload, "isBase64Encoded": is_b64}
    size = len(json.dumps(response, ensure_ascii=False).encode("utf-8"))
    if size > LAMBDA_RESPONSE_LIMIT_BYTES - RESPONSE_ENVELOPE_RESERVE_BYTES:
        raise GatewayError(
            502,
            "upstream_response_too_large",
            "The vendor's response is larger than this gateway can return (6 MB). "
            "Ask for a shorter answer or call the vendor directly.",
        )
    return response


# --------------------------------------------------------------------------- handler


def handler(event: dict, context: object = None) -> dict:
    started = time.monotonic()
    record: dict = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
        "request_id": getattr(context, "aws_request_id", None),
        "token": None,
        "vendor": None,
        "model": None,
    }
    try:
        response = _handle(event, record)
    except GatewayError as err:
        record["error"] = err.code
        response = _error(err.status, err.code, err.message)
    except Exception as exc:  # noqa: BLE001 - last resort; never log the message
        record["error"] = "internal_error"
        record["error_type"] = type(exc).__name__
        response = _error(500, "internal_error", "The gateway failed to handle the request.")
    record["status"] = response["statusCode"]
    record["duration_ms"] = int((time.monotonic() - started) * 1000)
    _log(record)
    return response


def _handle(event: dict, record: dict) -> dict:
    http_ctx = (event.get("requestContext") or {}).get("http") or {}
    method = str(http_ctx.get("method") or "").upper()
    path = str(event.get("rawPath") or http_ctx.get("path") or "/")
    # Some clients percent-encode the colon in Google's ":generateContent".
    path = path.replace("%3A", ":").replace("%3a", ":")

    if path == "/health":
        record["route"] = "/health"
        if method != "GET":
            raise GatewayError(405, "method_not_allowed", "Use GET for /health.")
        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"ok": True}),
            "isBase64Encoded": False,
        }

    if path == "/auth/github":
        record["route"] = "/auth/github"
        if method != "POST":
            raise GatewayError(405, "method_not_allowed", "Only POST is allowed on this route.")
        return _sign_in(event, record)

    vendor, upstream_path, model_source, place, path_model = _route(path)
    record["vendor"] = vendor
    record["route"] = re.sub(r"^/v1beta/models/[^:]+:", "/v1beta/models/{model}:", upstream_path)
    if method != "POST":
        raise GatewayError(405, "method_not_allowed", "Only POST is allowed on this route.")

    headers = _headers(event)
    query, query_keys = _split_query(str(event.get("rawQueryString") or ""))
    caller = _authenticate(_presented_tokens(headers, query_keys))
    record["token"] = caller["name"]
    if caller["kind"] == "github":
        _check_login_allowed(caller["login"])  # checked again on every request
        _check_access_period()

    body = _body(event)
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise GatewayError(413, "body_too_large", "The request body is larger than 1 MB.")

    model = path_model if model_source == "path" else _body_model(body)
    record["model"] = model[:200]
    _check_model(model)

    vendor_key = _vendor_key(vendor)
    if caller["kind"] == "github":
        _count_github(caller["login"])
    else:
        _count_owner()

    url = _upstream_base(vendor) + upstream_path + (f"?{query}" if query else "")
    timeout = float(_env("UPSTREAM_TIMEOUT_SECONDS", "280"))
    max_bytes = LAMBDA_RESPONSE_LIMIT_BYTES
    try:
        status, up_headers, up_body = _send_upstream(
            "POST", url, _forward_headers(headers, vendor_key, place, body), body, timeout, max_bytes
        )
    except TimeoutError as exc:
        raise GatewayError(504, "upstream_timeout", f"{vendor} did not answer in time.") from exc
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise GatewayError(502, "upstream_unreachable", f"The gateway could not reach {vendor}.") from exc
    record["upstream_status"] = status
    if len(up_body) > max_bytes:
        raise GatewayError(
            502,
            "upstream_response_too_large",
            "The vendor's response is larger than this gateway can return (6 MB). "
            "Ask for a shorter answer or call the vendor directly.",
        )

    out_headers = _response_headers(up_headers)
    usage = _usage(out_headers.get("content-type", ""), up_body)
    if usage:
        record["usage"] = usage
    return _client_response(status, out_headers, up_body)


def _sign_in(event: dict, record: dict) -> dict:
    """POST /auth/github {"github_token": "..."} -> {"session", "login", "expires_at"}."""
    body = _body(event)
    if len(body) > MAX_AUTH_BODY_BYTES:
        raise GatewayError(413, "body_too_large", "The sign-in request is too large.")
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise GatewayError(400, "invalid_body", 'Send JSON {"github_token": "..."}.') from exc
    github_token = parsed.get("github_token") if isinstance(parsed, dict) else None
    if not isinstance(github_token, str) or not re.fullmatch(r"[\x21-\x7e]{1,1024}", github_token.strip()):
        raise GatewayError(400, "invalid_body", 'Send JSON {"github_token": "..."}.')
    _check_access_period()

    login, user_id = _github_user(github_token.strip())
    key = login.lower()
    record["token"] = f"gh:{key}"
    _check_login_allowed(key)
    _count_signin(key)

    now = int(time.time())
    exp = now + SESSION_LIFETIME_SECONDS
    end = _access_end()
    if end is not None:
        exp = min(exp, int(end.timestamp()))
    expires_at = dt.datetime.fromtimestamp(exp, dt.UTC).isoformat().replace("+00:00", "Z")
    return _json_response(200, {"session": _issue_session(key, user_id, exp), "login": login, "expires_at": expires_at})
