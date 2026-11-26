"""Fixtures for the gateway tests: the handler is loaded from gateway/handler.py and its AWS
clients and outbound HTTP are replaced by in-memory fakes. No network, no AWS."""

from __future__ import annotations

import base64
import importlib.util
import json
import secrets
import sys
from pathlib import Path

import pytest

HANDLER_PATH = Path(__file__).resolve().parents[2] / "gateway" / "handler.py"
ALLOWED = "gpt-test-1,claude-test-1,gemini-test-1"
GITHUB_BASE = "https://github.test"


def load_handler():
    name = "tekti_gateway_handler"
    spec = importlib.util.spec_from_file_location(name, HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AwsError(Exception):
    def __init__(self, code, **extra):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": extra.pop("message", code)}, **extra}


class FakeSecrets:
    def __init__(self, values: dict[str, str]):
        self.values = values
        self.calls = 0

    def get_secret_value(self, SecretId):
        self.calls += 1
        return {"SecretString": self.values[SecretId]}


def _check_update(update):
    assert update["UpdateExpression"] == "ADD #c :one"
    assert update["ConditionExpression"] == "attribute_not_exists(#c) OR #c < :cap"
    assert update["ExpressionAttributeNames"] == {"#c": "count"}
    return update["Key"]["token"]["S"], int(update["ExpressionAttributeValues"][":cap"]["N"])


class FakeDynamo:
    """Mimics the conditional ADD and the two-item transaction the handler issues."""

    def __init__(self):
        self.counts: dict[str, int] = {}
        self.calls = 0
        self.transactions = 0
        self.conflicts_to_inject = 0  # next N transactions fail with TransactionConflict
        self.reasons_in_message_only = False

    def _passes(self, key, cap):
        current = self.counts.get(key)
        return current is None or current < cap

    def update_item(self, **kwargs):
        self.calls += 1
        key, cap = _check_update(kwargs)
        if not self._passes(key, cap):
            raise AwsError("ConditionalCheckFailedException")
        self.counts[key] = self.counts.get(key, 0) + 1
        return {}

    def transact_write_items(self, TransactItems):
        self.calls += 1
        self.transactions += 1
        updates = [_check_update(item["Update"]) for item in TransactItems]
        keys = [k for k, _ in updates]
        assert len(set(keys)) == len(keys), "a transaction may not touch the same item twice"
        if self.conflicts_to_inject:
            self.conflicts_to_inject -= 1
            codes = ["None", "TransactionConflict"]
        else:
            codes = ["None" if self._passes(k, c) else "ConditionalCheckFailed" for k, c in updates]
        if any(c != "None" for c in codes):
            message = (
                f"Transaction cancelled, please refer cancellation reasons for specific reasons [{', '.join(codes)}]"
            )
            if self.reasons_in_message_only:
                raise AwsError("TransactionCanceledException", message=message)
            raise AwsError(
                "TransactionCanceledException",
                message=message,
                CancellationReasons=[{"Code": c} for c in codes],
            )
        for k in keys:
            self.counts[k] = self.counts.get(k, 0) + 1
        return {}


class Upstream:
    """Stands in for handler._send_upstream; records every request. GitHub calls go to `github`."""

    def __init__(self):
        self.requests: list[dict] = []
        self.github_requests: list[dict] = []
        self.status = 200
        self.headers = [("Content-Type", "application/json"), ("x-request-id", "req_123")]
        self.body = json.dumps({"id": "x", "usage": {"input_tokens": 11, "output_tokens": 7}}).encode()
        self.raise_exc: BaseException | None = None
        # GitHub: token -> (status, body); unknown token -> 401
        self.github_users: dict[str, tuple[int, bytes]] = {}
        self.github_exc: BaseException | None = None

    def add_github_user(self, token, login, user_id):
        self.github_users[token] = (200, json.dumps({"login": login, "id": user_id, "type": "User"}).encode())

    def __call__(self, method, url, headers, body, timeout, max_bytes):
        if url.startswith(GITHUB_BASE):
            self.github_requests.append({"method": method, "url": url, "headers": dict(headers), "body": body})
            if self.github_exc:
                raise self.github_exc
            token = headers.get("authorization", "").removeprefix("Bearer ")
            status, data = self.github_users.get(token, (401, b'{"message":"Bad credentials"}'))
            return status, [("content-type", "application/json")], data
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        if self.raise_exc:
            raise self.raise_exc
        return self.status, list(self.headers), self.body[: max_bytes + 1]


class Gateway:
    def __init__(self, module, secrets_client, dynamo, upstream, tokens, vendor_keys, real_send):
        self.module = module
        self.real_send = real_send
        self.secrets = secrets_client
        self.dynamo = dynamo
        self.upstream = upstream
        self.tokens = tokens
        self.vendor_keys = vendor_keys

    def call(self, method="POST", path="/openai/v1/chat/completions", body=None, headers=None, query="", b64=False):
        if body is None:
            body = {"model": "gpt-test-1", "messages": [{"role": "user", "content": "SECRET-PROMPT-TEXT"}]}
        raw = body if isinstance(body, (bytes, str)) else json.dumps(body)
        raw = raw.encode() if isinstance(raw, str) else raw
        event = {
            "version": "2.0",
            "rawPath": path,
            "rawQueryString": query,
            "headers": {k.lower(): v for k, v in (headers or {}).items()},
            "requestContext": {"http": {"method": method, "path": path}},
            "body": base64.b64encode(raw).decode() if b64 else raw.decode("utf-8"),
            "isBase64Encoded": b64,
        }
        return self.module.handler(event, None)

    def sign_in(self, github_token):
        return self.call(path="/auth/github", body={"github_token": github_token})

    def session_for(self, login="Octo-Cat", user_id=42):
        """Sign a GitHub user in through /auth/github and return the session."""
        token = "gho_" + secrets.token_hex(16)
        self.upstream.add_github_user(token, login, user_id)
        resp = self.sign_in(token)
        assert resp["statusCode"] == 200, resp
        return json.loads(resp["body"])["session"]


@pytest.fixture
def gw(monkeypatch):
    module = load_handler()
    tokens = {"owner": secrets.token_urlsafe(30)}
    signing_key = secrets.token_urlsafe(48)
    vendor_keys = {name: f"vk-{name}-" + secrets.token_hex(12) for name in ("openai", "anthropic", "google")}
    arns = {
        "VENDOR_KEYS_SECRET_ARN": "arn:test:vendor",
        "OWNER_TOKEN_SECRET_ARN": "arn:test:owner",
        "SESSION_SIGNING_SECRET_ARN": "arn:test:signing",
    }
    for k, v in arns.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("COUNTER_TABLE", "counters")
    monkeypatch.setenv("OWNER_CAP", "3000")
    monkeypatch.setenv("GITHUB_USER_CAP", "400")
    monkeypatch.setenv("GITHUB_TOTAL_CAP", "4000")
    monkeypatch.setenv("GITHUB_SIGNIN_DAILY_LIMIT", "10")
    monkeypatch.setenv("GITHUB_EXPIRES", "")
    monkeypatch.setenv("ALLOWED_GITHUB_LOGINS", "")
    monkeypatch.setenv("GITHUB_API_URL", GITHUB_BASE)
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED)
    monkeypatch.setenv("SECRET_CACHE_SECONDS", "120")
    for var in ("UPSTREAM_OPENAI_URL", "UPSTREAM_ANTHROPIC_URL", "UPSTREAM_GOOGLE_URL"):
        monkeypatch.delenv(var, raising=False)
    secrets_client = FakeSecrets(
        {
            "arn:test:vendor": json.dumps(vendor_keys),
            "arn:test:owner": tokens["owner"],
            "arn:test:signing": signing_key,
        }
    )
    dynamo = FakeDynamo()
    module._CLIENTS["secretsmanager"] = secrets_client
    module._CLIENTS["dynamodb"] = dynamo
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    upstream = Upstream()
    real_send = module._send_upstream
    monkeypatch.setattr(module, "_send_upstream", upstream)
    yield Gateway(module, secrets_client, dynamo, upstream, tokens, vendor_keys, real_send)
    sys.modules.pop("tekti_gateway_handler", None)
