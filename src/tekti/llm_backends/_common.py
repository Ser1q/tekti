"""Pieces shared by the backends: error classification and the one-repair validation loop."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tekti.llm import LLMError

T = TypeVar("T", bound=BaseModel)

# Bedrock error codes worth retrying (with backoff). Everything else fails fast.
RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ServiceQuotaExceededException",
    }
)

REPAIR_PROMPT = (
    "Your previous answer did not match the required JSON schema. Validation errors:\n{errors}\n"
    "Answer again with the complete result, strictly following the schema."
)


def error_code(exc: BaseException) -> str | None:
    """Return the AWS error code of a botocore ClientError (or a wrapped one), else None."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        response = getattr(cur, "response", None)
        if isinstance(response, dict):
            code = response.get("Error", {}).get("Code")
            if code:
                return str(code)
        cur = cur.__cause__ or cur.__context__
    return None


def is_temperature_rejection(exc: BaseException) -> bool:
    """True when a provider refused the request because of the temperature parameter.

    Seen wording: Bedrock "doesn't support the temperature field"; OpenAI "Unsupported parameter:
    'temperature' is not supported with this model"; Anthropic "temperature: Extra inputs are not
    permitted" (API version 2026-10-21+) and "`temperature` may only be set to 1 ...".
    """
    text = str(exc).lower()
    return "temperature" in text and any(
        w in text
        for w in ("support", "deprecated", "invalid", "not allowed", "not permitted", "cannot", "only", "remove")
    )


# Model id prefixes whose vendor API refuses (or ignores) a non-default temperature, so sending
# temperature 0 would only cost a refused request (one per concurrent call before the backend learns
# it, each counted against the gateway's request cap). Verified 26 Nov 2026 on the vendors' docs:
#   * Anthropic: API version 2026-10-21 (the default of anthropic SDK 2.x) rejects temperature/top_p/top_k
#     on every model; Claude 4.7+ and 5.x reject non-default values on any version.
#   * OpenAI: GPT-5.x/6.x reasoning models: "When reasoning effort is not none, remove temperature".
#   * Google: "Gemini 4 and Gemini 3.6+ models ignore custom values for temperature" (deprecation header).
_NO_TEMPERATURE_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5", "gpt-6", "o1", "o3", "o4"),
    "google": ("gemini-4", "gemini-3.6", "gemini-3.7", "gemini-3.8", "gemini-3.9", "gemini-5"),
}


def sends_temperature(provider: str, model_id: str) -> bool:
    """Whether the first request for this model carries temperature 0.

    A refusal still triggers drop-and-retry for any provider; this only avoids requests that are
    known to be refused or ignored.
    """
    if provider == "anthropic":
        return False
    if provider == "openai_compatible" and model_id.lower().startswith(("openai/", "anthropic/", "google/")):
        return False  # vendor reasoning models behind OpenRouter: same rule as calling the vendor
    prefixes = _NO_TEMPERATURE_PREFIXES.get(provider, ())
    return not model_id.lower().startswith(prefixes)


RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_RETRYABLE_NAMES = frozenset(
    {
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "ServerError",  # google.genai.errors.ServerError: any 5xx
    }
)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and all(cur is not s for s in seen):
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    return seen


def is_gateway_refusal(exc: BaseException) -> bool:
    """True for an error produced by the Tekti gateway itself (bad token, cap reached, model not allowed).

    Those never succeed on retry. The gateway's body is {"error": {"type": "tekti_gateway", ...}}. Where
    each SDK keeps it (checked against a local imitation of the gateway): openai -> `APIStatusError.type`
    (the text is only the message); anthropic -> the text "Error code: 429 - {...}" and `.body`;
    google-genai -> `APIError.details` and the text "429 None. {...}".
    """
    for e in _chain(exc):
        if getattr(e, "type", None) == "tekti_gateway" or "tekti_gateway" in str(e):
            return True
        for attr in ("body", "details"):
            payload = getattr(e, attr, None)
            if isinstance(payload, dict):
                inner = payload.get("error", payload)
                if isinstance(inner, dict) and inner.get("type") == "tekti_gateway":
                    return True
    return False


class GatewayRefused(BaseException):
    """The Tekti gateway refused the credential, the person or the allowance: the whole run must stop.

    Deliberately a BaseException, like asyncio.CancelledError: `tekti.llm.ask` turns every Exception into
    LLMError and each stage catches LLMError to drop one reviewer, so an Exception would become 15 identical
    reviewer failures. This passes through `ask`, the stages and asyncio.gather to the CLI, which prints
    `str(exc)`: one Russian sentence saying what to do.
    """

    def __init__(self, status: int, code: str, message: str = "") -> None:
        self.status, self.code, self.message = status, code, message
        super().__init__(refusal_text(status, code, message))


def refusal_text(status: int, code: str, message: str = "") -> str:
    if code == "session_expired":
        return "Сессия шлюза истекла — выполните `uv run tekti login` и запустите проверку снова."
    if status == 401:
        return (
            "Шлюз не принял учётные данные — выполните `uv run tekti login` (или проверьте TEKTI_GATEWAY_TOKEN) "
            "и запустите проверку снова."
        )
    if code == "not_on_allowlist":
        return "Вашего логина GitHub нет в списке допуска шлюза — попросите автора добавить его и войдите снова."
    if status == 429:
        return (
            "Лимит запросов через шлюз исчерпан — дальше проверяйте со своими ключами поставщиков или с бесплатным "
            "ключом Gemini (README, варианты Б и В)."
        )
    if status == 413:
        return "Документ слишком велик для шлюза (запрос больше 1 МБ) — сократите документ или используйте свои ключи."
    detail = f"{code}: {message}" if message else code
    return f"Шлюз отказал в доступе ({detail}) — сообщите автору."


def gateway_refusal(exc: BaseException) -> GatewayRefused | None:
    """A GatewayRefused for a gateway 4xx (its own error body), else None. Gateway 5xx stay retryable errors."""
    if not is_gateway_refusal(exc):
        return None
    status = status_code(exc)
    if status is None or status >= 500:
        return None
    code, message = "", ""
    for e in _chain(exc):
        for attr in ("body", "details"):
            payload = getattr(e, attr, None)
            if isinstance(payload, dict):
                inner = payload.get("error", payload)
                if isinstance(inner, dict) and inner.get("type") == "tekti_gateway":
                    code, message = str(inner.get("code") or ""), str(inner.get("message") or "")
        if not code and getattr(e, "type", None) == "tekti_gateway" and isinstance(getattr(e, "code", None), str):
            code = e.code  # type: ignore[attr-defined]
    return GatewayRefused(status, code, message)


def status_code(exc: BaseException) -> int | None:
    """HTTP status of a vendor SDK error (openai/anthropic `.status_code`, google-genai `.code`)."""
    for e in _chain(exc):
        for attr in ("status_code", "code"):
            value = getattr(e, attr, None)
            if isinstance(value, int) and 100 <= value < 600:
                return value
    return None


def is_retryable(exc: BaseException) -> bool:
    """Throttling, timeouts, connection drops and 5xx: worth retrying with backoff. Gateway refusals are not."""
    from strands.types.exceptions import ModelThrottledException

    code = status_code(exc)
    if is_gateway_refusal(exc):
        # 4xx from the gateway (token, cap, allowlist, expiry, size) is final; a gateway 5xx is transient.
        return code is not None and code >= 500
    if any(isinstance(e, ModelThrottledException) for e in _chain(exc)):
        return True
    if error_code(exc) in RETRYABLE_CODES:
        return True
    if code is not None:
        return code in RETRYABLE_STATUS
    return any(type(e).__name__ in _RETRYABLE_NAMES for e in _chain(exc))


def extract_json(text: str) -> Any:
    """Parse JSON from model text, tolerating code fences and leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        return json.loads(text[start:])


def short_errors(err: ValidationError | ValueError, limit: int = 1500) -> str:
    return str(err)[:limit]


async def validate_with_repair(
    schema: type[T],
    attempt: Callable[[str | None], Awaitable[tuple[Any, int, int]]],
) -> tuple[T, int, int]:
    """Run `attempt(None)`, validate against `schema`; on failure run `attempt(errors)` once more.

    `attempt` returns (raw payload: dict or JSON text, input tokens, output tokens). It receives
    the validation errors of the previous try, or None on the first try. Tokens of both tries are
    summed. Raises LLMError when the repaired answer still does not validate.
    """
    tokens_in = tokens_out = 0
    errors: str | None = None
    for _ in range(2):
        raw, t_in, t_out = await attempt(errors)
        tokens_in += t_in
        tokens_out += t_out
        try:
            payload = extract_json(raw) if isinstance(raw, str) else raw
            return schema.model_validate(payload), tokens_in, tokens_out
        except (ValidationError, ValueError) as exc:
            errors = short_errors(exc)
    raise LLMError(f"output does not match schema {schema.__name__} after one repair: {errors}")
