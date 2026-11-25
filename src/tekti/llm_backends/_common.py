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
    """True when a provider refused the request because of the temperature parameter."""
    text = str(exc).lower()
    return "temperature" in text and any(
        w in text for w in ("support", "deprecated", "invalid", "not allowed", "cannot", "only", "remove")
    )


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
