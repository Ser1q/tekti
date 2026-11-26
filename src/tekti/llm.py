"""The only door to a language model.

Every LLM stage calls `ask`. No other module imports a provider SDK or an agent
framework, so the provider and framework choice stays replaceable.

INTERFACE IS FIXED (owned by the orchestrator). The backend behind it is package P0.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from tekti.contracts import Lens, ModelCall

T = TypeVar("T", bound=BaseModel)


class ModelSpec(BaseModel):
    family: str  # "anthropic", "openai", "xai", ... used for vote counting and challenger choice
    model_id: str  # provider-specific identifier
    # "openai", "anthropic" and "google" use each vendor's own API (directly with the vendor key, or through
    # the Tekti gateway); "openai_compatible" is any other endpoint speaking the OpenAI protocol (Ollama etc.).
    provider: Literal["bedrock", "openai", "anthropic", "google", "openai_compatible"] = "bedrock"


class LLMError(Exception):
    """Raised when a call fails after retries or returns output that does not fit the schema."""


class Backend(Protocol):
    async def complete(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        """Return (parsed output, input tokens, output tokens)."""
        ...


_backend: Backend | None = None
_calls: list[ModelCall] = []


_fallbacks: dict[tuple[str, str], ModelSpec] = {}


def use_fallbacks(mapping: dict[tuple[str, str], ModelSpec]) -> None:
    """Per-call stand-ins, keyed by (provider, model_id): when a call to that model fails or runs out of
    time, the same request is sent once to the stand-in. Set by the configuration; empty in tests."""
    _fallbacks.clear()
    _fallbacks.update(mapping)


# Share of the call limit the first model gets when a stand-in is waiting. A review legitimately takes
# minutes; a challenge, a merge or a question batch does not, so a stuck one is given up on sooner.
_FIRST_ATTEMPT_SHARE = {"panel": 0.8}
_FIRST_ATTEMPT_SHARE_DEFAULT = 0.4


def use_backend(backend: Backend) -> None:
    """Install a backend. Tests install a fake; the CLI installs the real one."""
    global _backend
    _backend = backend


def calls() -> list[ModelCall]:
    """Trace of every call made in this process, for the run passport."""
    return list(_calls)


def reset_calls() -> None:
    _calls.clear()


def call_timeout_seconds() -> float:
    """Wall-clock limit for one `ask`, retries included. TEKTI_CALL_TIMEOUT overrides the default of 300 s."""
    import os

    try:
        return max(10.0, float(os.environ.get("TEKTI_CALL_TIMEOUT", "300")))
    except ValueError:
        return 300.0


async def ask(
    model: ModelSpec,
    *,
    system: str,
    user: str,
    schema: type[T],
    stage: str,
    lens: Lens | None = None,
    max_tokens: int = 4000,
) -> T:
    """Ask one model for one structured answer.

    `system` holds instructions only. Document text goes in `user`, wrapped by the
    caller in <document>...</document> and described there as data to be reviewed.
    Records a ModelCall for every attempt, successful or not. If the model has a stand-in (see
    `use_fallbacks`) a failed call is retried on it once. Raises LLMError when every attempt failed.
    """
    import asyncio
    import time

    if _backend is None:
        raise LLMError("no backend installed; call tekti.llm.use_backend first")
    stand_in = _fallbacks.get((model.provider, model.model_id))
    if stand_in is not None and stand_in.model_id == model.model_id:
        stand_in = None
    attempts = [model] if stand_in is None else [model, stand_in]
    last_error: Exception | None = None
    for n, spec in enumerate(attempts):
        # One call may not stall the run: a vendor that keeps answering "overloaded" can otherwise hold a
        # reviewer for many minutes of retries. On the deadline the call is dropped and recorded; when a
        # stand-in model is configured it gets the same request once.
        limit = call_timeout_seconds()
        has_next = n + 1 < len(attempts)
        if has_next:
            limit *= _FIRST_ATTEMPT_SHARE.get(stage, _FIRST_ATTEMPT_SHARE_DEFAULT)
        started = time.monotonic()
        try:
            parsed, tokens_in, tokens_out = await asyncio.wait_for(
                _backend.complete(spec, system=system, user=user, schema=schema, max_tokens=max_tokens),
                timeout=limit,
            )
        except Exception as exc:  # noqa: BLE001 - any failure of one attempt is recorded, then the stand-in is tried
            note = f"; повтор на {attempts[n + 1].model_id}" if has_next else ""
            _calls.append(
                ModelCall(
                    stage=stage,
                    lens=lens,
                    model=spec.model_id,
                    ok=False,
                    error=(f"{type(exc).__name__}: {exc}"[:440] + note),
                    seconds=time.monotonic() - started,
                )
            )
            last_error = exc
            continue
        _calls.append(
            ModelCall(
                stage=stage,
                lens=lens,
                model=spec.model_id,
                ok=True,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                seconds=time.monotonic() - started,
            )
        )
        return parsed
    raise LLMError(str(last_error)) from last_error
