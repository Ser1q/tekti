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
    Records a ModelCall whether the call succeeds or fails. Raises LLMError on failure.
    """
    import asyncio
    import time

    if _backend is None:
        raise LLMError("no backend installed; call tekti.llm.use_backend first")
    started = time.monotonic()
    try:
        # One call may not stall the run: a vendor that keeps answering "overloaded" can otherwise hold a
        # reviewer for many minutes of retries. On the deadline the reviewer is dropped and recorded.
        parsed, tokens_in, tokens_out = await asyncio.wait_for(
            _backend.complete(model, system=system, user=user, schema=schema, max_tokens=max_tokens),
            timeout=call_timeout_seconds(),
        )
    except Exception as exc:
        _calls.append(
            ModelCall(
                stage=stage,
                lens=lens,
                model=model.model_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:500],
                seconds=time.monotonic() - started,
            )
        )
        raise LLMError(str(exc)) from exc
    _calls.append(
        ModelCall(
            stage=stage,
            lens=lens,
            model=model.model_id,
            ok=True,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            seconds=time.monotonic() - started,
        )
    )
    return parsed
