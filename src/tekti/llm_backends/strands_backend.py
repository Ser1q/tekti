"""Strands Agents backend (default): one backend, the provider chosen per model (`ModelSpec.provider`).

Verified against strands-agents 1.67 (source and https://strandsagents.com docs):
  * `Agent(model=..., system_prompt=..., tools=None, callback_handler=None)`; `callback_handler=None`
    stops Strands from printing streamed text to stdout.
  * `await agent.invoke_async(prompt, structured_output_model=Schema, limits={...})` returns an
    `AgentResult` whose `.structured_output` is a validated instance of `Schema`. Invalid tool input
    is sent back to the model by Strands; `limits={"structured_output_attempts": 2}` caps that at one
    repair. When the cap is hit the result has no structured output (or StructuredOutputException).
  * Token usage: `result.metrics.accumulated_usage["inputTokens" | "outputTokens"]`, summed over every
    model request the agent made (repair included). Anthropic reports cache reads/writes apart from
    inputTokens; they are added back here.
  * Throttling: `ModelRetryStrategy` retries `ModelThrottledException`; we widen it to timeouts,
    connection drops and 5xx of every vendor SDK, and never retry a refusal by the Tekti gateway.

How each provider's model is built (URLs, structured output, streaming, usage): see `providers.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from tekti.llm import LLMError, ModelSpec
from tekti.llm_backends._common import (
    GatewayRefused,
    gateway_refusal,
    is_retryable,
    is_temperature_rejection,
    sends_temperature,
)
from tekti.llm_backends.providers import VendorEndpoint, build_model

T = TypeVar("T", bound=BaseModel)

STRUCTURED_OUTPUT_ATTEMPTS = 2  # first answer + one repair

# The backend only sees the output schema, so the stage is read from it. Unknown schemas use "default".
SCHEMA_STAGE = {
    "ReviewerOutput": "panel",
    "SegmentOutput": "segment",
    "ChallengeAnswer": "challenge",
    "MergeAnswer": "merge",
    "PairAnswer": "merge",
    "FinalDedupAnswer": "merge",
    "QuestionsAnswer": "questions",
}


def stage_of(schema: type) -> str:
    return SCHEMA_STAGE.get(getattr(schema, "__name__", ""), "default")


RETRY_INITIAL_DELAY = 2  # seconds; tests set 0


def _retry_strategy() -> Any:
    from strands.agent import ModelRetryStrategy  # type: ignore[attr-defined]

    class _Retry(ModelRetryStrategy):
        def is_retryable(self, exception: Exception) -> bool:
            return is_retryable(exception)

    return _Retry(max_attempts=5, initial_delay=RETRY_INITIAL_DELAY, max_delay=30)


def usage_tokens(usage: Mapping[str, Any]) -> tuple[int, int]:
    """(input, output) tokens from a Strands usage dict. Input includes cache reads and writes when the
    provider reports them apart (Anthropic); OpenAI and Gemini already count cached tokens in inputTokens."""
    tokens_in = int(usage.get("inputTokens", 0) or 0)
    tokens_out = int(usage.get("outputTokens", 0) or 0)
    return tokens_in, tokens_out


class StrandsBackend:
    """`tekti.llm.Backend` on the Strands Agents SDK. Safe to call concurrently under asyncio.

    Dispatches on `ModelSpec.provider`, so one run can mix Bedrock, OpenAI, Anthropic, Google and an
    OpenAI-compatible endpoint. `vendors` maps "openai"/"anthropic"/"google" to how that vendor is reached
    (vendor key directly, or the Tekti gateway with its token).
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        vendors: Mapping[str, VendorEndpoint] | None = None,
        read_timeout: int = 300,
        reasoning: Mapping[str, str] | None = None,
    ) -> None:
        self._region = region
        self._base_url = base_url
        self._api_key = api_key
        self._vendors = dict(vendors or {})
        self._read_timeout = read_timeout
        self._reasoning = dict(reasoning or {})  # stage -> low/medium/high
        self._models: dict[tuple[str, str, int, bool, str | None], Any] = {}
        self._no_temperature: set[tuple[str, str]] = set()
        self._refused: GatewayRefused | None = None

    def __repr__(self) -> str:
        return f"StrandsBackend(region={self._region!r}, vendors={self._vendors!r})"

    # ------------------------------------------------------------------ model construction

    def _uses_temperature(self, spec: ModelSpec) -> bool:
        return (spec.provider, spec.model_id) not in self._no_temperature and sends_temperature(
            spec.provider, spec.model_id
        )

    def _level(self, provider: str, stage: str) -> str | None:
        r = self._reasoning
        return r.get(f"{provider}:{stage}") or r.get(stage) or r.get(f"{provider}:default") or r.get("default")

    def _model(self, spec: ModelSpec, max_tokens: int, stage: str = "default") -> Any:
        use_temp = self._uses_temperature(spec)
        level = self._level(spec.provider, stage)
        key = (spec.provider, spec.model_id, max_tokens, use_temp, level)
        if key in self._models:
            return self._models[key]
        model = build_model(
            spec,
            max_tokens=max_tokens,
            use_temperature=use_temp,
            endpoint=self._vendors.get(spec.provider),
            region=self._region,
            compat_base_url=self._base_url,
            compat_api_key=self._api_key,
            read_timeout=self._read_timeout,
            reasoning=level,
        )
        self._models[key] = model
        return model

    def _agent(self, spec: ModelSpec, system: str, max_tokens: int, stage: str = "default") -> Any:
        from strands import Agent

        return Agent(
            model=self._model(spec, max_tokens, stage),
            system_prompt=system,
            tools=None,
            callback_handler=None,
            retry_strategy=_retry_strategy(),
        )

    # ------------------------------------------------------------------ Backend protocol

    async def complete(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        """Raises GatewayRefused (a BaseException, so the run stops) on a gateway 4xx; after the first one,
        every later call raises it at once without a request, so concurrent reviewers do not pile up refusals."""
        if self._refused is not None:
            raise GatewayRefused(self._refused.status, self._refused.code, self._refused.message)
        try:
            return await self._complete_with_temperature_fallback(
                model, system=system, user=user, schema=schema, max_tokens=max_tokens
            )
        except Exception as exc:
            refusal = gateway_refusal(exc)
            if refusal is None:
                raise
            self._refused = refusal
            raise refusal from exc

    async def _complete_with_temperature_fallback(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        used_temperature = self._uses_temperature(model)
        try:
            return await self._complete_once(model, system=system, user=user, schema=schema, max_tokens=max_tokens)
        except LLMError:
            raise
        except Exception as exc:
            if used_temperature and is_temperature_rejection(exc):
                self._no_temperature.add((model.provider, model.model_id))
                return await self._complete_once(model, system=system, user=user, schema=schema, max_tokens=max_tokens)
            raise

    async def _complete_once(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        from strands.types.exceptions import StructuredOutputException

        agent = self._agent(model, system, max_tokens, stage_of(schema))  # fresh agent: no shared state
        try:
            result = await agent.invoke_async(
                user,
                structured_output_model=schema,
                limits={"structured_output_attempts": STRUCTURED_OUTPUT_ATTEMPTS},
            )
        except StructuredOutputException as exc:
            raise LLMError(f"output does not match schema {schema.__name__} after one repair: {exc}") from exc
        usage = result.metrics.accumulated_usage
        tokens_in, tokens_out = usage_tokens(usage)
        if model.provider == "anthropic":
            tokens_in += int(usage.get("cacheReadInputTokens", 0) or 0) + int(
                usage.get("cacheWriteInputTokens", 0) or 0
            )
        parsed = result.structured_output
        if parsed is None:
            raise LLMError(
                f"no structured output for schema {schema.__name__} (stop_reason={result.stop_reason}, "
                f"tokens in/out {tokens_in}/{tokens_out})"
            )
        if not isinstance(parsed, schema):  # defensive: re-validate whatever came back
            parsed = schema.model_validate(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed)
        return parsed, tokens_in, tokens_out


async def _smoke() -> None:  # pragma: no cover - manual check: uv run python -m tekti.llm_backends.strands_backend
    class Answer(BaseModel):
        capital: str

    backend = StrandsBackend(region="us-east-1")
    spec = ModelSpec(family="anthropic", model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    print(
        await backend.complete(spec, system="Answer briefly.", user="Capital of France?", schema=Answer, max_tokens=200)
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_smoke())
