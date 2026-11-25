"""Strands Agents backend (default).

Verified against strands-agents 1.67 (source and https://strandsagents.com docs):
  * `Agent(model=..., system_prompt=..., tools=None, callback_handler=None)`; `callback_handler=None`
    stops Strands from printing streamed text to stdout.
  * `await agent.invoke_async(prompt, structured_output_model=Schema, limits={...})` returns an
    `AgentResult` whose `.structured_output` is a validated instance of `Schema`. Invalid tool input
    is sent back to the model by Strands; `limits={"structured_output_attempts": 2}` caps that at one
    repair. When the cap is hit the result has no structured output (or StructuredOutputException).
  * Token usage: `result.metrics.accumulated_usage["inputTokens" | "outputTokens"]`.
  * Throttling: `ModelRetryStrategy` retries `ModelThrottledException`; we widen it to timeouts and
    5xx codes, and botocore adaptive retries sit underneath.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel

from tekti.llm import LLMError, ModelSpec
from tekti.llm_backends._common import RETRYABLE_CODES, error_code, is_temperature_rejection

T = TypeVar("T", bound=BaseModel)

STRUCTURED_OUTPUT_ATTEMPTS = 2  # first answer + one repair


def _retry_strategy() -> Any:
    from strands.agent import ModelRetryStrategy  # type: ignore[attr-defined]
    from strands.types.exceptions import ModelThrottledException

    class _Retry(ModelRetryStrategy):
        def is_retryable(self, exception: Exception) -> bool:
            if isinstance(exception, ModelThrottledException):
                return True
            if error_code(exception) in RETRYABLE_CODES:
                return True
            name = type(exception).__name__
            return name in {"ReadTimeoutError", "ConnectTimeoutError", "APITimeoutError", "RateLimitError"}

    return _Retry(max_attempts=5, initial_delay=2, max_delay=30)


class StrandsBackend:
    """`tekti.llm.Backend` on the Strands Agents SDK. Safe to call concurrently under asyncio."""

    def __init__(
        self,
        *,
        region: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        read_timeout: int = 300,
    ) -> None:
        self._region = region
        self._base_url = base_url
        self._api_key = api_key
        self._read_timeout = read_timeout
        self._models: dict[tuple[str, str, int, bool], Any] = {}
        self._no_temperature: set[str] = set()

    # ------------------------------------------------------------------ model construction

    def _model(self, spec: ModelSpec, max_tokens: int) -> Any:
        use_temp = spec.model_id not in self._no_temperature
        key = (spec.provider, spec.model_id, max_tokens, use_temp)
        if key in self._models:
            return self._models[key]
        if spec.provider == "bedrock":
            from botocore.config import Config as BotoConfig
            from strands.models import BedrockModel

            kwargs: dict[str, Any] = {
                "model_id": spec.model_id,
                "max_tokens": max_tokens,  # always explicit: unset reserves the model maximum of quota
                "streaming": False,
                "boto_client_config": BotoConfig(
                    retries={"max_attempts": 4, "mode": "adaptive"},
                    connect_timeout=10,
                    read_timeout=self._read_timeout,
                ),
            }
            if self._region:
                kwargs["region_name"] = self._region
            if use_temp:
                kwargs["temperature"] = 0.0
            model = BedrockModel(**kwargs)
        elif spec.provider == "openai_compatible":
            from strands.models.openai import OpenAIModel

            if not self._base_url:
                raise LLMError("openai_compatible provider needs TEKTI_BASE_URL")
            params: dict[str, Any] = {"max_tokens": max_tokens}
            if use_temp:
                params["temperature"] = 0.0
            model = OpenAIModel(
                client_args={
                    "base_url": self._base_url,
                    "api_key": self._api_key or "unused",  # Ollama ignores the key but the client wants one
                    "max_retries": 4,
                    "timeout": float(self._read_timeout),
                },
                model_id=spec.model_id,
                params=params,
            )
        else:  # pragma: no cover - ModelSpec restricts the literal
            raise LLMError(f"unknown provider {spec.provider}")
        self._models[key] = model
        return model

    def _agent(self, spec: ModelSpec, system: str, max_tokens: int) -> Any:
        from strands import Agent

        return Agent(
            model=self._model(spec, max_tokens),
            system_prompt=system,
            tools=None,
            callback_handler=None,
            retry_strategy=_retry_strategy(),
        )

    # ------------------------------------------------------------------ Backend protocol

    async def complete(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        try:
            return await self._complete_once(model, system=system, user=user, schema=schema, max_tokens=max_tokens)
        except LLMError:
            raise
        except Exception as exc:
            if model.model_id not in self._no_temperature and is_temperature_rejection(exc):
                self._no_temperature.add(model.model_id)
                return await self._complete_once(model, system=system, user=user, schema=schema, max_tokens=max_tokens)
            raise

    async def _complete_once(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        from strands.types.exceptions import StructuredOutputException

        agent = self._agent(model, system, max_tokens)  # fresh agent: no shared conversation state
        try:
            result = await agent.invoke_async(
                user,
                structured_output_model=schema,
                limits={"structured_output_attempts": STRUCTURED_OUTPUT_ATTEMPTS},
            )
        except StructuredOutputException as exc:
            raise LLMError(f"output does not match schema {schema.__name__} after one repair: {exc}") from exc
        usage = result.metrics.accumulated_usage
        tokens_in, tokens_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
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
