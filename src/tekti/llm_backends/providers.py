"""Building one Strands model object per (provider, model id): Bedrock, OpenAI, Anthropic, Google, OpenAI-compatible.

Verified against the installed packages (strands-agents 1.67.0, openai 3.59, anthropic 2.14, google-genai 2.34)
and https://strandsagents.com, 26 Nov 2026:

* OpenAI -> `strands.models.openai_responses.OpenAIResponsesModel(client_args=..., model_id=..., params=...)`,
  i.e. `POST {base_url}/responses`. Not the Chat Completions `OpenAIModel`: Strands' model info marks GPT-6.x
  as accepting tools on Chat Completions only with reasoning off (6.1 Terra, 6 Luna) or not at all
  (6.2 Sol/Astra), and Strands' structured output for OpenAI is a tool call. `params` are merged into the
  request as is, so the token cap is `max_output_tokens` (it includes reasoning tokens). `store` is False
  (stateful=False default). The request always streams (`"stream": True` is set before `params` are merged,
  but the response handler iterates SSE events, so it cannot be switched off).
  Usage: `response.completed` -> `usage.input_tokens/output_tokens` (+ `input_tokens_details.cached_tokens`).
* Anthropic -> `strands.models.anthropic.AnthropicModel(client_args=..., model_id=..., max_tokens=..., params=...)`,
  i.e. `POST {base_url}/v1/messages` with `"stream": true` (Strands always uses `client.messages.stream`).
  `_supports_structured_output = True`: Strands picks forced tool / native `output_config.format` / auto per
  model (Claude 5.5 models: native JSON schema output, forced tool choice unsupported).
  Usage: `message_start`/`message_delta` usage; `input_tokens` excludes cache reads/writes, which Strands
  reports separately as cacheRead/WriteInputTokens; we add them back so a call records all its input.
  Sampling parameters: anthropic 2.x sends `anthropic-version: 2026-10-21`, which rejects temperature on every
  model, so none is sent (see `sends_temperature`).
* Google -> `strands.models.gemini.GeminiModel(client_args=..., model_id=..., params=...)` over google-genai,
  i.e. `POST {base_url}/v1beta/models/<model>:streamGenerateContent?alt=sse` (Strands always calls
  `generate_content_stream`). Base URL via `client_args={"http_options": HttpOptions(base_url=..., api_version=...)}`;
  the key goes in the `x-goog-api-key` header. `params` become `GenerateContentConfig` fields, so the cap is
  `max_output_tokens` (thinking tokens count toward it). Structured output is Strands' tool path
  (function declaration + function calling config ANY on the forced retry).
  Usage: `usageMetadata.promptTokenCount (+toolUsePromptTokenCount)`; output = candidates + thoughts.
* Bedrock -> `BedrockModel` (unchanged); OpenAI-compatible -> Chat Completions `OpenAIModel` (unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from tekti.llm import LLMError, ModelSpec

VENDORS = ("openai", "anthropic", "google")

# Path the gateway prefix plus each SDK's own suffix must produce; see gateway/CONTRACT.md.
GATEWAY_PREFIX = {"openai": "/openai/v1", "anthropic": "/anthropic", "google": "/google"}
GOOGLE_API_VERSION = "v1beta"


@dataclass(frozen=True)
class VendorEndpoint:
    """Where one vendor is reached and with which key. `api_key` is the vendor key, or the gateway token."""

    api_key: SecretStr = field(repr=False)
    base_url: str | None = None  # None: the SDK's default vendor URL
    via_gateway: bool = False

    def __repr__(self) -> str:  # never show the key
        return f"VendorEndpoint(base_url={self.base_url!r}, via_gateway={self.via_gateway})"


# --------------------------------------------------------------------------- reasoning effort
#
# Every current vendor model thinks before answering, and the thinking counts against the output cap:
#   * OpenAI Responses API: `reasoning={"effort": ...}`; reasoning tokens count toward max_output_tokens.
#   * Anthropic: `output_config={"effort": ...}` (Claude 4.6+ / 5.x; adaptive thinking is on by default on
#     Claude Haiku 5.5 with default effort medium); thinking counts toward max_tokens.
#   * Google: `thinking_config={"thinking_level": ...}` (Gemini 3.6+ and 4); thoughts count toward
#     max_output_tokens. Strands passes `params` straight into GenerateContentConfig.
# One level per stage keeps thinking in proportion to the task, and the cap gets headroom for it, so the
# structured answer is not cut off (vendors-1: every Gemini review stopped at the 12000-token cap).
REASONING_LEVELS = ("off", "low", "medium", "high")
THINKING_HEADROOM = {"off": 0, "low": 4000, "medium": 12000, "high": 24000}
# OpenAI models documented to accept reasoning effort "none" (GPT-6.2 Sol/Astra reject it).
_OPENAI_EFFORT_NONE = ("gpt-6.1-terra", "gpt-6-luna", "gpt-6-sol")
_ANTHROPIC_EFFORT = (
    "claude-haiku-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-fable",
    "claude-mythos",
)
_OPENAI_EFFORT = ("gpt-5", "gpt-6", "o1", "o3", "o4")
_GOOGLE_LEVEL = ("gemini-4", "gemini-3.6", "gemini-3.7", "gemini-3.8", "gemini-3.9", "gemini-5")


def supports_reasoning_level(provider: str, model_id: str) -> bool:
    prefixes = {"openai": _OPENAI_EFFORT, "anthropic": _ANTHROPIC_EFFORT, "google": _GOOGLE_LEVEL}.get(provider, ())
    return bool(prefixes) and model_id.lower().startswith(prefixes)


def reasoning_params(provider: str, model_id: str, level: str | None) -> dict[str, Any]:
    """The request fields that set the thinking level, or {} when the model has no such control."""
    if not level or not supports_reasoning_level(provider, model_id):
        return {}
    mid = model_id.lower()
    if provider == "openai":
        if level == "off":
            return {"reasoning": {"effort": "none" if mid.startswith(_OPENAI_EFFORT_NONE) else "low"}}
        return {"reasoning": {"effort": level}}
    if provider == "anthropic":
        if level == "off":  # accepted with effort high or below (Claude Haiku 5.5, Opus 5); not on Opus/Sonnet 5.5
            if mid.startswith(("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable", "claude-mythos")):
                return {"output_config": {"effort": "low"}}
            return {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
        return {"output_config": {"effort": level}}
    return {"thinking_config": {"thinking_level": "LOW" if level == "off" else level.upper()}}


def output_cap(provider: str, model_id: str, max_tokens: int, level: str | None) -> int:
    """The answer budget plus room for thinking at this level."""
    if level and supports_reasoning_level(provider, model_id):
        return max_tokens + THINKING_HEADROOM.get(level, 0)
    return max_tokens


class _EventList:
    """An already-read SSE stream, iterable like openai's AsyncStream (a class, not an async generator, so
    nothing is left suspended when the consumer stops early)."""

    def __init__(self, events: list[Any]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> _EventList:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class _BufferedResponses:
    def __init__(self, client_args: dict[str, Any]) -> None:
        self._client_args = client_args

    async def create(self, **request: Any) -> Any:
        import openai

        async with openai.AsyncOpenAI(**self._client_args) as client:
            response = await client.responses.create(**request)
            if not request.get("stream"):
                return response
            async with response:  # read to the end, then close the HTTP stream
                return _EventList([event async for event in response])


class BufferedOpenAIClient:
    """Stands in for AsyncOpenAI in Strands' OpenAIResponsesModel (`client=`).

    Strands stops reading the Responses SSE stream at `response.completed` without closing it; the suspended
    httpx2 generators are then finalised at interpreter shutdown and print "RuntimeError: aclose():
    asynchronous generator is already running" (runs/vendors-1.log). Here each request opens its own client,
    reads the whole stream and closes it before Strands iterates the events. Nothing is shown while streaming
    anyway, and through the gateway the body arrives whole, so buffering costs no latency.
    """

    def __init__(self, client_args: dict[str, Any]) -> None:
        self.responses = _BufferedResponses(client_args)


def gateway_endpoint(vendor: str, gateway_url: str, token: SecretStr) -> VendorEndpoint:
    """Base URL that makes the vendor SDK hit `<gateway>/<prefix>/<vendor path>` exactly as the contract lists."""
    return VendorEndpoint(api_key=token, base_url=gateway_url.rstrip("/") + GATEWAY_PREFIX[vendor], via_gateway=True)


def build_model(
    spec: ModelSpec,
    *,
    max_tokens: int,
    use_temperature: bool,
    endpoint: VendorEndpoint | None = None,
    region: str | None = None,
    compat_base_url: str | None = None,
    compat_api_key: str | None = None,
    read_timeout: int = 300,
    reasoning: str | None = None,
) -> Any:
    """A Strands model for `spec`. Raises LLMError when the provider is not configured.

    `reasoning` (low/medium/high) sets the vendor's thinking level and widens the output cap by
    THINKING_HEADROOM; models without a thinking control ignore it.
    """
    provider = spec.provider
    extra = reasoning_params(provider, spec.model_id, reasoning)
    max_tokens = output_cap(provider, spec.model_id, max_tokens, reasoning)
    if provider == "bedrock":
        from botocore.config import Config as BotoConfig
        from strands.models import BedrockModel

        kwargs: dict[str, Any] = {
            "model_id": spec.model_id,
            "max_tokens": max_tokens,  # always explicit: unset reserves the model maximum of quota
            "streaming": False,
            "boto_client_config": BotoConfig(
                retries={"max_attempts": 4, "mode": "adaptive"}, connect_timeout=10, read_timeout=read_timeout
            ),
        }
        if region:
            kwargs["region_name"] = region
        if use_temperature:
            kwargs["temperature"] = 0.0
        return BedrockModel(**kwargs)

    if provider == "openai_compatible":
        from strands.models.openai import OpenAIModel

        if not compat_base_url:
            raise LLMError("openai_compatible provider needs TEKTI_BASE_URL")
        params: dict[str, Any] = {"max_tokens": max_tokens}
        if use_temperature:
            params["temperature"] = 0.0
        return OpenAIModel(
            client_args={
                "base_url": compat_base_url,
                "api_key": compat_api_key or "unused",  # Ollama ignores the key but the client wants one
                "max_retries": 4,
                "timeout": float(read_timeout),
            },
            model_id=spec.model_id,
            params=params,
        )

    if provider not in VENDORS:  # pragma: no cover - ModelSpec restricts the literal
        raise LLMError(f"unknown provider {provider}")
    if endpoint is None:
        hint = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}[provider]
        raise LLMError(
            f"{provider} model {spec.model_id} is configured but neither {hint} nor "
            "TEKTI_GATEWAY_URL (with `tekti login`) is set"
        )
    # Through the gateway the SDK must not retry on its own: a gateway 429 (cap reached) would be resent.
    # Our Strands retry strategy retries throttling/5xx and knows gateway refusals (see _common.is_retryable).
    sdk_retries = 0 if endpoint.via_gateway else 2
    key = endpoint.api_key.get_secret_value()

    if provider == "openai":
        from strands.models.openai_responses import OpenAIResponsesModel

        client_args: dict[str, Any] = {"api_key": key, "max_retries": sdk_retries, "timeout": float(read_timeout)}
        if endpoint.base_url:
            client_args["base_url"] = endpoint.base_url
        params = {"max_output_tokens": max_tokens, **extra}
        if use_temperature:
            params["temperature"] = 0.0
        return OpenAIResponsesModel(client=BufferedOpenAIClient(client_args), model_id=spec.model_id, params=params)

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        client_args = {"api_key": key, "max_retries": sdk_retries, "timeout": float(read_timeout)}
        if endpoint.base_url:
            client_args["base_url"] = endpoint.base_url
        params = {**({"temperature": 0.0} if use_temperature else {}), **extra}
        return AnthropicModel(client_args=client_args, model_id=spec.model_id, max_tokens=max_tokens, params=params)

    # google
    from google.genai import types as genai_types
    from strands.models.gemini import GeminiModel

    http_options = genai_types.HttpOptions(api_version=GOOGLE_API_VERSION, timeout=read_timeout * 1000)
    if endpoint.base_url:
        http_options.base_url = endpoint.base_url
    params = {"max_output_tokens": max_tokens, **extra}
    if use_temperature:
        params["temperature"] = 0.0
    return GeminiModel(
        client_args={"api_key": key, "vertexai": False, "http_options": http_options},
        model_id=spec.model_id,
        params=params,
    )
