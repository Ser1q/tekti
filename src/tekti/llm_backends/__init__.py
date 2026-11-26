"""Model backends behind `tekti.llm.ask` (package P0).

* `StrandsBackend` (default): Strands Agents SDK, provider chosen per model (`ModelSpec.provider`):
  Bedrock (`BedrockModel`), OpenAI (`OpenAIResponsesModel`), Anthropic (`AnthropicModel`), Google
  (`GeminiModel`), any OpenAI-compatible endpoint (`OpenAIModel`). OpenAI, Anthropic and Google are
  reached with the vendor key or through the Tekti gateway (see `providers.py`). One fresh agent per
  call, no tools, structured output through Strands with at most one repair attempt.
* `ConverseBackend` (fallback): boto3 `bedrock-runtime` Converse with a forced tool call that
  carries the JSON schema. Bedrock only. Same retry and repair rules.
* `RoutingBackend`: dispatch by provider (Converse for Bedrock, Strands for the rest).

No backend logs prompts, outputs or key values.
"""

from tekti.llm_backends._common import GatewayRefused
from tekti.llm_backends.converse import ConverseBackend
from tekti.llm_backends.providers import VendorEndpoint, gateway_endpoint
from tekti.llm_backends.routing import RoutingBackend
from tekti.llm_backends.strands_backend import StrandsBackend

__all__ = [
    "ConverseBackend",
    "GatewayRefused",
    "RoutingBackend",
    "StrandsBackend",
    "VendorEndpoint",
    "gateway_endpoint",
]
