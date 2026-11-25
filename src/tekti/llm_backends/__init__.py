"""Model backends behind `tekti.llm.ask` (package P0).

Two backends satisfy `tekti.llm.Backend`:

* `StrandsBackend` (default): Strands Agents SDK. Bedrock models via `BedrockModel`,
  OpenAI-compatible endpoints (Gemini, Ollama, ...) via `OpenAIModel`. One fresh agent per
  call, no tools, structured output through Strands' structured-output tool with at most one
  repair attempt.
* `ConverseBackend` (fallback): boto3 `bedrock-runtime` Converse with a forced tool call that
  carries the JSON schema. Bedrock only. Same retry and repair rules.

Neither backend logs prompts, outputs or key values.
"""

from tekti.llm_backends.converse import ConverseBackend
from tekti.llm_backends.strands_backend import StrandsBackend

__all__ = ["ConverseBackend", "StrandsBackend"]
