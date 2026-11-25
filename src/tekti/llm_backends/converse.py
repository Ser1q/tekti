"""Fallback backend: boto3 bedrock-runtime Converse with a forced tool call carrying the schema.

Bedrock only. One request per call (no conversation state), adaptive retries in botocore plus
backoff on throttling/timeouts here, explicit maxTokens, temperature 0 unless the model refuses it,
and one repair request when the tool input does not validate.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, TypeVar

from pydantic import BaseModel

from tekti.llm import LLMError, ModelSpec
from tekti.llm_backends._common import (
    REPAIR_PROMPT,
    RETRYABLE_CODES,
    error_code,
    is_temperature_rejection,
    validate_with_repair,
)

T = TypeVar("T", bound=BaseModel)

TOOL_NAME = "submit_result"


class ConverseBackend:
    def __init__(self, *, region: str | None = None, client: Any = None, read_timeout: int = 300) -> None:
        if client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=BotoConfig(
                    retries={"max_attempts": 4, "mode": "adaptive"},
                    connect_timeout=10,
                    read_timeout=read_timeout,
                    max_pool_connections=32,
                ),
            )
        self._client = client
        self._no_temperature: set[str] = set()
        self._no_forced_tool: set[str] = set()
        self.backoff_base = 2.0

    async def complete(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        if model.provider != "bedrock":
            raise LLMError("ConverseBackend supports provider 'bedrock' only")
        tool = {
            "toolSpec": {
                "name": TOOL_NAME,
                "description": f"Submit the final answer as a {schema.__name__} object.",
                "inputSchema": {"json": schema.model_json_schema()},
            }
        }

        async def attempt(errors: str | None) -> tuple[Any, int, int]:
            messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": user}]}]
            if errors:
                messages[0]["content"].append({"text": REPAIR_PROMPT.format(errors=errors)})
            return await self._call(model.model_id, system, messages, tool, max_tokens)

        return await validate_with_repair(schema, attempt)

    async def _call(
        self, model_id: str, system: str, messages: list[dict[str, Any]], tool: dict[str, Any], max_tokens: int
    ) -> tuple[Any, int, int]:
        for retry in range(5):
            request: dict[str, Any] = {
                "modelId": model_id,
                "system": [{"text": system}],
                "messages": messages,
                "inferenceConfig": {"maxTokens": max_tokens},
                "toolConfig": {"tools": [tool]},
            }
            if model_id not in self._no_temperature:
                request["inferenceConfig"]["temperature"] = 0.0
            if model_id not in self._no_forced_tool:
                request["toolConfig"]["toolChoice"] = {"tool": {"name": TOOL_NAME}}
            else:
                request["toolConfig"]["toolChoice"] = {"any": {}}
            try:
                response = await asyncio.to_thread(self._client.converse, **request)
            except Exception as exc:
                code = error_code(exc)
                if (
                    code == "ValidationException"
                    and is_temperature_rejection(exc)
                    and model_id not in self._no_temperature
                ):
                    self._no_temperature.add(model_id)
                    continue
                if (
                    code == "ValidationException"
                    and "toolchoice" in str(exc).lower().replace("_", "")
                    and model_id not in self._no_forced_tool
                ):
                    self._no_forced_tool.add(model_id)
                    continue
                if code in RETRYABLE_CODES and retry < 4:
                    await asyncio.sleep(self.backoff_base * (2**retry) * (0.5 + random.random()))
                    continue
                raise
            usage = response.get("usage", {})
            t_in, t_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
            content = response.get("output", {}).get("message", {}).get("content", [])
            for block in content:
                if "toolUse" in block and block["toolUse"].get("name") == TOOL_NAME:
                    return block["toolUse"].get("input"), t_in, t_out
            # No tool call: fall back to any JSON in the text, validated by the caller.
            text = "".join(b.get("text", "") for b in content)
            return text or "", t_in, t_out
        raise LLMError(f"{model_id}: gave up after retries")
