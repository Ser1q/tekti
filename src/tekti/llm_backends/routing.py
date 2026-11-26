"""A backend that hands each call to the backend registered for the model's provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from tekti.llm import LLMError, ModelSpec

T = TypeVar("T", bound=BaseModel)


class RoutingBackend:
    """`by_provider[model.provider]`, else `default`. Used for TEKTI_BACKEND=converse with a mixed panel:
    Bedrock models go to ConverseBackend, vendor models to StrandsBackend."""

    def __init__(self, by_provider: Mapping[str, Any], default: Any | None = None) -> None:
        self.by_provider = dict(by_provider)
        self.default = default

    async def complete(
        self, model: ModelSpec, *, system: str, user: str, schema: type[T], max_tokens: int
    ) -> tuple[T, int, int]:
        backend = self.by_provider.get(model.provider, self.default)
        if backend is None:
            raise LLMError(f"no backend for provider {model.provider}")
        return await backend.complete(model, system=system, user=user, schema=schema, max_tokens=max_tokens)
