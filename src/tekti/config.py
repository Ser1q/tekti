"""Run settings: which models sit on the panel, which backend talks to them.

Defaults come from the model spike (scripts/spike_models.py, 26 Nov 2026, Bedrock us-east-1).
Everything can be overridden with environment variables, or a `.env` file in the working
directory (real environment variables win over `.env`):

    TEKTI_PROVIDER         bedrock (default) | openai_compatible
    TEKTI_BACKEND          strands (default) | converse   (converse: Bedrock only, no Strands)
    TEKTI_REGION           AWS region for Bedrock; default AWS_REGION, then us-east-1
    TEKTI_PANEL            comma-separated family:model_id, strongest first, one per family, e.g.
                           anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0,openai:openai.gpt-oss-120b-1:0
                           fast mode uses the first entry, standard the first two, board the first three.
                           Required when TEKTI_PROVIDER=openai_compatible.
    TEKTI_SEGMENTER        family:model_id (or a bare model_id) for requirement segmentation
    TEKTI_QUESTION_WRITER  family:model_id (or a bare model_id) for clarifying questions
    TEKTI_LENSES           comma-separated subset of: functional,api_contract,data_integration,tester,developer
    TEKTI_MAX_PARALLEL     concurrent panel calls (default 6)
    TEKTI_MAX_OUTPUT_TOKENS  max output tokens per call (default 12000; kimi spends ~2-5k on reasoning
                           even for a two-requirement task)
    TEKTI_BASE_URL         OpenAI-compatible endpoint, e.g.
                           https://generativelanguage.googleapis.com/v1beta/openai/  (Gemini)
                           http://localhost:11434/v1                                 (Ollama)
    TEKTI_API_KEY          key for that endpoint (never printed or logged)

Bedrock uses the standard AWS credential chain (AWS_PROFILE, AWS_ACCESS_KEY_ID, ...).
Model ids may contain ':' (e.g. openai.gpt-oss-120b-1:0); only the first ':' separates the family.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from tekti.contracts import Lens, Mode
from tekti.llm import ModelSpec

# Strongest first, one per family; ordered so the first three are the most independent families
# that passed 3/3 schema-valid runs in the spike.
DEFAULT_BENCH: list[ModelSpec] = [
    ModelSpec(family="anthropic", model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ModelSpec(family="openai", model_id="openai.gpt-oss-120b-1:0"),
    ModelSpec(family="mistral", model_id="mistral.mistral-large-3-675b-instruct"),
    ModelSpec(family="qwen", model_id="qwen.qwen3-vl-235b-a22b"),
    # Kimi K3.1 sat third in the first full run: 2 of its 5 reviews failed (temperature field, output
    # limit) and it took up to 195 s per call. Kept on the bench for challenges only.
    ModelSpec(family="moonshot", model_id="us.moonshotai.kimi-k3.1"),
]
# Not on the bench: every claude-*-5*, gpt-6*, grok-4.6/4.7 and glm-5.3 answered AccessDenied
# ("not available for this account"); deepseek.v3.2 and zai.glm-5 were schema-valid but missed the
# planted Ф-3/Ф-4 contradiction (glm-5 returned empty lists); llama3-3-70b failed the schema.
# Nova 2 Lite is faster but did not split compound requirements on the organisers' document; Haiku did.
DEFAULT_SEGMENTER = DEFAULT_BENCH[0]
DEFAULT_QUESTION_WRITER = DEFAULT_BENCH[0]

PANEL_SIZE = {Mode.FAST: 1, Mode.STANDARD: 2, Mode.BOARD: 3}

# Model id prefixes -> family, for bare ids in TEKTI_SEGMENTER / TEKTI_QUESTION_WRITER.
_FAMILY_HINTS = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "xai": "xai",
    "grok": "xai",
    "zai": "zai",
    "glm": "zai",
    "moonshot": "moonshot",
    "kimi": "moonshot",
    "deepseek": "deepseek",
    "mistral": "mistral",
    "qwen": "qwen",
    "minimax": "minimax",
    "amazon": "amazon",
    "nova": "amazon",
    "meta": "meta",
    "llama": "meta",
    "gemini": "google",
    "gemma": "google",
    "google": "google",
}


class Config(BaseModel):
    mode: Mode
    panel: list[ModelSpec]  # one per family; fast=1, standard=2, board=3
    lenses: list[Lens]  # all five by default
    segmenter: ModelSpec
    question_writer: ModelSpec
    max_parallel: int  # concurrency cap for panel calls
    max_output_tokens: int
    # Backend settings (not part of the review logic).
    bench: list[ModelSpec] = Field(default_factory=list)  # every configured family, strongest first
    backend: Literal["strands", "converse"] = "strands"
    region: str = "us-east-1"
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)


def _env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is not None:
        return dict(env)
    merged: dict[str, str] = {}
    dotenv = Path.cwd() / ".env"
    if dotenv.is_file():
        from dotenv import dotenv_values

        merged.update({k: v for k, v in dotenv_values(dotenv).items() if v is not None})
    merged.update(os.environ)
    return merged


def _guess_family(model_id: str) -> str:
    tail = model_id.lower()
    for prefix in ("us.", "eu.", "apac.", "global."):
        if tail.startswith(prefix):
            tail = tail[len(prefix) :]
            break
    for hint, family in _FAMILY_HINTS.items():
        if tail.startswith(hint):
            return family
    return tail.split(".")[0].split("-")[0] or "unknown"


def parse_spec(text: str, provider: str) -> ModelSpec:
    """Parse 'family:model_id' or a bare model id. Only the first ':' splits, and only when the
    part before it looks like a family name (no dots), so 'openai.gpt-oss-120b-1:0' stays whole."""
    text = text.strip()
    family, sep, rest = text.partition(":")
    if sep and family and "." not in family and "/" not in family and rest:
        return ModelSpec(family=family.strip().lower(), model_id=rest.strip(), provider=provider)  # type: ignore[arg-type]
    return ModelSpec(family=_guess_family(text), model_id=text, provider=provider)  # type: ignore[arg-type]


def parse_panel(text: str, provider: str) -> list[ModelSpec]:
    specs = [parse_spec(part, provider) for part in text.split(",") if part.strip()]
    seen: set[str] = set()
    unique = []
    for spec in specs:  # one per family: keep the first (strongest)
        if spec.family not in seen:
            seen.add(spec.family)
            unique.append(spec)
    return unique


def load_config(mode: Mode | str = Mode.BOARD, env: Mapping[str, str] | None = None) -> Config:
    """Build the run config from defaults, `.env` and environment variables (see module docstring).

    Pass `env` to bypass the process environment and `.env` (tests do this).
    """
    mode = Mode(mode)
    e = _env(env)
    provider = e.get("TEKTI_PROVIDER", "bedrock").strip() or "bedrock"
    if provider not in ("bedrock", "openai_compatible"):
        raise ValueError(f"TEKTI_PROVIDER должен быть bedrock или openai_compatible, получено: {provider!r}")
    backend = e.get("TEKTI_BACKEND", "strands").strip() or "strands"
    if backend not in ("strands", "converse"):
        raise ValueError(f"TEKTI_BACKEND должен быть strands или converse, получено: {backend!r}")
    if backend == "converse" and provider != "bedrock":
        raise ValueError("TEKTI_BACKEND=converse работает только с TEKTI_PROVIDER=bedrock")

    if e.get("TEKTI_PANEL", "").strip():
        bench = parse_panel(e["TEKTI_PANEL"], provider)
    elif provider == "bedrock":
        bench = [s.model_copy() for s in DEFAULT_BENCH]
    else:
        raise ValueError(
            "Для TEKTI_PROVIDER=openai_compatible задайте TEKTI_PANEL (например google:gemini-2.5-pro) и TEKTI_BASE_URL"
        )
    if not bench:
        raise ValueError("TEKTI_PANEL пуст")
    panel = bench[: PANEL_SIZE[mode]]

    def pick(var: str, default: ModelSpec) -> ModelSpec:
        value = e.get(var, "").strip()
        if value:
            return parse_spec(value, provider)
        return default if provider == "bedrock" and not e.get("TEKTI_PANEL", "").strip() else bench[0]

    segmenter = pick("TEKTI_SEGMENTER", DEFAULT_SEGMENTER)
    question_writer = pick("TEKTI_QUESTION_WRITER", DEFAULT_QUESTION_WRITER)

    lenses = list(Lens)
    if e.get("TEKTI_LENSES", "").strip():
        lenses = [Lens(x.strip()) for x in e["TEKTI_LENSES"].split(",") if x.strip()]

    base_url = e.get("TEKTI_BASE_URL", "").strip() or None
    if provider == "openai_compatible" and not base_url:
        raise ValueError("Для TEKTI_PROVIDER=openai_compatible задайте TEKTI_BASE_URL")
    api_key = e.get("TEKTI_API_KEY", "").strip() or None

    return Config(
        mode=mode,
        panel=panel,
        lenses=lenses,
        segmenter=segmenter,
        question_writer=question_writer,
        max_parallel=int(e.get("TEKTI_MAX_PARALLEL", "6")),
        max_output_tokens=int(e.get("TEKTI_MAX_OUTPUT_TOKENS", "12000")),
        bench=bench,
        backend=backend,  # type: ignore[arg-type]
        region=e.get("TEKTI_REGION") or e.get("AWS_REGION") or e.get("AWS_DEFAULT_REGION") or "us-east-1",
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key else None,
    )


def make_backend(cfg: Config):
    from tekti.llm_backends import ConverseBackend, StrandsBackend

    if cfg.backend == "converse":
        return ConverseBackend(region=cfg.region)
    return StrandsBackend(
        region=cfg.region,
        base_url=cfg.base_url,
        api_key=cfg.api_key.get_secret_value() if cfg.api_key else None,
    )


def install_backend(cfg: Config) -> None:
    """Install the configured backend behind tekti.llm.ask."""
    from tekti import llm

    llm.use_backend(make_backend(cfg))


def challenger_for(cfg: Config, raised_by_families: set[str]) -> ModelSpec:
    """A model from a family that did not raise the finding.

    Looks through the active panel first, then the rest of the bench (so fast mode can still
    challenge with another family); falls back to the strongest panel model.
    """
    for spec in [*cfg.panel, *cfg.bench]:
        if spec.family not in raised_by_families:
            return spec
    return cfg.panel[0]
