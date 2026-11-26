"""Smoke check: one tiny structured-output request per configured model.

Usage:
    uv run python scripts/smoke_providers.py [--mode board] [--models openai:gpt-6.2-sol,google:gemini-4-carbon]
        [--max-tokens 2000]

Reads the same settings as `tekti` (environment and `.env`): through the gateway when TEKTI_GATEWAY_URL is set
(owner token TEKTI_GATEWAY_TOKEN, else the `tekti login` session), else with the vendor keys, else Bedrock. Without --models it checks every model
the run would use (panel, challengers, segmenter, merge model, question writer). For each model it prints
ok/fail, latency, tokens and the estimated cost; never a key or token. Exit code 1 when any model fails.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from urllib.parse import urlsplit

from pydantic import BaseModel

from tekti import llm
from tekti.config import Config, estimate_cost, install_backend, load_config, parse_spec, price_of
from tekti.contracts import Mode
from tekti.llm import ModelSpec
from tekti.llm_backends import GatewayRefused


class Capital(BaseModel):
    country: str
    capital: str


SYSTEM = "You answer geography questions. Reply only through the required structured output."
USER = "What is the capital of France? Answer in the required schema."


def _route(cfg: Config, spec: ModelSpec) -> str:
    if spec.provider in ("openai", "anthropic", "google"):
        if cfg.gateway_url:
            return f"gateway {urlsplit(cfg.gateway_url).hostname}"
        return "direct"
    if spec.provider == "bedrock":
        return f"bedrock {cfg.region}"
    return f"openai_compatible {urlsplit(cfg.base_url or '').hostname}"


def _models(cfg: Config, override: str | None) -> list[ModelSpec]:
    if override:
        specs = [parse_spec(part, cfg.provider_mode) for part in override.split(",") if part.strip()]
    else:
        specs = cfg.all_models()
    unique: dict[tuple[str, str], ModelSpec] = {}
    for spec in specs:
        unique.setdefault((spec.provider, spec.model_id), spec)
    return list(unique.values())


def _redact(text: str, cfg: Config) -> str:
    """Vendor error texts can echo part of a key (OpenAI: "Incorrect API key provided: sk-ab***yz")."""
    secrets = [s.get_secret_value() for s in [*cfg.vendor_keys.values(), cfg.gateway_token, cfg.api_key] if s]
    for secret in secrets:
        text = text.replace(secret, "***")
    return re.sub(r"\b(sk|key|AIza)[-_A-Za-z0-9*]{6,}", r"\1-***", text)


async def _one(spec: ModelSpec, max_tokens: int, cfg: Config) -> tuple[bool, str]:
    try:
        answer = await llm.ask(spec, system=SYSTEM, user=USER, schema=Capital, stage="smoke", max_tokens=max_tokens)
    except llm.LLMError as exc:
        return False, _redact(str(exc), cfg)[:300]
    except GatewayRefused as exc:  # a BaseException: report it like any other failure
        return False, str(exc)
    ok = "paris" in answer.capital.lower()
    return ok, f"capital={answer.capital!r}" if ok else f"unexpected answer {answer.model_dump()}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", default="board", choices=[m.value for m in Mode])
    parser.add_argument("--models", help="comma-separated models in TEKTI_PANEL syntax; default: the configured ones")
    parser.add_argument("--max-tokens", type=int, default=2000, help="output cap per call, reasoning included")
    args = parser.parse_args()

    try:
        cfg = load_config(args.mode)
    except ValueError as exc:
        print(f"config error: {exc}")
        return 2
    install_backend(cfg)
    specs = _models(cfg, args.models)
    print(f"provider mode: {cfg.provider_mode}; {len(specs)} model(s); keys are not shown")
    failed = 0
    for spec in specs:
        llm.reset_calls()
        started = time.monotonic()
        ok, detail = await _one(spec, args.max_tokens, cfg)
        seconds = time.monotonic() - started
        calls = llm.calls()
        t_in = sum(c.input_tokens for c in calls)
        t_out = sum(c.output_tokens for c in calls)
        cost = f"${estimate_cost(calls):.5f}" if price_of(spec.model_id) else "unpriced"
        failed += not ok
        print(
            f"{'ok  ' if ok else 'FAIL'} {spec.provider}:{spec.model_id} [{_route(cfg, spec)}] "
            f"{seconds:.1f}s tokens in/out {t_in}/{t_out} cost {cost} - {detail}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
