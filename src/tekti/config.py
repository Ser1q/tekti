"""Run settings: which models sit on the panel, which backend talks to them.

Three ways to reach models (one run may mix them; each model carries its own provider):

* vendors  - OpenAI, Anthropic and Google APIs, either through the Tekti gateway
             (TEKTI_GATEWAY_URL plus the owner's TEKTI_GATEWAY_TOKEN or a `tekti login` GitHub session)
             or directly with the vendor keys
             (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY).
* bedrock  - Amazon Bedrock with the standard AWS credential chain (the earlier default; legacy).
* openai_compatible - any other endpoint speaking the OpenAI Chat Completions protocol (Ollama, ...).

Precedence (first match wins):
  1. Explicit TEKTI_PANEL / TEKTI_SEGMENTER / TEKTI_MERGE_MODEL / TEKTI_QUESTION_WRITER.
  2. TEKTI_PROVIDER, when set, picks the defaults: bedrock | vendors | openai_compatible.
  3. Otherwise (TEKTI_PROVIDER unset or "auto"): the gateway if TEKTI_GATEWAY_URL is set, else the
     vendor keys that are set (the vendor defaults filtered to those vendors), else Bedrock.
  With the gateway set, every openai/anthropic/google model goes through it, vendor keys or not.
  Gateway credential: TEKTI_GATEWAY_TOKEN (the owner's) if set, else the stored unexpired session for that
  gateway URL (`tekti login`); with neither, or an expired session, start-up stops asking for `tekti login`.

Environment variables (a `.env` in the working directory is read too; real environment variables win):

    TEKTI_PROVIDER         auto (default) | vendors | bedrock | openai_compatible
    TEKTI_GATEWAY_URL      https://<id>.lambda-url.<region>.on.aws
    TEKTI_GATEWAY_TOKEN    the owner's gateway token (never printed or logged); others use `tekti login`
    TEKTI_GITHUB_CLIENT_ID GitHub OAuth app client id for `tekti login` (default DEFAULT_GITHUB_CLIENT_ID)
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)   vendor keys (never printed)
    TEKTI_BACKEND          strands (default) | converse   (converse: Bedrock models only; others still go
                           through Strands)
    TEKTI_REGION           AWS region for Bedrock; default AWS_REGION, then us-east-1
    TEKTI_PANEL            comma-separated models, strongest first, one per family. Fast mode uses the first
                           entry, standard the first two, board the first three; the rest are challengers.
    TEKTI_SEGMENTER        one model for requirement segmentation
    TEKTI_MERGE_MODEL      one model for grouping duplicate findings (default: the question writer)
    TEKTI_QUESTION_WRITER  one model for clarifying questions
    TEKTI_LENSES           comma-separated subset of: functional,api_contract,data_integration,tester,developer
    TEKTI_MAX_PARALLEL     concurrent panel calls (default 6; 2 when Google is the only vendor, for free-tier limits)
    TEKTI_MAX_OUTPUT_TOKENS  answer budget per call (default 12000); vendor models get extra room for thinking
                           by level (low +4000, medium +12000, high +24000)
    TEKTI_REASONING        thinking level per stage, e.g. "panel=high,challenge=low,google:panel=medium";
                           stages: segment, panel, challenge, merge, questions, default; levels off/low/medium/high
    TEKTI_PROBE_TIMEOUT    seconds a Google model gets to answer a tiny request at start-up before the next one
                           in GOOGLE_FALLBACKS replaces it (default 45; two tries; 0 turns the check off)
    TEKTI_BASE_URL         OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 (Ollama)
    TEKTI_API_KEY          key for that endpoint (never printed or logged)

Model syntax (TEKTI_PANEL entries and the single-model variables):

    [provider:][family:]model_id

    provider  bedrock | openai | anthropic | google | openai_compatible
    family    used for vote counting and challenger choice; guessed from the model id when omitted
    model_id  may contain ':' (e.g. openai.gpt-oss-120b-1:0); after the optional provider, only the first ':'
              splits, and only when the part before it looks like a family name (no dots or slashes)

    openai:gpt-6.1-terra, anthropic:claude-haiku-5-5, google:gemini-4-neon       vendor APIs
    bedrock:anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0, bedrock:openai.gpt-oss-120b-1:0
    openai_compatible:qwen:qwen3:32b

  Backward compatibility: `openai:`, `anthropic:` and `google:` are also family names. They mean the vendor
  API only when the run uses vendor defaults (auto with a gateway or vendor key, or TEKTI_PROVIDER=vendors)
  and the model id does not look like a Bedrock id (`<vendor>.<model>`, optionally with a us./eu./... prefix).
  Otherwise they keep the old meaning: a family on the run's provider. So
  `openai:openai.gpt-oss-120b-1:0` stays a Bedrock model in every mode. In vendor mode an entry without a
  vendor prefix goes to Bedrock unless its id names a vendor model (gpt-..., claude-..., gemini-...).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from tekti.contracts import Lens, Mode, ModelCall
from tekti.llm import ModelSpec

VENDORS = ("openai", "anthropic", "google")
# Public client id of the author's GitHub OAuth app (device flow enabled, no scopes, no secret). The owner
# fills this in after registering the app; TEKTI_GITHUB_CLIENT_ID overrides it.
DEFAULT_GITHUB_CLIENT_ID = ""
# The author's gateway address. Not applied automatically yet: to make the gateway the default for everyone,
# use it in load_config as the fallback of TEKTI_GATEWAY_URL (one line, marked there).
DEFAULT_GATEWAY_URL = ""
NO_GATEWAY_LOGIN = "Нет входа в шлюз Tekti: выполните `uv run tekti login` и запустите проверку снова."
GATEWAY_SESSION_EXPIRED = "Сессия шлюза Tekti истекла: выполните `uv run tekti login` и запустите проверку снова."
ProviderMode = Literal["bedrock", "vendors", "openai_compatible"]

# --------------------------------------------------------------------------- Bedrock defaults (legacy)

# Defaults from the model spike (scripts/spike_models.py, 26 Nov 2026, Bedrock us-east-1). Strongest first,
# one per family; ordered so the first three are the most independent families that passed 3/3
# schema-valid runs in the spike.
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

# --------------------------------------------------------------------------- vendor defaults
#
# Prices read on the vendors' official pages on 26 Nov 2026, USD per 1M tokens, standard tier
# (not batch), prompts under every long-context threshold. Thinking/reasoning tokens are billed as output.
#
#   model                 input   output  cached in  source
#   OpenAI (https://developers.openai.com/api/docs/pricing; openai.com/api/pricing now shows seats only)
#   gpt-6.2-astra          8.00   40.00    0.40
#   gpt-6.2-sol            2.00   10.00    0.10
#   gpt-6.1-terra          1.00    5.00    0.05     <- panel seat 1, merge, questions
#   gpt-6-luna             0.10    0.50    0.01
#   Anthropic (https://platform.claude.com/docs/en/about-claude/pricing)
#   claude-fable-5-2       8.00   40.00    0.20
#   claude-opus-5-5        4.00   20.00    0.20
#   claude-sonnet-5-5      2.00   10.00    0.10
#   claude-haiku-5-5       0.10    0.50    0.01     <- panel seat 2, segmenter (prompt <=100k tokens;
#                                                      0.50/2.50 above). Claude 4.7+ tokenizer: ~30% more tokens.
#   Google Gemini API (https://ai.google.dev/gemini-api/docs/pricing -> aistudio.google.com/docs/pricing)
#   gemini-4-carbon        2.00   10.00    0.10     4.00/20.00 from 2026-12-01
#   gemini-4-argon         2.00   10.00    0.10     4.00/20.00 from 2026-12-01
#   gemini-4-neon          3.00   15.00    0.15
#   gemini-4-hydrogen      1.50    7.50    0.075    free tier <- the only-Google-key setup
#   gemini-3.8-flash       0.75    3.75    0.075    1.50/7.50 from 2027-01-01
#   gemini-3.5-flash-lite  0.30    2.50    0.03     free tier <- seat 3 fallback when Gemini 4 is overloaded
#
# Measured 26 Nov 2026 (single live calls on the local/requirements.md panel prompt, ~16-21k tokens in):
#   gpt-6.1-terra      review ~100 s, 5/5 valid in runs/vendors-1
#   claude-haiku-5-5   review 25-64 s at effort medium once the panel prompt stops asking for a visible
#                      "worksheet" (with it: stop_reason=refusal, category reasoning_extraction)
#   gemini-4-hydrogen  review 138-165 s at thinking LOW (default thinking used up the 12k cap in vendors-1);
#                      a trivial request took >60 s while overloaded
#   gemini-3.5-flash-lite  review 6-18 s, valid, 1-5 findings per review
#   gemini-4-carbon    <- panel seat 3. 26 Nov 2026 evening: 6 of 6 reviews valid, 12-17 findings each,
#                      50-196 s; side by side gemini-4-neon gave 9 per review and timed out on 3 of 5;
#                      gemini-4-hydrogen gave 5-6 and lost 2 of 5 reviews to "high demand"; gemini-3.8-flash failed
#   segmentation of local/requirements.md: claude-haiku-5-5 thinking off 58 and 62 requirements, 18-35 s;
#     at effort low 71 (181 s); gpt-6.1-terra low 91 (124 s), off 31; gpt-6-luna low 87; flash-lite 42
#
# Measured board run runs/vendors-2 (26 Nov 2026, these defaults; Gemini 4 Hydrogen did not answer the start-up
# check, so Gemini 3.5 Flash-Lite took seat 3; panel prompt without the "worksheet" field, see the report):
#   OpenAI gpt-6.1-terra $0.40, Anthropic claude-haiku-5-5 $0.08, Google gemini-3.5-flash-lite $0.05
#   -> $0.52 per board run, 351 s wall, 14 of 15 panel reviews valid. With Hydrogen in seat 3, Google adds
#   about $0.30 (estimate from single calls).
#
# Prompt caching is not enabled: the 15 panel calls share the document turn but each angle's system
# prompt comes first, so no two panel calls share a cacheable prefix. OpenAI and Gemini cache prefixes
# automatically anyway (they help the structured-output repair request, which repeats the prefix).
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-4-carbon": (4.00, 20.00),  # price from 2026-12-01
    "gemini-4-neon": (3.00, 15.00),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-3.8-flash": (1.50, 7.50),  # price from 2027-01-01 (0.75/3.75 until then)
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

_TERRA = ModelSpec(family="openai", model_id="gpt-6.1-terra", provider="openai")
_HAIKU = ModelSpec(family="anthropic", model_id="claude-haiku-5-5", provider="anthropic")
_HYDROGEN = ModelSpec(family="google", model_id="gemini-4-hydrogen", provider="google")
_CARBON = ModelSpec(family="google", model_id="gemini-4-carbon", provider="google")
_FLASH_LITE = ModelSpec(family="google", model_id="gemini-3.5-flash-lite", provider="google")
# Strongest first; the order also decides who challenges: a finding raised only by GPT is challenged by
# Claude, one raised by Claude or Gemini by GPT, one raised by both GPT and Claude by Gemini. Gemini sits
# last because it is the slowest to answer today (vendors-1: Gemini took 42 challenges, up to 263 s each).
VENDOR_BENCH: list[ModelSpec] = [_TERRA, _HAIKU, _CARBON]
# Role -> preferred models in order; the first whose vendor is configured is used. These hold whatever the
# panel is (an explicit TEKTI_PANEL does not move them), so segmentation stays comparable across runs.
VENDOR_ROLES: dict[str, list[ModelSpec]] = {
    "segmenter": [_HAIKU, _TERRA, _CARBON],
    "merge_model": [_TERRA, _HAIKU, _CARBON],
    "question_writer": [_TERRA, _CARBON, _HAIKU],
}
# Google as the only vendor (typically a free AI Studio key): both Hydrogen and Flash-Lite have a free tier.
GOOGLE_ONLY = _HYDROGEN
# When a Google model does not answer a tiny request within TEKTI_PROBE_TIMEOUT seconds at start-up (Gemini 4
# answering "high demand"), the next one in this chain takes its place for the whole run.
GOOGLE_FALLBACKS = ["gemini-4-carbon", "gemini-4-hydrogen", "gemini-3.5-flash-lite"]
# Per call: when one request to a Gemini 4 model fails or runs out of time during the run, that request is
# sent once to this model instead (tekti.llm.use_fallbacks). It answers in seconds and was never throttled
# in the runs of 26 Nov 2026, so a loaded Gemini 4 costs one weaker answer, not a lost reviewer.
GOOGLE_CALL_STAND_IN = _FLASH_LITE

# Thinking level per stage ("provider:stage" overrides "stage"); see llm_backends/providers.py.
# off = no thinking where the model allows it. Override with TEKTI_REASONING, e.g. "panel=high,challenge=low".
DEFAULT_REASONING: dict[str, str] = {
    "segment": "off",  # Haiku 5.5 off: 58-62 atomic requirements, exact quotes, ~30 s
    "panel": "medium",
    "google:panel": "low",  # Gemini 4 at its default level spent the whole 12k cap on thinking
    "challenge": "low",
    "merge": "low",
    "questions": "medium",
    "default": "low",
}

PANEL_SIZE = {Mode.FAST: 1, Mode.STANDARD: 2, Mode.BOARD: 3}

# Model id prefixes -> family, for ids given without a family.
_FAMILY_HINTS = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
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
_PROVIDER_PREFIXES = ("bedrock", "openai_compatible", *VENDORS)
# Bedrock ids are "<vendor>.<model>", optionally with a cross-region prefix; vendor ids never have a
# purely alphabetic segment before their first dot (gpt-6.1-terra, gemini-3.8-flash, claude-haiku-5-5).
_BEDROCK_ID = re.compile(r"^(?:(?:us|eu|apac|global|jp|au|ca|us-gov)\.)?[a-z][a-z0-9]*\.[a-z0-9]")


class Config(BaseModel):
    mode: Mode
    panel: list[ModelSpec]  # one per family; fast=1, standard=2, board=3
    lenses: list[Lens]  # all five by default
    segmenter: ModelSpec
    question_writer: ModelSpec
    max_parallel: int  # concurrency cap for panel calls
    max_output_tokens: int
    merge_model: ModelSpec | None = None  # None: adjudication uses the question writer
    reasoning: dict[str, str] = Field(default_factory=dict)  # stage or "provider:stage" -> off/low/medium/high
    # Backend settings (not part of the review logic).
    bench: list[ModelSpec] = Field(default_factory=list)  # every configured family, strongest first
    provider_mode: ProviderMode = "bedrock"  # which defaults were used
    backend: Literal["strands", "converse"] = "strands"
    region: str = "us-east-1"
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    gateway_url: str | None = None
    gateway_token: SecretStr | None = Field(default=None, exclude=True, repr=False)  # owner token or session
    gateway_auth: Literal["owner_token", "github_session"] | None = None
    gateway_login: str | None = None  # GitHub login of the stored session
    gateway_expires_at: datetime | None = None  # session expiry
    vendor_keys: dict[str, SecretStr] = Field(default_factory=dict, exclude=True, repr=False)

    def vendors_available(self) -> set[str]:
        return set(VENDORS) if self.gateway_url and self.gateway_token else set(self.vendor_keys)

    def all_models(self) -> list[ModelSpec]:
        roles = [self.segmenter, self.question_writer, *([self.merge_model] if self.merge_model else [])]
        return [*self.bench, *self.panel, *roles]


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


def looks_like_bedrock_id(model_id: str) -> bool:
    return bool(_BEDROCK_ID.match(model_id.strip().lower()))


def _family_and_model(text: str) -> tuple[str, str]:
    """'family:model_id' or a bare model id; only a first part without dots or slashes is a family."""
    text = text.strip()
    family, sep, rest = text.partition(":")
    if sep and family and "." not in family and "/" not in family and rest:
        return family.strip().lower(), rest.strip()
    return _guess_family(text), text


def parse_spec(text: str, provider: str) -> ModelSpec:
    """Parse one model entry (syntax in the module docstring).

    `provider` is the run's default: "bedrock" or "openai_compatible" (entries without an explicit
    provider use it) or "vendors" (vendor model ids go to their vendor, anything else to Bedrock).
    """
    text = text.strip()
    head, sep, rest = text.partition(":")
    head = head.strip().lower()
    if sep and rest.strip() and head in ("bedrock", "openai_compatible"):
        family, model_id = _family_and_model(rest)
        return ModelSpec(family=family, model_id=model_id, provider=head)  # type: ignore[arg-type]
    if sep and rest.strip() and head in VENDORS:
        model_id = rest.strip()
        if provider != "vendors":
            return ModelSpec(family=head, model_id=model_id, provider=provider)  # type: ignore[arg-type]
        target = "bedrock" if looks_like_bedrock_id(model_id) else head
        return ModelSpec(family=head, model_id=model_id, provider=target)  # type: ignore[arg-type]
    family, model_id = _family_and_model(text)
    if provider != "vendors":
        return ModelSpec(family=family, model_id=model_id, provider=provider)  # type: ignore[arg-type]
    target = family if family in VENDORS and not looks_like_bedrock_id(model_id) else "bedrock"
    return ModelSpec(family=family, model_id=model_id, provider=target)  # type: ignore[arg-type]


def parse_panel(text: str, provider: str) -> list[ModelSpec]:
    specs = [parse_spec(part, provider) for part in text.split(",") if part.strip()]
    seen: set[str] = set()
    unique = []
    for spec in specs:  # one per family: keep the first (strongest)
        if spec.family not in seen:
            seen.add(spec.family)
            unique.append(spec)
    return unique


def gateway_url_from(e: Mapping[str, str]) -> str | None:
    """TEKTI_GATEWAY_URL without a trailing slash, or None. To bake in the author's gateway, return
    `... or DEFAULT_GATEWAY_URL or None` here."""
    url = e.get("TEKTI_GATEWAY_URL", "").strip().rstrip("/") or None
    if url and not url.startswith(("https://", "http://")):
        raise ValueError(f"TEKTI_GATEWAY_URL должен начинаться с https://, получено: {url!r}")
    return url


def github_client_id(e: Mapping[str, str]) -> str:
    return e.get("TEKTI_GITHUB_CLIENT_ID", "").strip() or DEFAULT_GITHUB_CLIENT_ID


def env_settings(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The merged settings (`.env` under the process environment) that load_config reads."""
    return _env(env)


def _vendor_keys(e: Mapping[str, str]) -> dict[str, SecretStr]:
    keys = {
        "openai": e.get("OPENAI_API_KEY", "").strip(),
        "anthropic": e.get("ANTHROPIC_API_KEY", "").strip(),
        "google": (e.get("GOOGLE_API_KEY", "").strip() or e.get("GEMINI_API_KEY", "").strip()),
    }
    return {vendor: SecretStr(key) for vendor, key in keys.items() if key}


def parse_reasoning(text: str) -> dict[str, str]:
    from tekti.llm_backends.providers import REASONING_LEVELS

    out: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, sep, level = part.partition("=")
        level = level.strip().lower()
        if not sep or level not in REASONING_LEVELS:
            raise ValueError(
                f"TEKTI_REASONING: ожидается этап=уровень ({'/'.join(REASONING_LEVELS)}), получено: {part!r}"
            )
        out[key.strip().lower()] = level
    return out


def _vendor_defaults(available: set[str]) -> tuple[list[ModelSpec], dict[str, ModelSpec]]:
    if available == {"google"}:
        return [GOOGLE_ONLY.model_copy()], {role: GOOGLE_ONLY.model_copy() for role in VENDOR_ROLES}
    bench = [s.model_copy() for s in VENDOR_BENCH if s.provider in available]
    roles = {}
    for role, prefs in VENDOR_ROLES.items():
        roles[role] = next(s for s in prefs if s.provider in available).model_copy()
    return bench, roles


def load_config(mode: Mode | str = Mode.BOARD, env: Mapping[str, str] | None = None) -> Config:
    """Build the run config from defaults, `.env` and environment variables (see module docstring).

    Pass `env` to bypass the process environment and `.env` (tests do this).
    """
    mode = Mode(mode)
    e = _env(env)

    gateway_url = gateway_url_from(e)
    owner_token = e.get("TEKTI_GATEWAY_TOKEN", "").strip() or None
    if owner_token and not gateway_url:
        raise ValueError("TEKTI_GATEWAY_TOKEN задан без TEKTI_GATEWAY_URL")
    vendor_keys = _vendor_keys(e)
    available = set(VENDORS) if gateway_url else set(vendor_keys)

    requested = (e.get("TEKTI_PROVIDER", "").strip() or "auto").lower()
    if requested not in ("auto", "vendors", "bedrock", "openai_compatible"):
        raise ValueError(
            f"TEKTI_PROVIDER должен быть auto, vendors, bedrock или openai_compatible, получено: {requested!r}"
        )
    provider: ProviderMode
    if requested == "auto":
        provider = "vendors" if available else "bedrock"
    else:
        provider = requested  # type: ignore[assignment]
    if provider == "vendors" and not available:
        raise ValueError(
            "TEKTI_PROVIDER=vendors: задайте TEKTI_GATEWAY_URL (и выполните `tekti login`) или хотя бы один ключ "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY"
        )

    gateway_token: SecretStr | None = None
    gateway_auth = gateway_login = None
    gateway_expires_at: datetime | None = None
    if gateway_url and provider == "vendors":
        if owner_token:
            gateway_token, gateway_auth = SecretStr(owner_token), "owner_token"
        else:
            from tekti.llm_backends.gateway_auth import load_session

            session = load_session(gateway_url, e)
            if session is None:
                raise ValueError(NO_GATEWAY_LOGIN)
            if session.expired():
                raise ValueError(GATEWAY_SESSION_EXPIRED)
            gateway_token, gateway_auth = session.token, "github_session"
            gateway_login, gateway_expires_at = session.login, session.expires_at

    backend = e.get("TEKTI_BACKEND", "strands").strip() or "strands"
    if backend not in ("strands", "converse"):
        raise ValueError(f"TEKTI_BACKEND должен быть strands или converse, получено: {backend!r}")
    if backend == "converse" and provider == "openai_compatible":
        raise ValueError("TEKTI_BACKEND=converse работает только с моделями Bedrock")

    explicit_panel = e.get("TEKTI_PANEL", "").strip()
    roles: dict[str, ModelSpec] = {}
    if explicit_panel:
        bench = parse_panel(explicit_panel, provider)
        if provider == "vendors":  # roles keep their own defaults; only the panel was overridden
            roles = _vendor_defaults(available)[1]
    elif provider == "bedrock":
        bench = [s.model_copy() for s in DEFAULT_BENCH]
        roles = {"segmenter": DEFAULT_SEGMENTER, "question_writer": DEFAULT_QUESTION_WRITER}
    elif provider == "vendors":
        bench, roles = _vendor_defaults(available)
    else:
        raise ValueError(
            "Для TEKTI_PROVIDER=openai_compatible задайте TEKTI_PANEL (например qwen:qwen3:32b) и TEKTI_BASE_URL"
        )
    if not bench:
        raise ValueError("TEKTI_PANEL пуст")
    panel = bench[: PANEL_SIZE[mode]]

    def pick(var: str, role: str) -> ModelSpec | None:
        value = e.get(var, "").strip()
        if value:
            return parse_spec(value, provider)
        if role in roles:
            return roles[role]
        return None if role == "merge_model" else bench[0]

    segmenter = pick("TEKTI_SEGMENTER", "segmenter")
    question_writer = pick("TEKTI_QUESTION_WRITER", "question_writer")
    merge_model = pick("TEKTI_MERGE_MODEL", "merge_model")
    assert segmenter is not None and question_writer is not None

    lenses = list(Lens)
    if e.get("TEKTI_LENSES", "").strip():
        lenses = [Lens(x.strip()) for x in e["TEKTI_LENSES"].split(",") if x.strip()]

    base_url = e.get("TEKTI_BASE_URL", "").strip() or None
    api_key = e.get("TEKTI_API_KEY", "").strip() or None

    default_parallel = "2" if provider == "vendors" and available == {"google"} else "6"
    reasoning = {**DEFAULT_REASONING, **parse_reasoning(e.get("TEKTI_REASONING", ""))}
    cfg = Config(
        mode=mode,
        panel=panel,
        lenses=lenses,
        segmenter=segmenter,
        question_writer=question_writer,
        merge_model=merge_model,
        reasoning=reasoning,
        max_parallel=int(e.get("TEKTI_MAX_PARALLEL", "").strip() or default_parallel),
        max_output_tokens=int(e.get("TEKTI_MAX_OUTPUT_TOKENS", "").strip() or "12000"),
        bench=bench,
        provider_mode=provider,
        backend=backend,  # type: ignore[arg-type]
        region=e.get("TEKTI_REGION") or e.get("AWS_REGION") or e.get("AWS_DEFAULT_REGION") or "us-east-1",
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key else None,
        gateway_url=gateway_url,
        gateway_token=gateway_token,
        gateway_auth=gateway_auth,
        gateway_login=gateway_login,
        gateway_expires_at=gateway_expires_at,
        vendor_keys=vendor_keys,
    )
    _check_reachable(cfg)
    return cfg


def _check_reachable(cfg: Config) -> None:
    """Fail at start-up, not after an hour of panel calls, when a configured model has no way to be reached."""
    available = cfg.vendors_available()
    hints = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}
    for spec in cfg.all_models():
        if spec.provider in VENDORS and spec.provider not in available:
            raise ValueError(
                f"Модель {spec.provider}:{spec.model_id} требует {hints[spec.provider]} "
                "или TEKTI_GATEWAY_URL со входом `tekti login`"
            )
        if spec.provider == "openai_compatible" and not cfg.base_url:
            raise ValueError(f"Модель {spec.model_id} (openai_compatible) требует TEKTI_BASE_URL")


def vendor_endpoints(cfg: Config) -> dict:
    """How each vendor is reached: through the gateway when configured, else with its own key."""
    from tekti.llm_backends.providers import VendorEndpoint, gateway_endpoint

    if cfg.gateway_url and cfg.gateway_token:
        return {v: gateway_endpoint(v, cfg.gateway_url, cfg.gateway_token) for v in VENDORS}
    return {v: VendorEndpoint(api_key=key) for v, key in cfg.vendor_keys.items()}


def make_backend(cfg: Config):
    from tekti.llm_backends import ConverseBackend, RoutingBackend, StrandsBackend

    strands = StrandsBackend(
        region=cfg.region,
        base_url=cfg.base_url,
        api_key=cfg.api_key.get_secret_value() if cfg.api_key else None,
        vendors=vendor_endpoints(cfg),
        reasoning=cfg.reasoning,
    )
    if cfg.backend == "converse":
        converse = ConverseBackend(region=cfg.region)
        if all(s.provider == "bedrock" for s in cfg.all_models()):
            return converse
        return RoutingBackend({"bedrock": converse}, default=strands)
    return strands


def install_backend(cfg: Config) -> None:
    """Install the configured backend behind tekti.llm.ask."""
    from tekti import llm

    llm.use_backend(make_backend(cfg))
    llm.use_fallbacks(
        {
            ("google", model_id): GOOGLE_CALL_STAND_IN
            for model_id in GOOGLE_FALLBACKS
            if model_id != GOOGLE_CALL_STAND_IN.model_id
        }
    )


# --------------------------------------------------------------------------- start-up check of Google models


def probe_timeout(env: Mapping[str, str] | None = None) -> float:
    e = _env(env)
    try:
        return max(0.0, float(e.get("TEKTI_PROBE_TIMEOUT", "").strip() or "45"))
    except ValueError:
        return 45.0


async def choose_responsive_models(cfg: Config, timeout: float = 45.0, backend=None) -> tuple[Config, list[str]]:
    """Replace a Google model that does not answer a tiny request within `timeout` seconds by the next model in
    GOOGLE_FALLBACKS, for the whole run (so the passport's model list shows what really ran).

    Returns the new config and one Russian line per decision. A gateway refusal propagates (GatewayRefused).
    """
    import asyncio

    if timeout <= 0:
        return cfg, []
    backend = backend or make_backend(cfg)

    class _Probe(BaseModel):
        capital: str

    async def answers(model_id: str) -> bool:
        # Two tries: Gemini 4 usually answers this in 5-20 s but one request in three is much slower
        # (measured 26 Nov 2026), and a single slow answer should not cost the seat for the whole run.
        return await answers_once(model_id) or await answers_once(model_id)

    async def answers_once(model_id: str) -> bool:
        spec = ModelSpec(family="google", model_id=model_id, provider="google")
        try:
            await asyncio.wait_for(
                backend.complete(
                    spec, system="Answer briefly.", user="Capital of France?", schema=_Probe, max_tokens=200
                ),
                timeout,
            )
            return True
        except Exception:  # noqa: BLE001 - overloaded, timed out or refused: try the next model
            return False

    used = {s.model_id for s in cfg.all_models() if s.provider == "google" and s.model_id in GOOGLE_FALLBACKS}
    if not used:
        return cfg, []
    notes: list[str] = []
    replace: dict[str, str] = {}
    for model_id in sorted(used, key=GOOGLE_FALLBACKS.index):
        for candidate in GOOGLE_FALLBACKS[GOOGLE_FALLBACKS.index(model_id) :]:
            if await answers(candidate):
                if candidate != model_id:
                    replace[model_id] = candidate
                    notes.append(f"Google: {model_id} не ответила за {timeout:.0f} с, вместо неё — {candidate}")
                break
        else:
            notes.append(f"Google: ни одна модель из {', '.join(GOOGLE_FALLBACKS)} не ответила за {timeout:.0f} с")
    if not replace:
        return cfg, notes

    def swap(spec: ModelSpec | None) -> ModelSpec | None:
        if spec is None or spec.provider != "google" or spec.model_id not in replace:
            return spec
        return spec.model_copy(update={"model_id": replace[spec.model_id]})

    update = {
        "panel": [swap(s) for s in cfg.panel],
        "bench": [swap(s) for s in cfg.bench],
        "segmenter": swap(cfg.segmenter),
        "question_writer": swap(cfg.question_writer),
        "merge_model": swap(cfg.merge_model),
    }
    return cfg.model_copy(update=update), notes


def challenger_for(cfg: Config, raised_by_families: set[str]) -> ModelSpec:
    """A model from a family that did not raise the finding.

    Looks through the active panel first, then the rest of the bench (so fast mode can still
    challenge with another family); falls back to the strongest panel model.
    """
    for spec in [*cfg.panel, *cfg.bench]:
        if spec.family not in raised_by_families:
            return spec
    return cfg.panel[0]


# --------------------------------------------------------------------------- cost


def price_of(model_id: str, prices: Mapping[str, tuple[float, float]] | None = None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens: exact id, else the longest priced id the model id starts with
    (dated snapshots, "models/" prefixes). None when unpriced."""
    table = PRICES_PER_1M if prices is None else prices
    mid = model_id.strip().lower().removeprefix("models/")
    if mid in table:
        return table[mid]
    matches = [k for k in table if mid.startswith(k)]
    return table[max(matches, key=len)] if matches else None


def estimate_cost(calls: Iterable[ModelCall], prices: Mapping[str, tuple[float, float]] | None = None) -> float:
    """Estimated USD for these calls at list prices (`PRICES_PER_1M`). Unpriced models count as 0;
    use `unpriced_models` to see which. Cache discounts, batch tiers and long-context surcharges are ignored."""
    total = 0.0
    for call in calls:
        rates = price_of(call.model, prices)
        if rates:
            total += (call.input_tokens * rates[0] + call.output_tokens * rates[1]) / 1_000_000
    return total


def unpriced_models(calls: Iterable[ModelCall], prices: Mapping[str, tuple[float, float]] | None = None) -> list[str]:
    return list(dict.fromkeys(c.model for c in calls if price_of(c.model, prices) is None))
