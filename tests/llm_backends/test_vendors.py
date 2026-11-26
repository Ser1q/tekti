"""Per-model providers, the gateway wire format, model syntax, precedence, secrets and cost.

The wire tests run each vendor SDK through Strands against a local imitation of the gateway
(fake_vendor.py) and check the exact path, auth header and body shape the gateway's allowlist relies on.
"""

import json

import pytest
from fake_vendor import FakeVendor
from pydantic import BaseModel, SecretStr

from tekti.config import (
    DEFAULT_BENCH,
    GOOGLE_ONLY,
    VENDOR_BENCH,
    challenger_for,
    estimate_cost,
    load_config,
    make_backend,
    parse_panel,
    parse_spec,
    price_of,
    unpriced_models,
    vendor_endpoints,
)
from tekti.contracts import Mode, ModelCall
from tekti.llm import LLMError, ModelSpec
from tekti.llm_backends import GatewayRefused, RoutingBackend, StrandsBackend, VendorEndpoint, gateway_endpoint
from tekti.llm_backends import strands_backend as sb
from tekti.llm_backends._common import is_gateway_refusal, is_retryable, sends_temperature

TOKEN = "gw-token-not-real-123"
KEYS = {"OPENAI_API_KEY": "sk-openai-not-real", "ANTHROPIC_API_KEY": "sk-ant-not-real", "GOOGLE_API_KEY": "g-not-real"}


class Answer(BaseModel):
    capital: str
    confidence: int


OPENAI = ModelSpec(family="openai", model_id="gpt-6.1-terra", provider="openai")
ANTHROPIC = ModelSpec(family="anthropic", model_id="claude-haiku-5-5", provider="anthropic")
GOOGLE = ModelSpec(family="google", model_id="gemini-4-neon", provider="google")


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(sb, "RETRY_INITIAL_DELAY", 0)


def openai_args(model) -> dict:
    """client_args of the buffered OpenAI client (providers.BufferedOpenAIClient)."""
    return model._custom_client.responses._client_args


def gateway_backend(url: str) -> StrandsBackend:
    return StrandsBackend(
        vendors={v: gateway_endpoint(v, url, SecretStr(TOKEN)) for v in ("openai", "anthropic", "google")}
    )


# ----------------------------------------------------------------------------- model construction / dispatch


def test_dispatch_builds_the_strands_model_of_each_provider():
    from strands.models import BedrockModel
    from strands.models.anthropic import AnthropicModel
    from strands.models.gemini import GeminiModel
    from strands.models.openai import OpenAIModel
    from strands.models.openai_responses import OpenAIResponsesModel

    backend = StrandsBackend(
        region="us-east-1",
        base_url="http://localhost:11434/v1",
        api_key="k",
        vendors={v: VendorEndpoint(api_key=SecretStr("k-" + v)) for v in ("openai", "anthropic", "google")},
    )
    cases = [
        (DEFAULT_BENCH[0], BedrockModel),
        (OPENAI, OpenAIResponsesModel),
        (ANTHROPIC, AnthropicModel),
        (GOOGLE, GeminiModel),
        (ModelSpec(family="qwen", model_id="qwen3:32b", provider="openai_compatible"), OpenAIModel),
    ]
    for spec, cls in cases:
        assert isinstance(backend._model(spec, 777), cls), spec


def test_vendor_models_carry_explicit_max_tokens_and_temperature_policy():
    backend = StrandsBackend(
        vendors={v: VendorEndpoint(api_key=SecretStr("k")) for v in ("openai", "anthropic", "google")}
    )
    # GPT-6.x reasoning models, every Claude via API version 2026-10-21, Gemini 4: no temperature sent
    assert backend._model(OPENAI, 500).get_config()["params"] == {"max_output_tokens": 500}
    anthropic_cfg = backend._model(ANTHROPIC, 500).get_config()
    assert anthropic_cfg["max_tokens"] == 500 and "temperature" not in (anthropic_cfg.get("params") or {})
    assert backend._model(GOOGLE, 500).get_config()["params"] == {"max_output_tokens": 500}
    # older models still get temperature 0
    old_openai = ModelSpec(family="openai", model_id="gpt-4.1", provider="openai")
    old_google = ModelSpec(family="google", model_id="gemini-3.5-flash-lite", provider="google")
    assert backend._model(old_openai, 9).get_config()["params"] == {"max_output_tokens": 9, "temperature": 0.0}
    assert backend._model(old_google, 9).get_config()["params"] == {"max_output_tokens": 9, "temperature": 0.0}


def test_temperature_policy():
    assert not sends_temperature("anthropic", "claude-haiku-4-5-20251001")
    assert not sends_temperature("openai", "gpt-6.2-sol") and sends_temperature("openai", "gpt-4.1")
    assert not sends_temperature("google", "gemini-4-hydrogen") and not sends_temperature("google", "gemini-3.8-flash")
    assert sends_temperature("google", "gemini-3.5-flash-lite")
    assert sends_temperature("bedrock", "openai.gpt-oss-120b-1:0")
    assert sends_temperature("openai_compatible", "qwen3:32b")


def test_vendor_without_endpoint_is_a_clear_error():
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        StrandsBackend()._model(OPENAI, 100)


def test_gateway_base_urls_and_key_per_vendor():
    backend = gateway_backend("https://abc.lambda-url.eu-central-1.on.aws/")
    openai_model = backend._model(OPENAI, 100)
    assert openai_args(openai_model)["base_url"] == "https://abc.lambda-url.eu-central-1.on.aws/openai/v1"
    assert openai_args(openai_model)["api_key"] == TOKEN and openai_args(openai_model)["max_retries"] == 0
    anthropic_model = backend._model(ANTHROPIC, 100)
    assert str(anthropic_model.client.base_url).rstrip("/") == "https://abc.lambda-url.eu-central-1.on.aws/anthropic"
    assert anthropic_model.client.api_key == TOKEN
    google_model = backend._model(GOOGLE, 100)
    opts = google_model.client_args["http_options"]
    assert opts.base_url == "https://abc.lambda-url.eu-central-1.on.aws/google" and opts.api_version == "v1beta"
    assert google_model.client_args["api_key"] == TOKEN and google_model.client_args["vertexai"] is False


def test_direct_mode_uses_vendor_defaults_and_vendor_keys():
    cfg = load_config(env=KEYS)
    endpoints = vendor_endpoints(cfg)
    assert {v: e.base_url for v, e in endpoints.items()} == {"openai": None, "anthropic": None, "google": None}
    assert endpoints["anthropic"].api_key.get_secret_value() == "sk-ant-not-real"
    backend = make_backend(cfg)
    assert "base_url" not in openai_args(backend._model(OPENAI, 10))
    assert str(backend._model(ANTHROPIC, 10).client.base_url).startswith("https://api.anthropic.com")


# ----------------------------------------------------------------------------- wire format through the gateway


@pytest.mark.parametrize("buffered", [True, False], ids=["gateway-buffered", "vendor-chunked"])
async def test_wire_format_per_vendor(buffered):
    with FakeVendor(answer={"capital": "Paris", "confidence": 9}, buffered=buffered) as fake:
        backend = gateway_backend(fake.url)
        results = {}
        for spec in (OPENAI, ANTHROPIC, GOOGLE):
            results[spec.provider] = await backend.complete(
                spec, system="Answer briefly.", user="Capital of France?", schema=Answer, max_tokens=500
            )
    for parsed, _, _ in results.values():
        assert parsed == Answer(capital="Paris", confidence=9)
    assert results["openai"][1:] == (120, 30)  # usage.input_tokens / output_tokens
    assert results["anthropic"][1:] == (200, 40)  # message_start input + message_delta output
    assert results["google"][1:] == (300, 35)  # promptTokenCount / candidates + thoughts

    openai_req, anthropic_req, google_req = fake.seen
    assert (openai_req.path, openai_req.query) == ("/openai/v1/responses", "")
    assert openai_req.headers["authorization"] == f"Bearer {TOKEN}"
    assert openai_req.body["model"] == "gpt-6.1-terra" and openai_req.body["stream"] is True
    assert openai_req.body["store"] is False and openai_req.body["max_output_tokens"] == 500
    assert openai_req.body["tools"][0]["type"] == "function" and "temperature" not in openai_req.body

    assert anthropic_req.path == "/anthropic/v1/messages"
    # anthropic SDK 2.x sends the key as a bearer token, not x-api-key (the contract accepts both)
    assert anthropic_req.headers["authorization"] == f"Bearer {TOKEN}" and "x-api-key" not in anthropic_req.headers
    assert anthropic_req.headers["anthropic-version"] == "2026-10-21"
    assert anthropic_req.body["model"] == "claude-haiku-5-5" and anthropic_req.body["max_tokens"] == 500
    assert anthropic_req.body["stream"] is True and "temperature" not in anthropic_req.body

    assert google_req.path == "/google/v1beta/models/gemini-4-neon:streamGenerateContent"
    assert google_req.query == "alt=sse"
    assert google_req.headers["x-goog-api-key"] == TOKEN and "authorization" not in google_req.headers
    assert google_req.body["generationConfig"] == {"maxOutputTokens": 500}
    assert "model" not in google_req.body  # the gateway reads the Google model from the path


@pytest.mark.parametrize("status", [401, 403, 429])
async def test_gateway_refusals_are_not_retried(status):
    with FakeVendor(answer={}, refuse_with=status) as fake:
        for spec in (OPENAI, ANTHROPIC, GOOGLE):
            backend = gateway_backend(fake.url)  # fresh: a refusal stops every later call of a backend
            before = len(fake.seen)
            with pytest.raises(GatewayRefused) as info:
                await backend.complete(spec, system="s", user="u", schema=Answer, max_tokens=50)
            sdk_error = info.value.__cause__
            assert is_gateway_refusal(sdk_error) and not is_retryable(sdk_error), spec.provider
            assert info.value.status == status and info.value.code == "cap_reached", spec.provider
            assert len(fake.seen) - before == 1, spec.provider


async def test_vendor_overload_is_retried():
    with FakeVendor(answer={"capital": "Paris", "confidence": 1}, fail_first=1) as fake:
        parsed, _, _ = await gateway_backend(fake.url).complete(
            OPENAI, system="s", user="u", schema=Answer, max_tokens=50
        )
    assert parsed.capital == "Paris" and len(fake.seen) == 2


@pytest.mark.parametrize("spec", [OPENAI, ANTHROPIC, GOOGLE], ids=lambda s: s.provider)
async def test_temperature_drop_and_retry_per_vendor(spec, monkeypatch):
    # force temperature on, as for an older model of each vendor; the fake refuses it in the vendor's error shape
    monkeypatch.setattr(sb, "sends_temperature", lambda provider, model_id: True)
    with FakeVendor(answer={"capital": "Paris", "confidence": 2}, reject_temperature=True) as fake:
        backend = gateway_backend(fake.url)
        parsed, _, _ = await backend.complete(spec, system="s", user="u", schema=Answer, max_tokens=50)
        assert parsed.capital == "Paris"
        # Anthropic: Strands refuses sampling parameters locally for API version 2026-10-21 (ValueError
        # "... remove them ..."), so the refused attempt never reaches the network.
        refused = 0 if spec.provider == "anthropic" else 1
        assert len(fake.seen) == refused + 1
        await backend.complete(spec, system="s", user="u", schema=Answer, max_tokens=50)
    assert len(fake.seen) == refused + 2  # learned: no temperature from then on
    bodies = [json.dumps(s.body) for s in fake.seen]
    assert all('"temperature"' in b for b in bodies[:refused])
    assert all('"temperature"' not in b for b in bodies[refused:])


def test_retry_classification_for_sdk_errors():
    import httpx
    import openai
    from google.genai import errors as genai_errors

    request = httpx.Request("POST", "https://x/openai/v1/responses")

    def openai_error(status, body):
        return openai.APIStatusError("boom", response=httpx.Response(status, request=request), body=body)

    assert is_retryable(openai_error(500, {"message": "server"}))
    assert is_retryable(openai_error(529, None))
    assert not is_retryable(openai_error(400, {"message": "bad"}))
    gateway_cap = openai_error(429, {"type": "tekti_gateway", "code": "cap_reached", "message": "cap"})
    assert is_gateway_refusal(gateway_cap) and not is_retryable(gateway_cap)
    assert is_retryable(openai_error(502, {"type": "tekti_gateway", "code": "upstream", "message": "x"}))
    assert is_retryable(openai.APITimeoutError(request=request))
    assert is_retryable(
        genai_errors.ServerError(503, {"error": {"code": 503, "message": "x", "status": "UNAVAILABLE"}})
    )
    assert not is_retryable(genai_errors.ClientError(400, {"error": {"code": 400, "message": "x"}}))


# ----------------------------------------------------------------------------- model syntax


def test_old_family_model_syntax_unchanged():
    specs = parse_panel(
        "openai:openai.gpt-oss-120b-1:0,anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"
    )
    assert [(s.family, s.model_id, s.provider) for s in specs] == [
        ("openai", "openai.gpt-oss-120b-1:0", "bedrock"),
        ("anthropic", "us.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock"),
    ]
    compat = parse_spec("google:gemini-2.5-pro", "openai_compatible")
    assert (compat.family, compat.provider) == ("google", "openai_compatible")


def test_provider_prefixed_syntax():
    s = parse_spec("bedrock:anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0", "vendors")
    assert (s.provider, s.family, s.model_id) == ("bedrock", "anthropic", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    s = parse_spec("bedrock:openai.gpt-oss-120b-1:0", "vendors")
    assert (s.provider, s.family, s.model_id) == ("bedrock", "openai", "openai.gpt-oss-120b-1:0")
    s = parse_spec("openai_compatible:qwen:qwen3:32b", "bedrock")
    assert (s.provider, s.family, s.model_id) == ("openai_compatible", "qwen", "qwen3:32b")
    for text, provider in [
        ("openai:gpt-6.2-sol", "openai"),
        ("anthropic:claude-sonnet-5-5", "anthropic"),
        ("google:gemini-4-carbon", "google"),
        ("gpt-6-luna", "openai"),
        ("claude-haiku-5-5", "anthropic"),
        ("gemini-3.8-flash", "google"),
    ]:
        s = parse_spec(text, "vendors")
        assert (s.provider, s.family) == (provider, provider), text


def test_vendor_mode_keeps_bedrock_ids_on_bedrock():
    for text in (
        "openai:openai.gpt-oss-120b-1:0",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "mistral:mistral.mistral-large-3-675b-instruct",
    ):
        assert parse_spec(text, "vendors").provider == "bedrock", text


def test_mixed_panel_in_one_run():
    env = {
        **KEYS,
        "TEKTI_PANEL": "google:gemini-4-neon,bedrock:mistral:mistral.mistral-large-3-675b-instruct,anthropic:claude-haiku-5-5",
    }
    cfg = load_config(Mode.BOARD, env=env)
    assert [s.provider for s in cfg.panel] == ["google", "bedrock", "anthropic"]
    backend = make_backend(cfg)
    assert isinstance(backend, StrandsBackend)
    converse = make_backend(load_config(Mode.BOARD, env={**env, "TEKTI_BACKEND": "converse"}))
    assert isinstance(converse, RoutingBackend) and isinstance(converse.default, StrandsBackend)


# ----------------------------------------------------------------------------- precedence


def test_no_credentials_means_bedrock_defaults():
    cfg = load_config(env={})
    assert cfg.provider_mode == "bedrock" and cfg.panel == DEFAULT_BENCH[:3]


def test_gateway_gives_vendor_defaults_and_wins_over_keys():
    cfg = load_config(env={**KEYS, "TEKTI_GATEWAY_URL": "https://gw.example/", "TEKTI_GATEWAY_TOKEN": TOKEN})
    assert cfg.provider_mode == "vendors" and cfg.panel == VENDOR_BENCH
    assert [s.model_id for s in cfg.panel] == ["gpt-6.1-terra", "claude-haiku-5-5", "gemini-4-hydrogen"]
    assert cfg.segmenter.model_id == "claude-haiku-5-5" and cfg.merge_model.model_id == "gpt-6.1-terra"
    assert cfg.question_writer.model_id == "gpt-6.1-terra"
    assert cfg.reasoning["segment"] == "off" and cfg.reasoning["google:panel"] == "low"
    endpoints = vendor_endpoints(cfg)
    assert all(e.via_gateway and e.api_key.get_secret_value() == TOKEN for e in endpoints.values())
    assert endpoints["openai"].base_url == "https://gw.example/openai/v1"


def test_vendor_keys_filter_the_defaults():
    cfg = load_config(env={"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk"})
    assert [s.model_id for s in cfg.bench] == ["gpt-6.1-terra", "claude-haiku-5-5"]
    assert cfg.segmenter.provider == "anthropic" and cfg.question_writer.provider == "openai"
    assert cfg.max_parallel == 6


def test_google_only_is_the_free_tier_profile():
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        cfg = load_config(env={key: "g"})
        assert cfg.bench == [GOOGLE_ONLY] and cfg.panel == [GOOGLE_ONLY]
        assert cfg.segmenter == cfg.question_writer == cfg.merge_model == GOOGLE_ONLY
        assert cfg.max_parallel == 2
    assert load_config(env={"GEMINI_API_KEY": "g", "TEKTI_MAX_PARALLEL": "5"}).max_parallel == 5


def test_explicit_settings_win():
    env = {
        **KEYS,
        "TEKTI_PANEL": "openai:gpt-6.2-sol,google:gemini-4-carbon",
        "TEKTI_MERGE_MODEL": "google:gemini-4-hydrogen",
    }
    cfg = load_config(env=env)
    assert [s.model_id for s in cfg.panel] == ["gpt-6.2-sol", "gemini-4-carbon"]
    # vendor mode: an explicit panel does not move the roles (vendors-1 segmented on the panel's first model)
    assert cfg.segmenter.model_id == "claude-haiku-5-5" and cfg.question_writer.model_id == "gpt-6.1-terra"
    assert cfg.merge_model.model_id == "gemini-4-hydrogen"
    # explicit reasoning levels merge over the defaults
    tuned = load_config(env={**KEYS, "TEKTI_REASONING": "panel=high, google:panel=medium"})
    assert tuned.reasoning["panel"] == "high" and tuned.reasoning["google:panel"] == "medium"
    assert tuned.reasoning["segment"] == "off"
    with pytest.raises(ValueError, match="TEKTI_REASONING"):
        load_config(env={**KEYS, "TEKTI_REASONING": "panel=extreme"})
    # TEKTI_PROVIDER=bedrock keeps today's behaviour even with vendor keys present
    legacy = load_config(env={**KEYS, "TEKTI_PROVIDER": "bedrock"})
    assert legacy.provider_mode == "bedrock" and legacy.panel == DEFAULT_BENCH[:3]


def test_configuration_errors(tmp_path):
    with pytest.raises(ValueError, match="tekti login"):  # no owner token and no stored session
        load_config(env={"TEKTI_GATEWAY_URL": "https://gw.example", "XDG_CONFIG_HOME": str(tmp_path)})
    with pytest.raises(ValueError, match="TEKTI_GATEWAY_URL"):
        load_config(env={"TEKTI_GATEWAY_TOKEN": TOKEN})
    with pytest.raises(ValueError, match="vendors"):
        load_config(env={"TEKTI_PROVIDER": "vendors"})
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        load_config(env={"OPENAI_API_KEY": "sk", "TEKTI_PANEL": "openai:gpt-6.1-terra,anthropic:claude-haiku-5-5"})
    with pytest.raises(ValueError):
        load_config(env={"TEKTI_PROVIDER": "nope"})


def test_challenger_order_with_vendor_defaults():
    cfg = load_config(Mode.BOARD, env=KEYS)
    assert challenger_for(cfg, {"openai"}).family == "anthropic"
    assert challenger_for(cfg, {"google"}).family == "openai"
    assert challenger_for(cfg, {"anthropic"}).family == "openai"
    assert challenger_for(cfg, {"anthropic", "openai"}).family == "google"
    fast = load_config(Mode.FAST, env=KEYS)
    assert fast.panel == [VENDOR_BENCH[0]] and challenger_for(fast, {"openai"}).family == "anthropic"


# ----------------------------------------------------------------------------- secrets


def test_secrets_never_in_repr_or_dump():
    env = {
        **KEYS,
        "TEKTI_GATEWAY_URL": "https://gw.example",
        "TEKTI_GATEWAY_TOKEN": TOKEN,
        "TEKTI_API_KEY": "compat-key-123",
    }
    cfg = load_config(env=env)
    backend = make_backend(cfg)
    texts = [
        repr(cfg),
        str(cfg),
        cfg.model_dump_json(),
        json.dumps(cfg.model_dump(mode="json")),
        repr(backend),
        repr(vendor_endpoints(cfg)),
    ]
    for secret in [TOKEN, "compat-key-123", *KEYS.values()]:
        for text in texts:
            assert secret not in text
    assert cfg.gateway_token.get_secret_value() == TOKEN
    assert cfg.vendor_keys["google"].get_secret_value() == "g-not-real"


# ----------------------------------------------------------------------------- cost


def test_estimate_cost():
    calls = [
        ModelCall(stage="panel", model="gemini-4-neon", ok=True, input_tokens=1_000_000, output_tokens=100_000),
        ModelCall(stage="panel", model="gpt-6.1-terra", ok=True, input_tokens=2_000_000, output_tokens=0),
        ModelCall(stage="panel", model="claude-haiku-5-5", ok=True, input_tokens=0, output_tokens=1_000_000),
        ModelCall(stage="panel", model="models/gemini-4-hydrogen", ok=True, input_tokens=1_000_000),
        ModelCall(stage="panel", model="mystery-model", ok=True, input_tokens=5_000_000, output_tokens=5_000_000),
        ModelCall(stage="panel", model="gpt-6.1-terra", ok=False),
    ]
    assert estimate_cost(calls) == pytest.approx(3.0 + 1.5 + 2.0 + 0.5 + 1.5)
    assert unpriced_models(calls) == ["mystery-model"]
    assert price_of("gpt-6.1-terra-2026-10-01") == (1.00, 5.00)  # dated snapshot
    assert estimate_cost(calls[:1], prices={"gemini-4-neon": (1.0, 1.0)}) == pytest.approx(1.1)
    assert all(price_of(s.model_id) for s in [*VENDOR_BENCH, GOOGLE_ONLY])


# ----------------------------------------------------------------------------- thinking levels per stage


def test_reasoning_params_per_provider():
    from tekti.llm_backends.providers import output_cap, reasoning_params

    assert reasoning_params("openai", "gpt-6.1-terra", "medium") == {"reasoning": {"effort": "medium"}}
    assert reasoning_params("openai", "gpt-6.1-terra", "off") == {"reasoning": {"effort": "none"}}
    assert reasoning_params("openai", "gpt-6.2-sol", "off") == {"reasoning": {"effort": "low"}}  # 6.2 rejects none
    assert reasoning_params("anthropic", "claude-haiku-5-5", "low") == {"output_config": {"effort": "low"}}
    assert reasoning_params("anthropic", "claude-haiku-5-5", "off") == {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }
    assert reasoning_params("anthropic", "claude-sonnet-5-5", "off") == {"output_config": {"effort": "low"}}
    assert reasoning_params("google", "gemini-4-hydrogen", "low") == {"thinking_config": {"thinking_level": "LOW"}}
    # models without a thinking control, and Bedrock, get nothing
    for provider, model in [("google", "gemini-3.5-flash-lite"), ("anthropic", "claude-haiku-4-5"), ("bedrock", "x")]:
        assert reasoning_params(provider, model, "medium") == {}
    assert output_cap("google", "gemini-4-hydrogen", 12000, "medium") == 24000
    assert output_cap("google", "gemini-3.5-flash-lite", 12000, "medium") == 12000


def test_backend_picks_the_level_by_stage_and_provider():
    from tekti.llm_backends.strands_backend import stage_of

    backend = StrandsBackend(
        vendors={v: VendorEndpoint(api_key=SecretStr("k")) for v in ("openai", "anthropic", "google")},
        reasoning={"panel": "medium", "google:panel": "low", "segment": "off", "default": "low"},
    )
    gemini_panel = backend._model(GOOGLE, 12000, "panel").get_config()["params"]
    assert gemini_panel == {"max_output_tokens": 16000, "thinking_config": {"thinking_level": "LOW"}}
    haiku_panel = backend._model(ANTHROPIC, 12000, "panel").get_config()
    assert haiku_panel["max_tokens"] == 24000 and haiku_panel["params"] == {"output_config": {"effort": "medium"}}
    haiku_segment = backend._model(ANTHROPIC, 12000, "segment").get_config()
    assert haiku_segment["max_tokens"] == 12000 and haiku_segment["params"]["thinking"] == {"type": "disabled"}
    assert backend._model(OPENAI, 12000, "challenge").get_config()["params"]["reasoning"] == {"effort": "low"}

    class ReviewerOutput(BaseModel):
        pass

    assert stage_of(ReviewerOutput) == "panel" and stage_of(Answer) == "default"


# ----------------------------------------------------------------------------- start-up check of Google models


class _SlowGoogle:
    """Backend stub: `slow` model ids never answer in time, others answer at once."""

    def __init__(self, slow):
        self.slow, self.asked = set(slow), []

    async def complete(self, model, *, system, user, schema, max_tokens):
        import asyncio

        self.asked.append(model.model_id)
        if model.model_id in self.slow:
            await asyncio.sleep(10)
        return schema(capital="Paris"), 1, 1


async def test_overloaded_google_model_is_replaced_for_the_whole_run():
    from tekti.config import choose_responsive_models

    cfg = load_config(Mode.BOARD, env={**KEYS, "TEKTI_QUESTION_WRITER": "google:gemini-4-hydrogen"})
    stub = _SlowGoogle(slow={"gemini-4-hydrogen"})
    new, notes = await choose_responsive_models(cfg, timeout=0.2, backend=stub)
    assert stub.asked == ["gemini-4-hydrogen", "gemini-4-hydrogen", "gemini-3.5-flash-lite"]  # two tries, then the next
    assert [s.model_id for s in new.panel] == ["gpt-6.1-terra", "claude-haiku-5-5", "gemini-3.5-flash-lite"]
    assert new.question_writer.model_id == "gemini-3.5-flash-lite" and new.segmenter == cfg.segmenter
    assert len(notes) == 1 and "gemini-4-hydrogen" in notes[0] and "gemini-3.5-flash-lite" in notes[0]
    # answering in time, or the check switched off, changes nothing
    same, notes = await choose_responsive_models(cfg, timeout=0.2, backend=_SlowGoogle(slow=set()))
    assert same == cfg and notes == []
    assert (await choose_responsive_models(cfg, timeout=0, backend=stub))[0] == cfg
    # nothing to check without a Google model from the chain
    no_google = load_config(Mode.BOARD, env={"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk"})
    stub = _SlowGoogle(slow=set())
    assert (await choose_responsive_models(no_google, timeout=0.2, backend=stub))[0] == no_google
    assert stub.asked == []


def test_estimate_cost_prices_every_default_model():
    from tekti.config import GOOGLE_FALLBACKS, VENDOR_ROLES

    models = {s.model_id for s in VENDOR_BENCH} | {s.model_id for v in VENDOR_ROLES.values() for s in v}
    assert all(price_of(m) for m in models | set(GOOGLE_FALLBACKS))
