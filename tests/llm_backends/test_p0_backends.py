from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from tekti.llm import LLMError, ModelSpec
from tekti.llm_backends import ConverseBackend, StrandsBackend
from tekti.llm_backends._common import extract_json, is_temperature_rejection, validate_with_repair

SPEC = ModelSpec(family="anthropic", model_id="m-1")


class Out(BaseModel):
    items: list[int]


def client_error(code: str, message: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


class StubClient:
    """Plays back a list of Converse responses or exceptions and records the requests."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            content = [{"text": item}]
        else:
            content = [{"toolUse": {"toolUseId": "t", "name": "submit_result", "input": item}}]
        return {"output": {"message": {"content": content}}, "usage": {"inputTokens": 10, "outputTokens": 5}}


async def complete(backend):
    return await backend.complete(SPEC, system="s", user="u", schema=Out, max_tokens=123)


# ----------------------------------------------------------------------------- shared helpers


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"items": [1]}\n```') == {"items": [1]}
    assert extract_json('Here you go: {"items": [2]}') == {"items": [2]}


def test_temperature_rejection_detection():
    assert is_temperature_rejection(
        client_error("ValidationException", "This model doesn't support the temperature field. Remove temperature")
    )
    assert not is_temperature_rejection(client_error("ValidationException", "This model doesn't support tool use."))


async def test_validate_with_repair_passes_errors_to_second_attempt():
    seen = []

    async def attempt(errors):
        seen.append(errors)
        return ({"items": "bad"} if errors is None else {"items": [1, 2]}), 3, 4

    parsed, t_in, t_out = await validate_with_repair(Out, attempt)
    assert parsed.items == [1, 2] and (t_in, t_out) == (6, 8)
    assert seen[0] is None and "items" in seen[1]


async def test_validate_with_repair_fails_after_one_repair():
    calls = 0

    async def attempt(errors):
        nonlocal calls
        calls += 1
        return "not json at all", 1, 1

    with pytest.raises(LLMError):
        await validate_with_repair(Out, attempt)
    assert calls == 2


# ----------------------------------------------------------------------------- Converse


async def test_converse_forced_tool_and_explicit_max_tokens():
    client = StubClient([{"items": [1]}])
    parsed, t_in, t_out = await complete(ConverseBackend(client=client))
    assert parsed.items == [1] and (t_in, t_out) == (10, 5)
    req = client.requests[0]
    assert req["inferenceConfig"] == {"maxTokens": 123, "temperature": 0.0}
    assert req["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_result"}}
    assert req["system"] == [{"text": "s"}]
    assert req["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["title"] == "Out"


async def test_converse_repairs_once_then_succeeds():
    client = StubClient([{"items": "x"}, {"items": [7]}])
    parsed, t_in, _ = await complete(ConverseBackend(client=client))
    assert parsed.items == [7] and t_in == 20
    assert "did not match" in client.requests[1]["messages"][0]["content"][-1]["text"]


async def test_converse_fails_after_repair():
    client = StubClient([{"items": "x"}, {"wrong": 1}])
    with pytest.raises(LLMError):
        await complete(ConverseBackend(client=client))
    assert len(client.requests) == 2


async def test_converse_drops_temperature_when_refused_and_retries_throttling():
    client = StubClient(
        [
            client_error("ValidationException", "This model doesn't support the temperature field."),
            client_error("ThrottlingException", "Too many requests"),
            {"items": [1]},
        ]
    )
    backend = ConverseBackend(client=client)
    backend.backoff_base = 0
    parsed, _, _ = await complete(backend)
    assert parsed.items == [1]
    assert "temperature" not in client.requests[-1]["inferenceConfig"]


async def test_converse_does_not_retry_access_denied():
    client = StubClient([client_error("AccessDeniedException", "not available for this account")])
    with pytest.raises(ClientError):
        await complete(ConverseBackend(client=client))
    assert len(client.requests) == 1


# ----------------------------------------------------------------------------- Strands (agent stubbed)


class FakeAgent:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    async def invoke_async(self, prompt, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(
            structured_output=self.outcome,
            stop_reason="end_turn" if self.outcome else "limit_structured_output_attempts",
            metrics=SimpleNamespace(accumulated_usage={"inputTokens": 11, "outputTokens": 7}),
        )


def strands_with(outcomes):
    backend = StrandsBackend(region="us-east-1")
    agents = []
    queue = list(outcomes)

    def fake_agent(spec, system, max_tokens, stage="default"):
        agents.append((spec.model_id, (spec.provider, spec.model_id) in backend._no_temperature, max_tokens))
        agent = FakeAgent(queue.pop(0))
        agents[-1] = (*agents[-1], agent)
        return agent

    backend._agent = fake_agent  # type: ignore[method-assign]
    return backend, agents


async def test_strands_returns_parsed_and_tokens_with_one_repair_limit():
    backend, agents = strands_with([Out(items=[3])])
    parsed, t_in, t_out = await complete(backend)
    assert parsed.items == [3] and (t_in, t_out) == (11, 7)
    agent = agents[0][-1]
    assert agent.kwargs["structured_output_model"] is Out
    assert agent.kwargs["limits"] == {"structured_output_attempts": 2}


async def test_strands_raises_when_no_structured_output():
    backend, _ = strands_with([None])
    with pytest.raises(LLMError, match="limit_structured_output_attempts"):
        await complete(backend)


async def test_strands_structured_output_exception_becomes_llm_error():
    from strands.types.exceptions import StructuredOutputException

    backend, _ = strands_with([StructuredOutputException("bad")])
    with pytest.raises(LLMError):
        await complete(backend)


async def test_strands_retries_without_temperature():
    refusal = client_error("ValidationException", "This model doesn't support the temperature field.")
    backend, agents = strands_with([refusal, Out(items=[1])])
    parsed, _, _ = await complete(backend)
    assert parsed.items == [1]
    assert [a[1] for a in agents] == [False, True]


def test_strands_builds_real_bedrock_model_with_max_tokens_and_temperature():
    backend = StrandsBackend(region="us-east-1")
    model = backend._model(SPEC, 321)
    cfg = model.get_config()
    assert cfg["max_tokens"] == 321 and cfg["temperature"] == 0.0 and cfg["streaming"] is False
    assert backend._model(SPEC, 321) is model  # cached per model/max_tokens


def test_strands_openai_compatible_needs_base_url():
    with pytest.raises(LLMError):
        StrandsBackend()._model(ModelSpec(family="google", model_id="g", provider="openai_compatible"), 100)
    model = StrandsBackend(base_url="http://localhost:11434/v1", api_key="k")._model(
        ModelSpec(family="google", model_id="g", provider="openai_compatible"), 100
    )
    assert model.get_config()["params"] == {"max_tokens": 100, "temperature": 0.0}
