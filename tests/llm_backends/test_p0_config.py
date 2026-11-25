import pytest

from tekti.config import DEFAULT_BENCH, challenger_for, load_config, parse_panel
from tekti.contracts import Lens, Mode


def test_defaults_per_mode():
    for mode, n in [(Mode.FAST, 1), (Mode.STANDARD, 2), (Mode.BOARD, 3)]:
        cfg = load_config(mode, env={})
        assert len(cfg.panel) == n
        assert len({s.family for s in cfg.panel}) == n
        assert cfg.panel == DEFAULT_BENCH[:n]
    cfg = load_config("board", env={})
    assert cfg.lenses == list(Lens)
    assert cfg.max_output_tokens > 0 and cfg.max_parallel > 0
    assert cfg.backend == "strands" and cfg.region == "us-east-1"
    assert all(s.provider == "bedrock" for s in [*cfg.panel, cfg.segmenter, cfg.question_writer])


def test_panel_override_keeps_colons_in_model_ids():
    env = {
        "TEKTI_PANEL": "openai:openai.gpt-oss-120b-1:0, anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0,"
        "openai:other",
        "TEKTI_SEGMENTER": "us.amazon.nova-2-lite-v1:0",
        "TEKTI_LENSES": "tester,developer",
        "TEKTI_MAX_PARALLEL": "3",
    }
    cfg = load_config(Mode.BOARD, env=env)
    assert [(s.family, s.model_id) for s in cfg.panel] == [
        ("openai", "openai.gpt-oss-120b-1:0"),
        ("anthropic", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ]  # duplicate family dropped
    assert cfg.segmenter.family == "amazon" and cfg.segmenter.model_id == "us.amazon.nova-2-lite-v1:0"
    assert cfg.question_writer == cfg.panel[0]
    assert cfg.lenses == [Lens.TESTER, Lens.DEVELOPER]
    assert cfg.max_parallel == 3


def test_bare_ids_get_a_family():
    specs = parse_panel("openai.gpt-oss-120b-1:0,us.moonshotai.kimi-k3.1,deepseek.v3.2", "bedrock")
    assert [s.family for s in specs] == ["openai", "moonshot", "deepseek"]


def test_openai_compatible_requires_panel_and_url_and_hides_key():
    with pytest.raises(ValueError):
        load_config(env={"TEKTI_PROVIDER": "openai_compatible"})
    with pytest.raises(ValueError):
        load_config(env={"TEKTI_PROVIDER": "openai_compatible", "TEKTI_PANEL": "google:gemini-2.5-pro"})
    cfg = load_config(
        Mode.FAST,
        env={
            "TEKTI_PROVIDER": "openai_compatible",
            "TEKTI_PANEL": "google:gemini-2.5-pro",
            "TEKTI_BASE_URL": "http://localhost:11434/v1",
            "TEKTI_API_KEY": "sk-test-not-real",
        },
    )
    assert cfg.panel[0].provider == "openai_compatible"
    assert cfg.segmenter.model_id == "gemini-2.5-pro"
    assert "sk-test-not-real" not in repr(cfg)
    assert "sk-test-not-real" not in cfg.model_dump_json()
    assert cfg.api_key is not None and cfg.api_key.get_secret_value() == "sk-test-not-real"


def test_converse_only_with_bedrock():
    with pytest.raises(ValueError):
        load_config(
            env={
                "TEKTI_BACKEND": "converse",
                "TEKTI_PROVIDER": "openai_compatible",
                "TEKTI_PANEL": "google:g",
                "TEKTI_BASE_URL": "http://x",
            }
        )
    assert load_config(env={"TEKTI_BACKEND": "converse"}).backend == "converse"


def test_dotenv_is_loaded_and_environment_wins(tmp_path, monkeypatch):
    for var in ("TEKTI_PANEL", "TEKTI_MAX_PARALLEL", "TEKTI_PROVIDER", "TEKTI_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text("TEKTI_PANEL=zai:zai.glm-4.7\nTEKTI_MAX_PARALLEL=2\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEKTI_MAX_PARALLEL", "9")
    cfg = load_config(Mode.FAST)
    assert cfg.panel[0].model_id == "zai.glm-4.7"
    assert cfg.max_parallel == 9


def test_challenger_choice():
    cfg = load_config(Mode.BOARD, env={})
    fams = [s.family for s in cfg.panel]
    assert challenger_for(cfg, {fams[0]}).family == fams[1]
    assert challenger_for(cfg, {fams[0], fams[1]}).family == fams[2]
    # fast mode: panel has one family, the challenger comes from the bench
    fast = load_config(Mode.FAST, env={})
    assert challenger_for(fast, {fams[0]}).family != fams[0]
    # every family raised it: fall back to the strongest panel model
    everyone = {s.family for s in cfg.bench}
    assert challenger_for(cfg, everyone) == cfg.panel[0]


def test_install_backend(monkeypatch):
    from tekti import config, llm
    from tekti.llm_backends import ConverseBackend, StrandsBackend

    installed = []
    monkeypatch.setattr(llm, "use_backend", installed.append)
    config.install_backend(load_config(env={}))
    assert isinstance(installed[-1], StrandsBackend)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    config.install_backend(load_config(env={"TEKTI_BACKEND": "converse"}))
    assert isinstance(installed[-1], ConverseBackend)
