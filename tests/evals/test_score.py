"""Offline unit tests for tekti.evals."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Evidence,
    Finding,
    FindingStatus,
    Lens,
    Mode,
    ModelCall,
    Passport,
    ReviewerId,
    Run,
    Severity,
    Verdict,
)
from tekti.evals import (
    COMPATIBLE_CRITERION_PAIRS,
    GOLD_CAVEAT,
    GoldItem,
    GoldSet,
    criteria_compatible,
    load_gold,
    main,
    quotes_overlap,
    render_json,
    render_markdown,
    score_run,
    score_runs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_run(
    *,
    mode: Mode = Mode.BOARD,
    models: list[str] | None = None,
    findings: list[Finding] | None = None,
    rejected: list[Finding] | None = None,
    calls: list[ModelCall] | None = None,
) -> Run:
    doc = Document(
        name="requirements.md",
        source_path="local/requirements.md",
        format="markdown",
        sha256="deadbeef",
        raw_text="Placeholder raw text",
        blocks=[
            Block(
                id="B-0001",
                kind=BlockKind.PARAGRAPH,
                text="Placeholder raw text",
                section_path=[],
                line_start=1,
                line_end=1,
            )
        ],
    )
    passport = Passport(
        document_name=doc.name,
        document_sha256=doc.sha256,
        started_at=datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC),
        agent_version="0.1.0-test",
        mode=mode,
        models=["claude-haiku-4-5", "gpt-oss-120b"] if models is None else models,
    )
    return Run(
        passport=passport,
        document=doc,
        requirements=[],
        facts=[],
        findings=findings or [],
        rejected=rejected or [],
        questions=[],
        verdict=Verdict(
            ready_for_development=False,
            ready_for_testing=False,
            fix_first=[],
            rationale="Тестовый вердикт.",
        ),
        calls=calls or [],
    )


def _make_finding(
    fid: str,
    quotes: list[str],
    *,
    criterion: Criterion = Criterion.SET_CONSISTENT,
    severity: Severity = Severity.CRITICAL,
    status: FindingStatus = FindingStatus.CONFIRMED,
    title: str = "Дефект в требованиях",
) -> Finding:
    return Finding(
        id=fid,
        requirement_ids=["R-001"],
        criterion=criterion,
        severity=severity,
        title=title,
        explanation="Пояснение к дефекту.",
        evidence=[Evidence(quote=q, block_id="B-0001", requirement_id="R-001") for q in quotes],
        raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="claude-haiku-4-5")],
        families=1,
        status=status,
    )


def _sample_gold() -> GoldSet:
    return GoldSet(
        document="sample.md",
        items=[
            GoldItem(
                id="G-01",
                part="A",
                anchor="Месяц определяется по дате совершения покупки в Алматы",
                related="Февральский счётчик вырос на 4 000 ₸",
                criterion="set_consistent",
                critical=True,
                note="Month mismatch.",
            ),
            GoldItem(
                id="G-02",
                part="A",
                anchor="Возврат уменьшает расходы месяца исходной покупки",
                criterion="complete",
                critical=True,
                note="Refund behaviour missing.",
            ),
            GoldItem(
                id="G-03",
                part="B",
                anchor="Dev: `/alerts/saveLimit` . Prod: —",
                criterion="complete",
                critical=False,
                note="Prod missing.",
            ),
        ],
    )


# --------------------------------------------------------------------------- quotes_overlap


def test_quotes_overlap_containment_and_line_wraps() -> None:
    a = "Месяц\nопределяется   по   дате\tсовершения покупки"
    b = "По правилу Ф-2 месяц определяется по дате совершения покупки в Алматы."
    # b is not case-insensitively equal; wait: exact substring after normalize_ws:
    assert quotes_overlap(
        "определяется\nпо   дате\tсовершения покупки",
        "Месяц определяется по дате совершения покупки в Алматы",
    )
    assert quotes_overlap(b, "определяется по дате совершения покупки")
    # Different casing does not match containment here:
    assert not quotes_overlap(a, "другой текст без совпадения")


def test_quotes_overlap_25_char_threshold() -> None:
    shared_24 = "abcdefghijklmnopqrstuvwx"  # 24 chars
    assert len(shared_24) == 24
    assert not quotes_overlap(f"PREFIX1#{shared_24}!END1", f"OTHER2@{shared_24}$TAIL2")

    shared_25 = "abcdefghijklmnopqrstuvwxy"  # 25 chars
    assert len(shared_25) == 25
    assert quotes_overlap(f"PREFIX1#{shared_25}!END1", f"OTHER2@{shared_25}$TAIL2")


def test_quotes_overlap_empty_returns_false() -> None:
    assert not quotes_overlap("", "non empty text")
    assert not quotes_overlap("   \n\t ", "non empty text")
    assert not quotes_overlap("non empty text", "")
    assert not quotes_overlap("", "")


# --------------------------------------------------------------------------- criteria_compatible


def test_criteria_compatible_exact_pairs_and_incompatibles() -> None:
    assert len(COMPATIBLE_CRITERION_PAIRS) == 6
    for criterion in Criterion:
        assert criteria_compatible(criterion.value, criterion.value) is True

    expected_pairs = [
        ("correct", "set_consistent"),
        ("complete", "set_complete"),
        ("unambiguous", "verifiable"),
        ("verifiable", "set_validatable"),
        ("conforming", "set_complete"),
        ("singular", "modifiable"),
    ]
    for a, b in expected_pairs:
        assert criteria_compatible(a, b) is True
        assert criteria_compatible(b, a) is True

    for a, b in [
        ("singular", "unambiguous"),
        ("complete", "verifiable"),
        ("correct", "complete"),
        ("necessary", "set_consistent"),
    ]:
        assert criteria_compatible(a, b) is False
        assert criteria_compatible(b, a) is False


# --------------------------------------------------------------------------- matching & metrics


def test_many_to_many_criterion_independence_and_rejected_ignored() -> None:
    gold = _sample_gold()
    f1 = _make_finding(
        "F-001",
        [
            "Февральский счётчик вырос на 4 000 ₸",  # matches G-01 related
            "Возврат уменьшает расходы месяца исходной покупки",  # matches G-02 anchor
        ],
        criterion=Criterion.UNAMBIGUOUS,  # differs & incompatible with G-01 (set_consistent) and G-02 (complete)
    )
    f2 = _make_finding(
        "F-002",
        ["Месяц определяется по дате совершения покупки в Алматы"],  # also matches G-01 anchor
        criterion=Criterion.SET_CONSISTENT,
    )
    f_unmatched = _make_finding(
        "F-003",
        ["Совершенно отдельная цитата про таймаут"],
        criterion=Criterion.VERIFIABLE,
        severity=Severity.MAJOR,
        title="Не задан таймаут",
    )
    f_rejected_in_findings = _make_finding(
        "F-004",
        ["Dev: `/alerts/saveLimit` . Prod: —"],
        status=FindingStatus.REJECTED,
    )
    f_rejected_list = _make_finding(
        "F-005",
        ["Dev: `/alerts/saveLimit` . Prod: —"],
        status=FindingStatus.REJECTED,
    )

    run = _make_run(
        findings=[f1, f2, f_unmatched, f_rejected_in_findings],
        rejected=[f_rejected_list],
    )
    score = score_run(run, gold, label="single")

    # Loose: G-01 and G-02 matched; G-03 missed because F-004 and F-005 are rejected
    assert score.gold_total == 3
    assert score.gold_matched == 2
    assert score.recall_all == 2 / 3
    assert score.critical_total == 2
    assert score.critical_matched == 2
    assert score.recall_critical == 1.0

    # Strict: G-01 matched by F-002 (set_consistent); G-02 only overlapped by F-001 (unambiguous != complete)
    assert score.strict_gold_matched == 1
    assert score.strict_recall_all == 1 / 3
    assert score.strict_critical_matched == 1
    assert score.strict_recall_critical == 0.5

    # Only confirmed findings counted: F-001, F-002, F-003 -> 2 matched (loose), 1 unmatched
    assert score.confirmed_total == 3
    assert score.confirmed_matched == 2
    assert score.confirmed_unmatched == 1
    assert score.unmatched_share == 1 / 3

    hit_g1, hit_g2, hit_g3 = score.hits
    assert hit_g1.matched is True
    assert hit_g1.strict_matched is True
    assert hit_g1.strict_finding_ids == ["F-002"]
    assert hit_g1.matched_on == "both"
    assert [(m.finding_id, m.criterion_matches, m.criterion_compatible, m.matched_on) for m in hit_g1.matches] == [
        ("F-001", False, False, "related"),
        ("F-002", True, True, "anchor"),
    ]
    assert hit_g2.matched is True
    assert hit_g2.strict_matched is False
    assert hit_g2.strict_finding_ids == []
    assert hit_g2.matched_on == "anchor"
    assert [(m.finding_id, m.criterion_matches, m.criterion_compatible) for m in hit_g2.matches] == [
        ("F-001", False, False)
    ]
    assert hit_g3.matched is False
    assert hit_g3.strict_matched is False
    assert hit_g3.strict_finding_ids == []
    assert hit_g3.matched_on is None

    # Neutral unmatched reporting uses loose rule (F-001 is NOT unmatched even though it failed strict on G-02)
    assert [u.id for u in score.unmatched_findings] == ["F-003"]
    assert score.unmatched_findings[0].primary_quote == "Совершенно отдельная цитата про таймаут"


def test_loose_only_vs_compatible_pair_strict_hits() -> None:
    gold = GoldSet(
        document="g11.md",
        items=[
            GoldItem(
                id="G-11",
                part="A",
                anchor="Пользователь может настраивать лимит и получать уведомления по дате покупки",
                criterion="unambiguous",
                critical=False,
                note="Ambiguous wording.",
            ),
            GoldItem(
                id="G-12",
                part="A",
                anchor="Сумма лимита должна совпадать с валютой счёта клиента",
                criterion="set_consistent",
                critical=True,
                note="Consistency between sections.",
            ),
        ],
    )
    # Overlaps G-11 with incompatible criterion (singular vs unambiguous) -> loose hit, strict miss, still not unmatched
    f_loose_only = _make_finding(
        "F-010",
        ["Пользователь может настраивать лимит и получать уведомления по дате покупки"],
        criterion=Criterion.SINGULAR,
    )
    # Overlaps G-12 with compatible criterion (correct vs set_consistent) -> strict hit!
    f_compat = _make_finding(
        "F-011",
        ["Сумма лимита должна совпадать с валютой счёта клиента"],
        criterion=Criterion.CORRECT,
    )
    run = _make_run(findings=[f_loose_only, f_compat])
    score = score_run(run, gold, label="compat-check")

    assert score.gold_matched == 2
    assert score.strict_gold_matched == 1
    assert score.critical_matched == 1
    assert score.strict_critical_matched == 1
    assert score.confirmed_matched == 2
    assert score.confirmed_unmatched == 0
    assert score.unmatched_findings == []

    hit_11, hit_12 = score.hits
    assert hit_11.matched is True
    assert hit_11.strict_matched is False
    assert hit_11.strict_finding_ids == []
    assert hit_11.matches[0].criterion_matches is False
    assert hit_11.matches[0].criterion_compatible is False

    assert hit_12.matched is True
    assert hit_12.strict_matched is True
    assert hit_12.strict_finding_ids == ["F-011"]
    assert hit_12.matches[0].criterion_matches is False
    assert hit_12.matches[0].criterion_compatible is True


def test_calls_aggregation_cost_and_unpriced_models() -> None:
    gold = _sample_gold()
    calls = [
        ModelCall(
            stage="segment",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            ok=True,
            input_tokens=1_000_000,
            output_tokens=200_000,
            seconds=2.5,
        ),
        ModelCall(
            stage="panel",
            lens=Lens.FUNCTIONAL,
            model="openai.gpt-oss-120b-1:0",
            ok=True,
            input_tokens=500_000,
            output_tokens=100_000,
            seconds=4.0,
        ),
        ModelCall(
            stage="panel",
            lens=Lens.TESTER,
            model="gemini-2.5-flash",
            ok=False,
            error="timeout",
            input_tokens=0,
            output_tokens=0,
            seconds=1.5,
        ),
    ]
    run = _make_run(calls=calls)
    score = score_run(run, gold)

    assert score.calls_total == 3
    assert score.calls_ok == 2
    assert score.calls_failed == 1
    assert score.input_tokens == 1_500_000
    assert score.output_tokens == 300_000
    assert score.total_call_seconds == 8.0
    assert score.max_call_seconds == 4.0
    # segment (2.5) + panel max (4.0) = 6.5
    assert score.estimated_wall_seconds == 6.5
    assert score.unpriced_models == []
    # haiku: 1.0*1.0 + 0.2*5.0 = 2.00; gpt-oss: 0.5*0.15 + 0.1*0.60 = 0.135 -> 2.135
    assert score.estimated_cost_usd is not None
    assert abs(score.estimated_cost_usd - 2.135) < 1e-9

    # Zero calls -> $0.0000
    zero_score = score_run(_make_run(calls=[]), gold)
    assert zero_score.estimated_cost_usd == 0.0
    assert zero_score.unpriced_models == []

    # Unpriced model -> estimated_cost_usd is None and unpriced_models lists it
    unknown_score = score_run(
        _make_run(
            calls=[
                ModelCall(stage="segment", model="custom-local-llm-7b", ok=True, input_tokens=1000, output_tokens=500, seconds=1.0)
            ]
        ),
        gold,
    )
    assert unknown_score.estimated_cost_usd is None
    assert unknown_score.unpriced_models == ["custom-local-llm-7b"]


def test_markdown_and_json_neutral_and_caveat() -> None:
    gold = _sample_gold()
    run = _make_run(
        findings=[
            _make_finding("F-001", ["Месяц определяется по дате совершения покупки в Алматы"], criterion=Criterion.CORRECT),
            _make_finding("F-002", ["Возврат уменьшает расходы месяца исходной покупки"], criterion=Criterion.SINGULAR),
            _make_finding("F-003", ["Неопознанный | фрагмент"]),
        ]
    )
    report = score_runs([("run1 (board)", run)], gold)

    md = render_markdown(report)
    assert GOLD_CAVEAT in md
    assert "Loose recall (all) | 2/3 (66.7%)" in md
    assert "Strict recall (all) | 1/3 (33.3%)" in md
    assert "Loose recall (critical) | 2/2 (100.0%)" in md
    assert "Strict recall (critical) | 1/2 (50.0%)" in md
    assert "Unmatched confirmed (loose) | 1/3 (33.3%)" in md
    assert "| G-01 | A | yes | `set_consistent` | HIT | HIT | F-001 |" in md
    assert "| G-02 | A | yes | `complete` | HIT | MISS | — |" in md
    assert "Unmatched confirmed findings (for human review)" in md
    assert "false positive" not in md.lower()
    assert "G-01" in md
    assert "HIT" in md
    assert "MISS" in md
    assert "Неопознанный \\| фрагмент" in md

    parsed = json.loads(render_json(report))
    assert parsed["caveat"] == GOLD_CAVEAT
    run_json = parsed["runs"][0]
    assert run_json["label"] == "run1 (board)"
    assert run_json["gold_matched"] == 2
    assert run_json["strict_gold_matched"] == 1
    assert run_json["critical_matched"] == 2
    assert run_json["strict_critical_matched"] == 1
    assert run_json["hits"][0]["strict_matched"] is True
    assert run_json["hits"][0]["strict_finding_ids"] == ["F-001"]
    assert run_json["hits"][1]["strict_matched"] is False
    assert run_json["hits"][1]["strict_finding_ids"] == []


def test_multi_run_markdown_shows_loose_and_strict_matrix() -> None:
    gold = _sample_gold()
    run_a = _make_run(
        findings=[
            _make_finding("F-001", ["Месяц определяется по дате совершения покупки в Алматы"], criterion=Criterion.CORRECT),
            _make_finding("F-002", ["Февральский счётчик вырос на 4 000 ₸"], criterion=Criterion.SINGULAR),
        ]
    )
    run_b = _make_run(
        findings=[
            _make_finding("F-010", ["Возврат уменьшает расходы месяца исходной покупки"], criterion=Criterion.SINGULAR),
        ]
    )
    report = score_runs([("run-a", run_a), ("run-b", run_b)], gold)
    md = render_markdown(report)

    assert "Loose recall (all) | 1/3 (33.3%) | 1/3 (33.3%)" in md
    assert "Strict recall (all) | 1/3 (33.3%) | 0/3 (0.0%)" in md
    assert "Loose recall (critical) | 1/2 (50.0%) | 1/2 (50.0%)" in md
    assert "Strict recall (critical) | 1/2 (50.0%) | 0/2 (0.0%)" in md
    assert "Unmatched share (loose) | 0/2 (0.0%) | 0/1 (0.0%)" in md
    assert "| G-01 | A | yes | `set_consistent` | strict: F-001; loose: +F-002 | — |" in md
    assert "| G-02 | A | yes | `complete` | — | (loose: F-010) |" in md


# --------------------------------------------------------------------------- CLI & real gold.yaml


def test_real_gold_yaml_loads_48_items_20_critical() -> None:
    gold = load_gold(REPO_ROOT / "evals" / "gold.yaml")
    assert len(gold.items) == 48
    assert sum(1 for item in gold.items if item.critical) == 20


def test_cli_single_and_multi_run(tmp_path: Path, capsys) -> None:
    gold_path = REPO_ROOT / "evals" / "gold.yaml"

    run_rules = _make_run(
        mode=Mode.FAST,
        models=[],
        findings=[
            _make_finding("F-001", ["заполняются тестировщиками"], criterion=Criterion.SET_COMPLETE),
        ],
        calls=[],
    )
    run_single = _make_run(
        mode=Mode.FAST,
        models=["claude-haiku-4-5"],
        findings=[
            _make_finding("F-001", ["Месяц определяется по дате совершения покупки в Алматы"]),
            _make_finding("F-002", ["Лишнее замечание"]),
        ],
        calls=[ModelCall(stage="panel", model="claude-haiku-4-5", ok=True, input_tokens=1000, output_tokens=500, seconds=1.2)],
    )
    run_panel = _make_run(
        mode=Mode.BOARD,
        models=["claude-haiku-4-5", "openai.gpt-oss-120b-1:0", "us.moonshotai.kimi-k3.1"],
        findings=[
            _make_finding("F-001", ["Месяц определяется по дате совершения покупки в Алматы"]),
            _make_finding("F-002", ["Возврат уменьшает расходы месяца исходной покупки"], criterion=Criterion.COMPLETE),
        ],
        calls=[
            ModelCall(stage="panel", model="claude-haiku-4-5", ok=True, input_tokens=1000, output_tokens=500, seconds=1.2),
            ModelCall(stage="panel", model="openai.gpt-oss-120b-1:0", ok=True, input_tokens=1000, output_tokens=500, seconds=2.4),
        ],
    )

    p1 = tmp_path / "rules.json"
    p2 = tmp_path / "single.json"
    p3 = tmp_path / "panel.json"
    p1.write_text(run_rules.model_dump_json(), encoding="utf-8")
    p2.write_text(run_single.model_dump_json(), encoding="utf-8")
    p3.write_text(run_panel.model_dump_json(), encoding="utf-8")

    # Single run JSON stdout
    rc = main([str(p2), "--gold", str(gold_path), "--format", "json"])
    assert rc == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["gold_total"] == 48
    assert out_json["critical_total"] == 20
    assert out_json["runs"][0]["label"] == "single (fast)"
    # "Месяц определяется по дате совершения покупки в Алматы" matches both G-01 and G-04
    assert out_json["runs"][0]["gold_matched"] == 2

    # Multi-run side-by-side markdown + --out + --json-out
    md_out = tmp_path / "compare.md"
    json_out = tmp_path / "compare.json"
    rc = main(
        [
            str(p1),
            str(p2),
            str(p3),
            "--gold",
            str(gold_path),
            "--label",
            "rules-only",
            "--label",
            "single-model",
            "--label",
            "panel+rules",
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ]
    )
    assert rc == 0
    stdout_md = capsys.readouterr().out
    assert md_out.read_text(encoding="utf-8") == stdout_md
    assert "## Summary comparison" in stdout_md
    assert "## Per-gold-item hit matrix" in stdout_md
    assert "| Metric | rules-only | single-model | panel+rules |" in stdout_md
    assert "G-26" in stdout_md
    assert "G-48" in stdout_md
    assert json.loads(json_out.read_text(encoding="utf-8"))["runs"][2]["label"] == "panel+rules"


def test_python_m_tekti_evals_entrypoint(tmp_path: Path) -> None:
    run_path = tmp_path / "run1.json"
    run_path.write_text(_make_run().model_dump_json(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "tekti.evals", str(run_path), "--gold", str(REPO_ROOT / "evals" / "gold.yaml")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "run1 (board)" in proc.stdout
    assert GOLD_CAVEAT in proc.stdout
