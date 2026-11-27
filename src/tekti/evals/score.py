"""Score saved Tekti Run JSON files against the gold defect set."""

from __future__ import annotations

import argparse
import difflib
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from tekti.contracts import Finding, FindingStatus, ModelCall, Run, normalize_ws

OVERLAP_MIN_RUN = 25

COMPATIBLE_CRITERION_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"correct", "set_consistent"}),
        frozenset({"complete", "set_complete"}),
        frozenset({"unambiguous", "verifiable"}),
        frozenset({"verifiable", "set_validatable"}),
        frozenset({"conforming", "set_complete"}),
        frozenset({"singular", "modifiable"}),
    }
)


def criteria_compatible(finding_criterion: str, gold_criterion: str) -> bool:
    """Return True when finding and gold criteria match exactly or belong to a compatible pair."""
    return (
        finding_criterion == gold_criterion
        or frozenset({finding_criterion, gold_criterion}) in COMPATIBLE_CRITERION_PAIRS
    )

GOLD_CAVEAT = (
    "STATUS: unreviewed. Written by one AI model (Claude) on 25 November 2026 from a "
    "single reading. It may favour reviewers that reason the same way. Present any "
    "score computed from it with that caveat."
)

# Approximate reference USD rates per 1M tokens: (input_usd_per_1m, output_usd_per_1m).
# Used for relative cost comparison across runs; matched by substring on model.lower().
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-oss-120b": (0.15, 0.60),
    "kimi-k3": (0.60, 2.50),
    "deepseek.v3": (0.58, 1.68),
    "mistral-large-3": (2.00, 6.00),
    "qwen3-vl-235b": (0.80, 3.00),
    "qwen3": (0.30, 1.20),
    "nova-2-lite": (0.30, 2.50),
    "nova-pro": (0.80, 3.20),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}


class GoldItem(BaseModel):
    """One known defect in the gold set."""

    id: str
    part: str
    anchor: str
    related: str | None = None
    criterion: str
    critical: bool
    note: str


class GoldSet(BaseModel):
    """Parsed gold defect list from YAML."""

    document: str
    items: list[GoldItem]


class FindingMatch(BaseModel):
    """A confirmed finding matched to a gold item."""

    finding_id: str
    finding_criterion: str
    criterion_matches: bool
    criterion_compatible: bool
    matched_on: Literal["anchor", "related", "both"]


class GoldHit(BaseModel):
    """Match result for a single gold item within one run."""

    gold_id: str
    part: str
    critical: bool
    gold_criterion: str
    note: str
    matched: bool
    strict_matched: bool = False
    strict_finding_ids: list[str] = Field(default_factory=list)
    matched_on: Literal["anchor", "related", "both"] | None = None
    matches: list[FindingMatch] = Field(default_factory=list)


class UnmatchedFinding(BaseModel):
    """Confirmed finding that did not overlap any gold item, listed neutrally for review."""

    id: str
    severity: str
    criterion: str
    title: str
    primary_quote: str


class RunScore(BaseModel):
    """Evaluation summary, per-item hits, and call telemetry for one run."""

    label: str
    document_name: str
    mode: str
    models: list[str]
    models_count: int
    gold_total: int
    gold_matched: int
    recall_all: float
    strict_gold_matched: int
    strict_recall_all: float
    critical_total: int
    critical_matched: int
    recall_critical: float
    strict_critical_matched: int
    strict_recall_critical: float
    confirmed_total: int
    confirmed_matched: int
    confirmed_unmatched: int
    unmatched_share: float
    calls_total: int
    calls_ok: int
    calls_failed: int
    input_tokens: int
    output_tokens: int
    total_call_seconds: float
    max_call_seconds: float
    estimated_wall_seconds: float
    estimated_cost_usd: float | None
    unpriced_models: list[str] = Field(default_factory=list)
    hits: list[GoldHit]
    unmatched_findings: list[UnmatchedFinding]


class EvalReport(BaseModel):
    """Complete evaluation report covering one or more runs."""

    caveat: str = GOLD_CAVEAT
    gold_document: str
    gold_total: int
    critical_total: int
    runs: list[RunScore]


def load_gold(path: Path) -> GoldSet:
    """Load and validate a gold YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GoldSet.model_validate(data)


def quotes_overlap(a: str, b: str) -> bool:
    """Return True when two quotes overlap after whitespace normalisation.

    Two non-empty normalised quotes overlap if one contains the other or their longest
    contiguous common substring is at least 25 characters.
    """
    a_norm = normalize_ws(a)
    b_norm = normalize_ws(b)
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return True
    longest = difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False).find_longest_match(
        0, len(a_norm), 0, len(b_norm)
    )
    return longest.size >= OVERLAP_MIN_RUN


def lookup_model_pricing(model: str) -> tuple[float, float] | None:
    """Return (input_usd_per_1m, output_usd_per_1m) for a known model ID, or None."""
    lowered = model.lower()
    for pattern, prices in MODEL_PRICING_PER_1M.items():
        if pattern in lowered:
            return prices
    from tekti.config import price_of  # the run-time price table covers the current default models

    return price_of(model)


def _combine_targets(targets: Sequence[Literal["anchor", "related", "both"]]) -> Literal["anchor", "related", "both"] | None:
    if not targets:
        return None
    unique = set(targets)
    if "both" in unique or ("anchor" in unique and "related" in unique):
        return "both"
    return "anchor" if "anchor" in unique else "related"


def _match_finding_to_item(finding: Finding, item: GoldItem) -> FindingMatch | None:
    quotes = [ev.quote for ev in finding.evidence]
    hit_anchor = any(quotes_overlap(q, item.anchor) for q in quotes)
    has_related = bool(item.related and normalize_ws(item.related))
    hit_related = has_related and any(quotes_overlap(q, item.related or "") for q in quotes)
    if not hit_anchor and not hit_related:
        return None
    if hit_anchor and hit_related:
        matched_on: Literal["anchor", "related", "both"] = "both"
    elif hit_anchor:
        matched_on = "anchor"
    else:
        matched_on = "related"
    return FindingMatch(
        finding_id=finding.id,
        finding_criterion=finding.criterion.value,
        criterion_matches=(finding.criterion.value == item.criterion),
        criterion_compatible=criteria_compatible(finding.criterion.value, item.criterion),
        matched_on=matched_on,
    )


def _summarize_calls(
    calls: Sequence[ModelCall],
) -> tuple[int, int, int, int, int, float, float, float, float | None, list[str]]:
    calls_total = len(calls)
    calls_ok = sum(1 for c in calls if c.ok)
    calls_failed = calls_total - calls_ok
    input_tokens = sum(c.input_tokens for c in calls)
    output_tokens = sum(c.output_tokens for c in calls)
    total_call_seconds = sum(c.seconds for c in calls)
    max_call_seconds = max((c.seconds for c in calls), default=0.0)

    stage_max: dict[str, float] = defaultdict(float)
    stage_order: list[str] = []
    for c in calls:
        if c.stage not in stage_max:
            stage_order.append(c.stage)
        stage_max[c.stage] = max(stage_max[c.stage], c.seconds)
    estimated_wall_seconds = sum(stage_max[stage] for stage in stage_order)

    if calls_total == 0:
        return (0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, [])

    total_cost = 0.0
    unpriced_seen: list[str] = []
    unpriced_set: set[str] = set()
    for c in calls:
        pricing = lookup_model_pricing(c.model)
        if pricing is None:
            if c.model not in unpriced_set:
                unpriced_set.add(c.model)
                unpriced_seen.append(c.model)
        else:
            in_rate, out_rate = pricing
            total_cost += (c.input_tokens * in_rate + c.output_tokens * out_rate) / 1_000_000.0

    estimated_cost_usd: float | None = None if unpriced_seen else total_cost
    return (
        calls_total,
        calls_ok,
        calls_failed,
        input_tokens,
        output_tokens,
        total_call_seconds,
        max_call_seconds,
        estimated_wall_seconds,
        estimated_cost_usd,
        unpriced_seen,
    )


def score_run(run: Run, gold: GoldSet, *, label: str | None = None) -> RunScore:
    """Score a single Run against a GoldSet."""
    confirmed = [f for f in run.findings if f.status == FindingStatus.CONFIRMED]
    matched_finding_ids: set[str] = set()
    hits: list[GoldHit] = []

    for item in gold.items:
        item_matches: list[FindingMatch] = []
        for finding in confirmed:
            fm = _match_finding_to_item(finding, item)
            if fm is not None:
                item_matches.append(fm)
                matched_finding_ids.add(finding.id)
        strict_ids = [m.finding_id for m in item_matches if m.criterion_compatible]
        hits.append(
            GoldHit(
                gold_id=item.id,
                part=item.part,
                critical=item.critical,
                gold_criterion=item.criterion,
                note=item.note,
                matched=bool(item_matches),
                strict_matched=bool(strict_ids),
                strict_finding_ids=strict_ids,
                matched_on=_combine_targets([m.matched_on for m in item_matches]),
                matches=item_matches,
            )
        )

    unmatched_findings = [
        UnmatchedFinding(
            id=f.id,
            severity=f.severity.value,
            criterion=f.criterion.value,
            title=f.title,
            primary_quote=f.evidence[0].quote if f.evidence else "",
        )
        for f in confirmed
        if f.id not in matched_finding_ids
    ]

    gold_total = len(gold.items)
    gold_matched = sum(1 for h in hits if h.matched)
    recall_all = gold_matched / gold_total if gold_total else 0.0
    strict_gold_matched = sum(1 for h in hits if h.strict_matched)
    strict_recall_all = strict_gold_matched / gold_total if gold_total else 0.0

    critical_total = sum(1 for item in gold.items if item.critical)
    critical_matched = sum(1 for h in hits if h.critical and h.matched)
    recall_critical = critical_matched / critical_total if critical_total else 0.0
    strict_critical_matched = sum(1 for h in hits if h.critical and h.strict_matched)
    strict_recall_critical = strict_critical_matched / critical_total if critical_total else 0.0

    confirmed_total = len(confirmed)
    confirmed_matched = len(matched_finding_ids)
    confirmed_unmatched = confirmed_total - confirmed_matched
    unmatched_share = confirmed_unmatched / confirmed_total if confirmed_total else 0.0

    (
        calls_total,
        calls_ok,
        calls_failed,
        input_tokens,
        output_tokens,
        total_call_seconds,
        max_call_seconds,
        estimated_wall_seconds,
        estimated_cost_usd,
        unpriced_models,
    ) = _summarize_calls(run.calls)

    run_label = label or f"{run.passport.document_name} ({run.passport.mode.value})"
    return RunScore(
        label=run_label,
        document_name=run.passport.document_name,
        mode=run.passport.mode.value,
        models=list(run.passport.models),
        models_count=len(run.passport.models),
        gold_total=gold_total,
        gold_matched=gold_matched,
        recall_all=recall_all,
        strict_gold_matched=strict_gold_matched,
        strict_recall_all=strict_recall_all,
        critical_total=critical_total,
        critical_matched=critical_matched,
        recall_critical=recall_critical,
        strict_critical_matched=strict_critical_matched,
        strict_recall_critical=strict_recall_critical,
        confirmed_total=confirmed_total,
        confirmed_matched=confirmed_matched,
        confirmed_unmatched=confirmed_unmatched,
        unmatched_share=unmatched_share,
        calls_total=calls_total,
        calls_ok=calls_ok,
        calls_failed=calls_failed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_call_seconds=total_call_seconds,
        max_call_seconds=max_call_seconds,
        estimated_wall_seconds=estimated_wall_seconds,
        estimated_cost_usd=estimated_cost_usd,
        unpriced_models=unpriced_models,
        hits=hits,
        unmatched_findings=unmatched_findings,
    )


def score_runs(runs: Sequence[tuple[str, Run]], gold: GoldSet) -> EvalReport:
    """Score one or more labeled runs against a GoldSet."""
    scored = [score_run(run, gold, label=label) for label, run in runs]
    critical_total = sum(1 for item in gold.items if item.critical)
    return EvalReport(
        caveat=GOLD_CAVEAT,
        gold_document=gold.document,
        gold_total=len(gold.items),
        critical_total=critical_total,
        runs=scored,
    )


def _md_cell(text: str) -> str:
    return normalize_ws(text).replace("|", "\\|")


def _fmt_pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def _fmt_cost(score: RunScore) -> str:
    if score.estimated_cost_usd is None:
        return f"n/a (unpriced: {', '.join(score.unpriced_models)})"
    return f"${score.estimated_cost_usd:.4f}"


def _render_unmatched_table(unmatched: Sequence[UnmatchedFinding]) -> list[str]:
    if not unmatched:
        return ["No unmatched confirmed findings — every confirmed finding overlapped at least one gold item."]
    lines = [
        "| ID | Severity | Criterion | Title | Primary Quote |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for item in unmatched:
        lines.append(
            f"| {_md_cell(item.id)} | {_md_cell(item.severity)} | {_md_cell(item.criterion)} | "
            f"{_md_cell(item.title)} | {_md_cell(item.primary_quote)} |"
        )
    return lines


def _format_matrix_cell(hit: GoldHit) -> str:
    if not hit.matches:
        return "—"
    strict_ids = hit.strict_finding_ids
    loose_only_ids = [m.finding_id for m in hit.matches if not m.criterion_compatible]
    if strict_ids and loose_only_ids:
        return f"strict: {', '.join(strict_ids)}; loose: +{', +'.join(loose_only_ids)}"
    if strict_ids:
        return f"strict: {', '.join(strict_ids)}"
    return f"(loose: {', '.join(loose_only_ids)})"


def _render_single_markdown(report: EvalReport, score: RunScore) -> str:
    lines: list[str] = [
        f"# Evaluation Report: {_md_cell(score.label)}",
        "",
        f"Gold set: `{report.gold_document}` ({report.gold_total} items, {report.critical_total} critical)",
        "",
        f"> **Caveat:** {report.caveat}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Mode | `{score.mode}` |",
        f"| Models ({score.models_count}) | {_md_cell(', '.join(score.models)) or '—'} |",
        f"| Confirmed findings | {score.confirmed_total} |",
        f"| Loose recall (all) | {score.gold_matched}/{score.gold_total} ({_fmt_pct(score.recall_all)}) |",
        f"| Strict recall (all) | {score.strict_gold_matched}/{score.gold_total} ({_fmt_pct(score.strict_recall_all)}) |",
        f"| Loose recall (critical) | {score.critical_matched}/{score.critical_total} ({_fmt_pct(score.recall_critical)}) |",
        f"| Strict recall (critical) | {score.strict_critical_matched}/{score.critical_total} ({_fmt_pct(score.strict_recall_critical)}) |",
        f"| Unmatched confirmed (loose) | {score.confirmed_unmatched}/{score.confirmed_total} ({_fmt_pct(score.unmatched_share)}) |",
        f"| Calls (ok / total) | {score.calls_ok}/{score.calls_total} (failed: {score.calls_failed}) |",
        f"| Input tokens | {score.input_tokens:,} |",
        f"| Output tokens | {score.output_tokens:,} |",
        f"| Cumulative call time (s) | {score.total_call_seconds:.2f} |",
        f"| Max call time (s) | {score.max_call_seconds:.2f} |",
        f"| Estimated wall time (s) | {score.estimated_wall_seconds:.2f} |",
        f"| Estimated cost (USD, approx.) | {_fmt_cost(score)} |",
        "",
        "## Per-item hit table",
        "",
        "| Gold ID | Part | Critical | Gold Criterion | Loose | Strict | Strict Findings | All Loose Findings (Criterion) | Matched On |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for hit in score.hits:
        loose_status = "HIT" if hit.matched else "MISS"
        strict_status = "HIT" if hit.strict_matched else "MISS"
        strict_desc = ", ".join(hit.strict_finding_ids) if hit.strict_finding_ids else "—"
        if hit.matches:
            matched_desc = ", ".join(
                f"{m.finding_id} ({m.finding_criterion}, {'exact' if m.criterion_matches else 'compatible' if m.criterion_compatible else 'differs'})"
                for m in hit.matches
            )
        else:
            matched_desc = "—"
        matched_on = hit.matched_on or "—"
        crit = "yes" if hit.critical else "no"
        lines.append(
            f"| {hit.gold_id} | {hit.part} | {crit} | `{hit.gold_criterion}` | {loose_status} | {strict_status} | "
            f"{_md_cell(strict_desc)} | {_md_cell(matched_desc)} | {matched_on} |"
        )

    lines.extend(
        [
            "",
            "## Unmatched confirmed findings (for human review)",
            "",
            "Confirmed findings that did not overlap any gold item. Listed neutrally for human judgement.",
            "",
            *_render_unmatched_table(score.unmatched_findings),
            "",
        ]
    )
    return "\n".join(lines)


def _render_multi_markdown(report: EvalReport) -> str:
    runs = report.runs
    labels = [_md_cell(r.label) for r in runs]
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "| :--- | " + " | ".join(":---" for _ in runs) + " |"

    def row(metric: str, values: Sequence[str]) -> str:
        return f"| {metric} | " + " | ".join(values) + " |"

    lines: list[str] = [
        "# Evaluation Comparison",
        "",
        f"Gold set: `{report.gold_document}` ({report.gold_total} items, {report.critical_total} critical)",
        "",
        f"> **Caveat:** {report.caveat}",
        "",
        "## Summary comparison",
        "",
        header,
        sep,
        row("Mode", [f"`{r.mode}`" for r in runs]),
        row("Models count", [str(r.models_count) for r in runs]),
        row("Confirmed findings", [str(r.confirmed_total) for r in runs]),
        row("Loose recall (all)", [f"{r.gold_matched}/{r.gold_total} ({_fmt_pct(r.recall_all)})" for r in runs]),
        row("Strict recall (all)", [f"{r.strict_gold_matched}/{r.gold_total} ({_fmt_pct(r.strict_recall_all)})" for r in runs]),
        row("Loose recall (critical)", [f"{r.critical_matched}/{r.critical_total} ({_fmt_pct(r.recall_critical)})" for r in runs]),
        row("Strict recall (critical)", [f"{r.strict_critical_matched}/{r.critical_total} ({_fmt_pct(r.strict_recall_critical)})" for r in runs]),
        row("Unmatched share (loose)", [f"{r.confirmed_unmatched}/{r.confirmed_total} ({_fmt_pct(r.unmatched_share)})" for r in runs]),
        row("Calls (ok / total)", [f"{r.calls_ok}/{r.calls_total}" for r in runs]),
        row("Input tokens", [f"{r.input_tokens:,}" for r in runs]),
        row("Output tokens", [f"{r.output_tokens:,}" for r in runs]),
        row("Cumulative call time (s)", [f"{r.total_call_seconds:.2f}" for r in runs]),
        row("Max call time (s)", [f"{r.max_call_seconds:.2f}" for r in runs]),
        row("Estimated wall time (s)", [f"{r.estimated_wall_seconds:.2f}" for r in runs]),
        row("Estimated cost (USD, approx.)", [_md_cell(_fmt_cost(r)) for r in runs]),
        "",
        "## Per-gold-item hit matrix",
        "",
        "| Gold ID | Part | Critical | Gold Criterion | " + " | ".join(labels) + " |",
        "| :--- | :--- | :--- | :--- | " + " | ".join(":---" for _ in runs) + " |",
    ]

    for idx in range(report.gold_total):
        first_hit = runs[0].hits[idx]
        crit = "yes" if first_hit.critical else "no"
        cells = [_md_cell(_format_matrix_cell(r.hits[idx])) for r in runs]
        lines.append(
            f"| {first_hit.gold_id} | {first_hit.part} | {crit} | `{first_hit.gold_criterion}` | "
            + " | ".join(cells)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Unmatched confirmed findings (for human review)",
            "",
            "Confirmed findings that did not overlap any gold item. Listed neutrally for human judgement.",
        ]
    )
    for r in runs:
        lines.extend(["", f"### {_md_cell(r.label)}", "", *_render_unmatched_table(r.unmatched_findings)])
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: EvalReport) -> str:
    """Render an EvalReport as Markdown (single-run or side-by-side multi-run)."""
    if len(report.runs) == 1:
        return _render_single_markdown(report, report.runs[0])
    return _render_multi_markdown(report)


def render_json(report: EvalReport) -> str:
    """Render an EvalReport as formatted JSON."""
    return report.model_dump_json(indent=2) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tekti.evals",
        description="Score one or more saved Tekti Run JSON files against a gold defect set.",
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="One or more saved Run JSON files.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("evals/gold.yaml"),
        help="Path to the gold YAML file (default: evals/gold.yaml).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Primary stdout output format (default: markdown).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file path to write the primary output.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional file path to write JSON output alongside the primary output.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional custom label for a run (repeat once per run).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for scoring Tekti runs."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if len(args.label) > len(args.runs):
        parser.error(f"received {len(args.label)} --label values for {len(args.runs)} run files")

    gold = load_gold(args.gold)
    labeled_runs: list[tuple[str, Run]] = []
    for i, path in enumerate(args.runs):
        run = Run.model_validate_json(path.read_text(encoding="utf-8"))
        label = args.label[i] if i < len(args.label) else f"{path.stem} ({run.passport.mode.value})"
        labeled_runs.append((label, run))

    report = score_runs(labeled_runs, gold)
    primary = render_json(report) if args.format == "json" else render_markdown(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(primary, encoding="utf-8")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(render_json(report), encoding="utf-8")

    sys.stdout.write(primary)
    return 0
