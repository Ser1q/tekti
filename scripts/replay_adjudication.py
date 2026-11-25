"""Replay adjudication and questions on a saved run, without re-running the panel.

    uv run python scripts/replay_adjudication.py runs/board-2.run.json [--out new.run.json] [--no-llm]

Loads the run, re-runs `adjudicate.merge` on `run.raw_findings` with the run's document and
requirements, re-runs `questions.write` and `verdict.decide`, writes the new run JSON and
prints before/after numbers: confirmed count, set-level vs single-requirement findings,
findings per requirement, and confirmed pairs whose evidence overlaps strongly.

--no-llm installs a stub backend: every challenge is upheld, nothing is grouped by a model,
the question writer returns nothing (so only the fallback questions appear). Use it for a
fast structural check of id normalisation and code-level merging.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from tekti import adjudicate, llm, questions, verdict
from tekti.adjudicate import _strong_overlap
from tekti.config import challenger_for, install_backend, load_config
from tekti.contracts import SET_CRITERIA, Finding, Run
from tekti.prompts.challenge import ChallengeAnswer

REPLAYED_STAGES = {"challenge", "merge", "questions"}


class StubBackend:
    async def complete(self, model, *, system, user, schema, max_tokens):
        if schema is ChallengeAnswer:
            return ChallengeAnswer(reasoning="stub: upheld without a model", upheld=True), 0, 0
        return schema(), 0, 0


def likely_duplicates(findings: list[Finding]) -> list[tuple[Finding, Finding]]:
    """Pairs whose evidence covers each other both ways (every quote strongly overlaps one on the other side)."""

    def covered(src: list[str], dst: list[str]) -> bool:
        return bool(src) and all(any(_strong_overlap(x, y) for y in dst) for x in src)

    out = []
    for i, a in enumerate(findings):
        for b in findings[i + 1 :]:
            qa = [e.quote for e in a.evidence]
            qb = [e.quote for e in b.evidence]
            if covered(qa, qb) and covered(qb, qa):
                out.append((a, b))
    return out


def summarize(label: str, run: Run) -> None:
    fs = run.findings
    individual = [f for f in fs if f.criterion not in SET_CRITERIA]
    one_id = [f for f in individual if len(f.requirement_ids) == 1]
    per_req = Counter(r for f in fs for r in f.requirement_ids)
    print(f"\n== {label}")
    print(f"confirmed: {len(fs)}  rejected: {len(run.rejected)}  questions: {len(run.questions)}")
    print(f"set-level: {sum(f.is_set_level for f in fs)}  single-requirement: {sum(not f.is_set_level for f in fs)}")
    print(f"individual-criterion findings with exactly one id: {len(one_id)}/{len(individual)}")
    print("id count per finding:", dict(sorted(Counter(len(f.requirement_ids) for f in fs).items())))
    print("findings per requirement:", " ".join(f"{r}:{n}" for r, n in sorted(per_req.items())))
    dups = likely_duplicates(fs)
    print(f"strongly overlapping confirmed pairs: {len(dups)}")
    for a, b in dups:
        print(f"  {a.id} [{a.criterion.value} {','.join(a.requirement_ids)}] {a.title}")
        print(f"  {b.id} [{b.criterion.value} {','.join(b.requirement_ids)}] {b.title}")
        print("  --")


async def replay(run: Run, no_llm: bool) -> Run:
    cfg = load_config(run.passport.mode)
    if no_llm:
        llm.use_backend(StubBackend())
    else:
        install_backend(cfg)
    llm.reset_calls()
    confirmed, rejected = await adjudicate.merge(
        run.document, run.requirements, run.raw_findings, cfg, challenger_for=challenger_for
    )
    print("adjudication stats:", adjudicate.last_stats, file=sys.stderr)
    for line in adjudicate.last_merge_log:
        print("  merged by model:", line, file=sys.stderr)
    qs = await questions.write(
        confirmed, run.requirements, cfg, drafts=dict(adjudicate.last_question_drafts), doc=run.document
    )
    print("question stats:", questions.last_stats, file=sys.stderr)
    calls = [c for c in run.calls if c.stage not in REPLAYED_STAGES] + llm.calls()
    return run.model_copy(
        update={
            "findings": confirmed,
            "rejected": rejected,
            "questions": qs,
            "verdict": verdict.decide(confirmed),
            "calls": calls,
        }
    )


def _dump(model: BaseModel) -> str:
    return model.model_dump_json(indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_json", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args(argv)

    run = Run.model_validate_json(args.run_json.read_text(encoding="utf-8"))
    if not run.raw_findings:
        print(f"{args.run_json}: the run has no raw_findings; nothing to replay", file=sys.stderr)
        return 2
    out = args.out or args.run_json.with_name(args.run_json.name.replace(".run.json", "") + ".replay.run.json")
    new = asyncio.run(replay(run, args.no_llm))
    out.write_text(_dump(new), encoding="utf-8")
    summarize(f"before ({args.run_json})", run)
    summarize(f"after ({out})", new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
