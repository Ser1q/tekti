"""The fixed pipeline: ingest → segment → rules → panel → adjudicate → questions → verdict.

Only the order lives here. Every stage is a function over the contracts.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

from tekti import adjudicate, ingest, llm, panel, questions, segment
from tekti.config import Config, challenger_for, install_backend, load_config
from tekti.contracts import (
    Fact,
    Finding,
    FindingStatus,
    Mode,
    Passport,
    RawFinding,
    Run,
    Severity,
    Verdict,
)

Progress = Callable[[str], None]


def agent_version() -> str:
    """Git commit of the agent's own checkout, or the package version when not in a checkout."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return f"{commit}{'+изменения' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return f"tekti {version('tekti')}"


def _optional(module: str):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            return None
        raise


def _decide(findings: list[Finding]) -> Verdict:
    verdict = _optional("tekti.verdict")
    if verdict is not None:
        return verdict.decide(findings)
    # Stand-in until the verdict package lands; same rules as docs/review-design.md.
    confirmed = [f for f in findings if f.status == FindingStatus.CONFIRMED]
    critical = [f for f in confirmed if f.severity == Severity.CRITICAL]
    dev = not critical
    blocking_tests = [
        f for f in confirmed if f.severity != Severity.MINOR and f.criterion.value in {"verifiable", "set_validatable"}
    ]
    return Verdict(
        ready_for_development=dev,
        ready_for_testing=dev and not blocking_tests,
        fix_first=[f.id for f in critical[:10]],
        rationale=f"Подтверждено критичных дефектов: {len(critical)}.",
    )


async def run(
    input_path: Path,
    mode: Mode | str = Mode.BOARD,
    *,
    use_rules: bool = True,
    cfg: Config | None = None,
    progress: Progress | None = None,
) -> Run:
    say = progress or (lambda _msg: None)
    cfg = cfg or load_config(mode)
    install_backend(cfg)
    llm.reset_calls()
    started = datetime.now().astimezone()

    say("Чтение документа")
    doc = ingest.load(input_path)

    say(f"Выделение требований ({cfg.segmenter.model_id})")
    requirements = await segment.split(doc, cfg)
    seg = segment.last_result
    seg_dropped = [f"{d.quote[:120]} ({d.reason})" for d in seg.dropped] if seg else []
    if not requirements:
        raise RuntimeError("В документе не найдено ни одного требования.")

    facts: list[Fact] = []
    raw: list[RawFinding] = []
    rules = _optional("tekti.rules") if use_rules else None
    if rules is not None:
        say("Проверка правилами")
        facts, rule_findings = rules.run(doc, requirements)
        raw.extend(rule_findings)

    say(f"Проверка коллегией: {len(cfg.lenses)} углов × {len(cfg.panel)} семейств моделей")
    raw.extend(await panel.review(doc, requirements, facts, cfg))
    dropped = list(panel.dropped())

    say(f"Сведение и перекрёстная проверка ({len(raw)} исходных замечаний)")
    confirmed, rejected = await adjudicate.merge(doc, requirements, raw, cfg, challenger_for=challenger_for)
    failed = adjudicate.last_stats.get("challenges_failed", 0)
    if failed:
        dropped.append(f"перекрёстная проверка: не выполнено вызовов — {failed}")

    say(f"Уточняющие вопросы ({len(confirmed)} подтверждённых замечаний)")
    qs = await questions.write(confirmed, requirements, cfg, drafts=dict(adjudicate.last_question_drafts), doc=doc)

    models = list(dict.fromkeys([m.model_id for m in cfg.panel] + [c.model for c in llm.calls()]))
    return Run(
        passport=Passport(
            document_name=doc.name,
            document_sha256=doc.sha256,
            started_at=started,
            agent_version=agent_version(),
            mode=cfg.mode,
            models=models,
            dropped_reviewers=dropped,
        ),
        document=doc,
        requirements=requirements,
        segmentation_dropped=seg_dropped,
        facts=facts,
        raw_findings=raw,
        findings=confirmed,
        rejected=rejected,
        questions=qs,
        verdict=_decide(confirmed),
        calls=llm.calls(),
    )
