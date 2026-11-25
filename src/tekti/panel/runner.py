"""Fan-out of (lens, model) reviewers and light validation of what they return.

Validation here is deliberately light: quotes must be found in the document,
unknown requirement ids are mapped or stripped, malformed findings are dropped.
Merging and deduplication belong to adjudication.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from tekti import llm
from tekti.contracts import (
    Criterion,
    Document,
    Evidence,
    Fact,
    Lens,
    RawFinding,
    Requirement,
    ReviewerId,
    Role,
    Severity,
    normalize_ws,
)
from tekti.llm import LLMError, ModelSpec
from tekti.panel.prompts import system_prompt, user_prompt
from tekti.panel.schema import ReviewerFinding, ReviewerOutput


class PanelConfig(Protocol):
    @property
    def panel(self) -> Sequence[ModelSpec]: ...
    @property
    def lenses(self) -> Sequence[Lens]: ...
    @property
    def max_parallel(self) -> int: ...
    @property
    def max_output_tokens(self) -> int: ...


@dataclass
class PanelStats:
    """Counts from the last `review` call, for debugging and the run log."""

    reviewers: int = 0
    raw_findings: int = 0
    kept_findings: int = 0
    quotes_repaired: int = 0
    quotes_dropped: int = 0
    findings_without_evidence: int = 0
    findings_malformed: int = 0
    unknown_requirement_ids: int = 0
    dropped_examples: list[str] = field(default_factory=list)


# Reviewers dropped in the last `review` call, as "lens/model_id: error". The pipeline
# copies this into Passport.dropped_reviewers.
last_dropped: list[str] = []
last_stats: PanelStats = PanelStats()


# --------------------------------------------------------------------------- quote matching

_DASHES = "‐‑‒–—―−"
_QUOTES = "«»“”„‟‘’‚′″'"
_DROP = "*`"


def _loose_char(ch: str) -> str:
    """Map one character to its loose form; '' means skip it."""
    if ch in _DROP:
        return ""
    if ch in _DASHES:
        return "-"
    if ch in _QUOTES:
        return '"'
    if ch in "ёЁ":
        return "е" if ch == "ё" else "Е"
    if ch == " " or ch.isspace():
        return " "
    if ch == "…":
        return "..."
    return ch


def _loosen(text: str) -> tuple[str, list[int]]:
    """Loose form of `text` plus, for each loose char, the index of its source char."""
    out: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        mapped = _loose_char(ch)
        for m in mapped:
            if m == " " and (not out or out[-1] == " "):
                continue
            out.append(m.lower())
            index.append(i)
    return "".join(out), index


class QuoteIndex:
    """Finds a reviewer's quote in the document, exactly or after a loose repair."""

    def __init__(self, doc: Document) -> None:
        self.text = normalize_ws(doc.raw_text)
        self.loose, self.loose_index = _loosen(self.text)
        self.blocks = [(b.id, normalize_ws(b.text)) for b in doc.blocks]

    def find(self, quote: str) -> str | None:
        """Return the exact document text for `quote`, or None if it is not there."""
        q = normalize_ws(quote)
        if not q:
            return None
        if q in self.text:
            return q
        lq, _ = _loosen(q)
        lq = lq.strip(' ."')
        if len(lq) < 4:
            return None
        pos = self.loose.find(lq)
        if pos < 0:
            return None
        start = self.loose_index[pos]
        end = self.loose_index[pos + len(lq) - 1] + 1
        # Take in adjacent markup so "**X.** text" is not cut to "X.** text".
        while start > 0 and self.text[start - 1] in _DROP:
            start -= 1
        while end < len(self.text) and self.text[end] in _DROP:
            end += 1
        repaired = self.text[start:end].strip()
        return repaired if repaired and repaired in self.text else None

    def block_for(self, quote: str) -> str | None:
        for block_id, text in self.blocks:
            if quote in text:
                return block_id
        head = quote[:40]
        for block_id, text in self.blocks:
            if head in text:
                return block_id
        return None


# --------------------------------------------------------------------------- conversion

_ID_RE = re.compile(r"[\s«»\"'*`.:]+")


def _id_key(raw: str) -> str:
    return _ID_RE.sub("", raw).upper()


class _RequirementIndex:
    def __init__(self, requirements: list[Requirement]) -> None:
        self.ids = {r.id for r in requirements}
        self.by_key: dict[str, str] = {}
        for r in requirements:
            self.by_key.setdefault(_id_key(r.id), r.id)
        for r in requirements:
            if r.source_id:
                self.by_key.setdefault(_id_key(r.source_id), r.id)
        self.quotes = [(r.id, normalize_ws(r.quote)) for r in requirements]

    def resolve(self, raw: str | None) -> str | None:
        if not raw:
            return None
        if raw in self.ids:
            return raw
        return self.by_key.get(_id_key(raw))

    def containing(self, quote: str) -> list[str]:
        """Requirement ids whose quote contains, or is contained in, the evidence quote."""
        return [rid for rid, rq in self.quotes if rq and (quote in rq or (len(rq) >= 20 and rq in quote))]


def _enum[E](enum: type[E], raw: str | None) -> E | None:
    if raw is None:
        return None
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return enum(key)  # type: ignore[call-arg]
    except ValueError:
        return None


def _convert(
    item: ReviewerFinding,
    reviewer: ReviewerId,
    quotes: QuoteIndex,
    reqs: _RequirementIndex,
    stats: PanelStats,
) -> RawFinding | None:
    criterion = _enum(Criterion, item.criterion)
    severity = _enum(Severity, item.severity)
    if criterion is None or severity is None or not item.title.strip():
        stats.findings_malformed += 1
        stats.dropped_examples.append(f"malformed: {item.criterion}/{item.severity}: {item.title[:80]}")
        return None

    evidence: list[Evidence] = []
    for ev in item.evidence:
        exact = quotes.find(ev.quote)
        if exact is None:
            stats.quotes_dropped += 1
            stats.dropped_examples.append(f"quote not found: {ev.quote[:120]}")
            continue
        if exact != normalize_ws(ev.quote):
            stats.quotes_repaired += 1
        rid = reqs.resolve(ev.requirement_id)
        if ev.requirement_id and rid is None:
            stats.unknown_requirement_ids += 1
        if any(e.quote == exact for e in evidence):
            continue
        evidence.append(Evidence(quote=exact, block_id=quotes.block_for(exact), requirement_id=rid))
    if not evidence:
        stats.findings_without_evidence += 1
        return None

    requirement_ids: list[str] = []
    named = [r for r in item.requirement_ids if r and r.strip()]
    for raw in named:
        rid = reqs.resolve(raw)
        if rid is None:
            stats.unknown_requirement_ids += 1
        elif rid not in requirement_ids:
            requirement_ids.append(rid)
    if named and not requirement_ids:
        # The reviewer meant specific requirements but named none we know: infer from evidence.
        for e in evidence:
            for rid in [e.requirement_id] if e.requirement_id else reqs.containing(e.quote):
                if rid and rid not in requirement_ids:
                    requirement_ids.append(rid)

    question = (item.question or "").strip() or None
    addressee = _enum(Role, item.addressee) if question else None
    return RawFinding(
        reviewer=reviewer,
        requirement_ids=requirement_ids,
        criterion=criterion,
        severity=severity,
        title=item.title.strip(),
        explanation=item.explanation.strip(),
        evidence=evidence,
        question=question,
        addressee=addressee,
    )


def convert_output(
    output: ReviewerOutput,
    reviewer: ReviewerId,
    quotes: QuoteIndex,
    requirements: list[Requirement],
    stats: PanelStats | None = None,
) -> list[RawFinding]:
    """Turn one reviewer's answer into validated RawFindings."""
    stats = stats if stats is not None else PanelStats()
    reqs = _RequirementIndex(requirements)
    result: list[RawFinding] = []
    for item in output.findings:
        stats.raw_findings += 1
        finding = _convert(item, reviewer, quotes, reqs, stats)
        if finding is not None:
            result.append(finding)
    stats.kept_findings += len(result)
    return result


# --------------------------------------------------------------------------- fan-out


async def review(
    doc: Document, requirements: list[Requirement], facts: list[Fact], cfg: PanelConfig
) -> list[RawFinding]:
    """Run every (lens, model) reviewer concurrently and return their validated findings."""
    global last_stats
    last_dropped.clear()
    stats = PanelStats()
    last_stats = stats

    user = user_prompt(doc, requirements, facts)
    quotes = QuoteIndex(doc)
    semaphore = asyncio.Semaphore(max(1, cfg.max_parallel))
    pairs = [(lens, model) for lens in cfg.lenses for model in cfg.panel]
    stats.reviewers = len(pairs)

    async def one(lens: Lens, model: ModelSpec) -> list[RawFinding]:
        reviewer = ReviewerId(lens=lens, model=model.model_id)
        async with semaphore:
            try:
                output = await llm.ask(
                    model,
                    system=system_prompt(lens),
                    user=user,
                    schema=ReviewerOutput,
                    stage="panel",
                    lens=lens,
                    max_tokens=cfg.max_output_tokens,
                )
            except LLMError as exc:
                last_dropped.append(f"{lens.value}/{model.model_id}: {exc}"[:500])
                return []
        try:
            return convert_output(output, reviewer, quotes, requirements, stats)
        except Exception as exc:  # noqa: BLE001 - a bug in conversion must not sink the other reviewers
            last_dropped.append(f"{lens.value}/{model.model_id}: conversion failed: {type(exc).__name__}: {exc}"[:500])
            return []

    results = await asyncio.gather(*(one(lens, model) for lens, model in pairs))
    return [finding for batch in results for finding in batch]
