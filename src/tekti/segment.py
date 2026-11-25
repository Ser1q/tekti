"""Segmentation (P2): split a Document into Requirements with one model call.

`split(doc, cfg) -> list[Requirement]` is the contract. The stage also records what it had to
drop; read it from `last_result` (a `SegmentResult`) right after `split`, or call
`split_with_report` to get it directly.

Quote handling:
  * every quote must be a substring of `normalize_ws(doc.raw_text)`;
  * a quote that only misses Markdown emphasis or code marks ("Ф-1." for "**Ф-1.**") is
    repaired by mapping it back onto the exact document span (the stored quote is then the
    document text, markup included); nothing else is repaired;
  * candidates with failing quotes are sent back once with the reasons; whatever still fails
    is dropped and listed in `SegmentResult.dropped`.
Block ids are recomputed in code from where the quote sits in the document; the model's
block ids only break ties when a quote occurs more than once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from tekti import llm
from tekti.contracts import Document, Requirement, RequirementKind, normalize_ws
from tekti.llm import LLMError, ModelSpec
from tekti.prompts import segment as prompts

STAGE = "segment"
_MARKUP = frozenset("*`")


class SegmentConfig(Protocol):
    segmenter: ModelSpec
    max_output_tokens: int


# --------------------------------------------------------------------------- model output schema


class ChildCandidate(BaseModel):
    quote: str = Field(description="Exact contiguous piece of the parent's quote")
    summary: str = Field(description="Short wording in Russian")
    kind: RequirementKind | None = None


class Candidate(BaseModel):
    block_ids: list[str] = Field(default_factory=list)
    quote: str = Field(description="Exact text copied from the document")
    source_id: str | None = Field(default=None, description="Identifier given by the document, e.g. Ф-1")
    kind: RequirementKind
    summary: str = Field(description="Short wording in Russian")
    children: list[ChildCandidate] = Field(default_factory=list)
    replaces: int | None = Field(
        default=None, description="Only in a retry: the number of the failed requirement this one corrects"
    )


class SegmentOutput(BaseModel):
    requirements: list[Candidate]


# --------------------------------------------------------------------------- result


@dataclass
class DroppedCandidate:
    quote: str
    summary: str
    source_id: str | None
    reason: str
    parent_quote: str | None = None  # set when only a child was dropped


@dataclass
class SegmentResult:
    requirements: list[Requirement]
    dropped: list[DroppedCandidate] = field(default_factory=list)
    retried: int = 0  # candidates sent back for a retry
    repaired: int = 0  # quotes fixed by markup-insensitive matching


last_result: SegmentResult | None = None


# --------------------------------------------------------------------------- quote location


@dataclass
class _Span:
    start: int
    end: int
    quote: str
    repaired: bool


class _DocIndex:
    """Normalized document text, a markup-free view of it, and block spans inside it."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.text = normalize_ws(doc.raw_text)
        stripped: list[str] = []
        self.back: list[int] = []  # index in stripped view -> index in self.text
        for i, ch in enumerate(self.text):
            if ch in _MARKUP:
                continue
            if ch == " " and stripped and stripped[-1] == " ":
                continue
            stripped.append(ch)
            self.back.append(i)
        self.stripped = "".join(stripped)
        self.block_spans: list[tuple[str, int, int]] = []
        cursor = 0
        for b in doc.blocks:
            nt = normalize_ws(b.text)
            if not nt:
                continue
            pos = self.text.find(nt, cursor)
            if pos < 0:
                pos = self.text.find(nt)
            if pos < 0:
                continue
            self.block_spans.append((b.id, pos, pos + len(nt)))
            cursor = pos + len(nt)
        self.block_ids = {b.id for b in doc.blocks}

    def blocks_for(self, start: int, end: int) -> list[str]:
        return [bid for bid, s, e in self.block_spans if s < end and start < e]

    def _occurrences(self, hay: str, needle: str, lo: int, hi: int) -> list[int]:
        out, pos = [], hay.find(needle, lo, hi)
        while pos >= 0:
            out.append(pos)
            pos = hay.find(needle, pos + 1, hi)
        return out

    def locate(self, quote: str, hint: list[str], within: tuple[int, int] | None = None) -> _Span | None:
        nq = normalize_ws(quote)
        if not nq:
            return None
        lo, hi = within if within else (0, len(self.text))
        spans = [(p, p + len(nq), False) for p in self._occurrences(self.text, nq, lo, hi)]
        if not spans:
            sq = normalize_ws("".join(ch for ch in nq if ch not in _MARKUP))
            if sq:
                for p in self._occurrences(self.stripped, sq, 0, len(self.stripped)):
                    s, e = self.back[p], self.back[p + len(sq) - 1] + 1
                    # Take in markup glued to the ends ("**Ф-1.**"), so emphasis stays balanced.
                    while s > 0 and self.text[s - 1] in _MARKUP:
                        s -= 1
                    while e < len(self.text) and self.text[e] in _MARKUP:
                        e += 1
                    if lo <= s and e <= hi:
                        spans.append((s, e, True))
        if not spans:
            return None
        hint_set = set(hint)
        best = next((sp for sp in spans if hint_set & set(self.blocks_for(sp[0], sp[1]))), spans[0])
        s, e, repaired = best
        return _Span(s, e, self.text[s:e], repaired)


# --------------------------------------------------------------------------- validation


@dataclass
class _Checked:
    cand: Candidate
    span: _Span | None
    children: list[tuple[ChildCandidate, _Span | None, str | None]]  # (child, span, failure reason)
    reason: str | None  # failure of the parent quote

    @property
    def ok(self) -> bool:
        return self.reason is None and all(r is None for _, _, r in self.children)

    def reasons(self) -> list[str]:
        out = [f"quote of the requirement: {self.reason}"] if self.reason else []
        out += [f'child quote "{c.quote}": {r}' for c, _, r in self.children if r]
        return out


def _check(index: _DocIndex, cand: Candidate) -> _Checked:
    span = index.locate(cand.quote, cand.block_ids)
    if span is None:
        reason = "quote is empty" if not normalize_ws(cand.quote) else "not found in the document text"
        return _Checked(cand, None, [(c, None, None) for c in cand.children], reason)
    kids: list[tuple[ChildCandidate, _Span | None, str | None]] = []
    for child in cand.children:
        cspan = index.locate(child.quote, cand.block_ids, within=(span.start, span.end))
        if cspan is not None:
            kids.append((child, cspan, None))
        elif index.locate(child.quote, cand.block_ids) is not None:
            kids.append((child, None, "found in the document but not inside the parent's quote"))
        else:
            kids.append((child, None, "not found in the document text"))
    return _Checked(cand, span, kids, None)


def _render_blocks(doc: Document) -> str:
    parts = []
    for b in doc.blocks:
        path = " > ".join(b.section_path) or "—"
        parts.append(f"[{b.id}] ({path})\n{b.text}")
    return "\n\n".join(parts)


def _describe_failures(failed: list[_Checked]) -> str:
    items = []
    for n, chk in enumerate(failed, 1):
        payload = chk.cand.model_dump(mode="json", exclude={"replaces"})
        items.append(
            f"{n}. {json.dumps(payload, ensure_ascii=False)}\n   reasons: " + "; ".join(chk.reasons())
        )
    return "\n".join(items) + "\nSet `replaces` to the number of the failed requirement each answer corrects."


# --------------------------------------------------------------------------- stage


async def split_with_report(doc: Document, cfg: SegmentConfig) -> SegmentResult:
    index = _DocIndex(doc)
    blocks = _render_blocks(doc)
    first = await llm.ask(
        cfg.segmenter,
        system=prompts.SYSTEM,
        user=prompts.USER_TEMPLATE.format(name=doc.name, blocks=blocks),
        schema=SegmentOutput,
        stage=STAGE,
        max_tokens=cfg.max_output_tokens,
    )
    checked = [_check(index, c) for c in first.requirements]
    failed = [c for c in checked if not c.ok]
    final: list[_Checked] = [c for c in checked if c.ok]
    dropped: list[DroppedCandidate] = []

    if failed:
        replacements: dict[int, _Checked] = {}
        retry_error: str | None = None
        try:
            second = await llm.ask(
                cfg.segmenter,
                system=prompts.SYSTEM,
                user=prompts.RETRY_TEMPLATE.format(
                    failures=_describe_failures(failed), name=doc.name, blocks=blocks
                ),
                schema=SegmentOutput,
                stage=STAGE,
                max_tokens=cfg.max_output_tokens,
            )
            answers = second.requirements
            positional = all(a.replaces is None for a in answers) and len(answers) == len(failed)
            for pos, ans in enumerate(answers):
                n = pos + 1 if positional else ans.replaces
                if n is not None and 1 <= n <= len(failed) and n not in replacements:
                    replacements[n] = _check(index, ans)
        except LLMError as exc:
            retry_error = f"retry call failed: {exc}"[:300]
        for n, orig in enumerate(failed, 1):
            best = replacements.get(n)
            if best is not None and best.ok:
                final.append(best)
                continue
            # Salvage what holds: the newest version whose parent quote is valid, minus bad children.
            version = best if best is not None and best.reason is None else orig
            why = retry_error or "quote still not exact after one retry"
            if version.reason is not None:
                dropped.append(
                    DroppedCandidate(
                        quote=orig.cand.quote,
                        summary=orig.cand.summary,
                        source_id=orig.cand.source_id,
                        reason=f"{why}; " + "; ".join(orig.reasons()),
                    )
                )
                continue
            for child, _, r in version.children:
                if r:
                    dropped.append(
                        DroppedCandidate(
                            quote=child.quote,
                            summary=child.summary,
                            source_id=None,
                            reason=f"{why}; child quote {r}",
                            parent_quote=version.cand.quote,
                        )
                    )
            version.children = [k for k in version.children if k[2] is None]
            final.append(version)

    requirements, repaired = _assign_ids(index, final)
    return SegmentResult(requirements=requirements, dropped=dropped, retried=len(failed), repaired=repaired)


def _assign_ids(index: _DocIndex, accepted: list[_Checked]) -> tuple[list[Requirement], int]:
    seen: set[tuple[int, int]] = set()
    unique: list[_Checked] = []
    for chk in sorted(accepted, key=lambda c: (c.span.start, -c.span.end)):  # type: ignore[union-attr]
        key = (chk.span.start, chk.span.end)  # type: ignore[union-attr]
        if key in seen:
            continue
        seen.add(key)
        unique.append(chk)

    out: list[Requirement] = []
    repaired = 0

    def make(span: _Span, cand_kind: RequirementKind, summary: str, source_id: str | None, hint: list[str],
             parent_id: str | None) -> Requirement:
        nonlocal repaired
        repaired += span.repaired
        block_ids = index.blocks_for(span.start, span.end) or [b for b in hint if b in index.block_ids]
        req = Requirement(
            id=f"R-{len(out) + 1:03d}",
            source_id=source_id,
            kind=cand_kind,
            summary=summary.strip(),
            quote=span.quote,
            block_ids=block_ids,
            parent_id=parent_id,
        )
        out.append(req)
        return req

    for chk in unique:
        cand, span = chk.cand, chk.span
        assert span is not None
        source_id = cand.source_id.strip() if cand.source_id else None
        if source_id and normalize_ws(source_id) not in index.text:
            source_id = None
        parent = make(span, cand.kind, cand.summary, source_id, cand.block_ids, None)
        kids = sorted(
            ((c, s) for c, s, r in chk.children if r is None and s is not None), key=lambda k: (k[1].start, k[1].end)
        )
        child_seen: set[tuple[int, int]] = set()
        for child, cspan in kids:
            ckey = (cspan.start, cspan.end)
            if ckey in child_seen or ckey == (span.start, span.end):
                continue
            child_seen.add(ckey)
            make(cspan, child.kind or cand.kind, child.summary, None, cand.block_ids, parent.id)
    return out, repaired


async def split(doc: Document, cfg: SegmentConfig) -> list[Requirement]:
    """Contract entry point. Details of drops and retries are left in `last_result`."""
    global last_result
    result = await split_with_report(doc, cfg)
    last_result = result
    return result.requirements
