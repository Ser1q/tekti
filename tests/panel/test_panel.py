from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tekti import llm, panel
from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Lens,
    Requirement,
    RequirementKind,
    Severity,
)
from tekti.llm import ModelSpec
from tekti.panel.prompts import system_prompt, user_prompt
from tekti.panel.schema import ReviewerEvidence, ReviewerFinding, ReviewerOutput

# A tiny invented document (library loans), not the evaluation document.
_LINES = [
    "# Выдача книг",
    "## Правила",
    "* **К-1.** Читатель может взять не более 5 книг одновременно.",
    "* **К-2.** При просрочке возврата более недели абонемент приостанавливается.",
    "## Критерии приёмки",
    "| Читатель с 5 книгами | Выдать шестую книгу | Книга выдана |",
]


def _doc() -> Document:
    blocks = [
        Block(id="B-0001", kind=BlockKind.HEADING, text=_LINES[0], section_path=[], line_start=1, line_end=1),
        Block(id="B-0002", kind=BlockKind.HEADING, text=_LINES[1], section_path=["Правила"], line_start=2, line_end=2),
        Block(
            id="B-0003", kind=BlockKind.LIST_ITEM, text=_LINES[2], section_path=["Правила"], line_start=3, line_end=3
        ),
        Block(
            id="B-0004", kind=BlockKind.LIST_ITEM, text=_LINES[3], section_path=["Правила"], line_start=4, line_end=4
        ),
        Block(
            id="B-0005",
            kind=BlockKind.TABLE_ROW,
            text=_LINES[5],
            section_path=["Критерии приёмки"],
            line_start=6,
            line_end=6,
        ),
    ]
    return Document(
        name="booking.md",
        source_path="booking.md",
        format="markdown",
        sha256="0" * 64,
        raw_text="\n".join(_LINES),
        blocks=blocks,
    )


def _reqs() -> list[Requirement]:
    return [
        Requirement(
            id="R-001",
            source_id="К-1",
            kind=RequirementKind.FUNCTIONAL,
            summary="Не более 5 книг",
            quote="Читатель может взять не более 5 книг одновременно.",
            block_ids=["B-0003"],
        ),
        Requirement(
            id="R-002",
            source_id="К-2",
            kind=RequirementKind.FUNCTIONAL,
            summary="Приостановка при просрочке",
            quote="При просрочке возврата более недели абонемент приостанавливается.",
            block_ids=["B-0004"],
        ),
        Requirement(
            id="R-003",
            source_id=None,
            kind=RequirementKind.ACCEPTANCE,
            summary="Шестая книга выдаётся",
            quote="| Читатель с 5 книгами | Выдать шестую книгу | Книга выдана |",
            block_ids=["B-0005"],
        ),
    ]


@dataclass
class Cfg:
    panel: list[ModelSpec]
    lenses: list[Lens]
    max_parallel: int = 3
    max_output_tokens: int = 2000


def _models() -> list[ModelSpec]:
    return [
        ModelSpec(family="anthropic", model_id="fake.claude"),
        ModelSpec(family="xai", model_id="fake.grok"),
    ]


def _good_finding() -> ReviewerFinding:
    return ReviewerFinding(
        requirement_ids=["R-001", "R-003"],
        criterion="set_consistent",
        severity="critical",
        title="Строка приёмки выдаёт шестую книгу",
        explanation="К-1 разрешает 5 книг, строка приёмки выдаёт шестую.",
        evidence=[
            ReviewerEvidence(quote="не более 5 книг одновременно", requirement_id="R-001"),
            ReviewerEvidence(quote="Выдать шестую книгу", requirement_id="R-003"),
        ],
        question="Какой предел верен: 5 или 6 книг?",
        addressee="product_owner",
    )


@dataclass
class FakeBackend:
    output: ReviewerOutput | None = None
    fail_for: set[tuple[str, str]] = field(default_factory=set)  # (model_id, lens marker in system)
    seen: list[tuple[str, str, str]] = field(default_factory=list)
    in_flight: int = 0
    peak: int = 0

    async def complete(self, model, *, system, user, schema, max_tokens):
        import asyncio

        self.seen.append((model.model_id, system, user))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        for model_id, marker in self.fail_for:
            if model.model_id == model_id and marker in system:
                raise RuntimeError("throttled")
        assert schema is ReviewerOutput
        return (self.output or ReviewerOutput(findings=[_good_finding()])), 100, 50


@pytest.fixture(autouse=True)
def _reset():
    llm.reset_calls()
    yield
    llm.reset_calls()


async def test_fan_out_runs_every_lens_model_pair_under_the_semaphore():
    backend = FakeBackend()
    llm.use_backend(backend)
    cfg = Cfg(panel=_models(), lenses=list(Lens), max_parallel=3)

    findings = await panel.review(_doc(), _reqs(), [], cfg)

    assert len(backend.seen) == len(Lens) * 2
    assert backend.peak <= 3
    assert len(findings) == len(Lens) * 2
    reviewers = {(f.reviewer.lens, f.reviewer.model) for f in findings}
    assert reviewers == {(lens, m.model_id) for lens in Lens for m in _models()}
    assert all(c.stage == "panel" for c in llm.calls())
    assert panel.dropped() == []


async def test_one_failing_reviewer_is_dropped_and_recorded():
    tester_marker = system_prompt(Lens.TESTER).split("\n")[1][:40]
    backend = FakeBackend(fail_for={("fake.grok", tester_marker)})
    llm.use_backend(backend)
    cfg = Cfg(panel=_models(), lenses=[Lens.FUNCTIONAL, Lens.TESTER])

    findings = await panel.review(_doc(), _reqs(), [], cfg)

    assert len(findings) == 3
    assert ("tester", "fake.grok") not in {(f.reviewer.lens.value, f.reviewer.model) for f in findings}
    assert len(panel.dropped()) == 1
    assert panel.dropped()[0].startswith("tester/fake.grok: ")
    assert "throttled" in panel.dropped()[0]
    assert panel.last_dropped == panel.dropped()


async def test_reviewer_id_and_fields_are_converted():
    llm.use_backend(FakeBackend())
    cfg = Cfg(panel=_models()[:1], lenses=[Lens.API_CONTRACT])

    [finding] = await panel.review(_doc(), _reqs(), [], cfg)

    assert finding.reviewer.lens is Lens.API_CONTRACT
    assert finding.reviewer.model == "fake.claude"
    assert finding.criterion is Criterion.SET_CONSISTENT
    assert finding.severity is Severity.CRITICAL
    assert finding.requirement_ids == ["R-001", "R-003"]
    assert [e.block_id for e in finding.evidence] == ["B-0003", "B-0005"]
    assert finding.addressee is not None and finding.addressee.value == "product_owner"


async def test_inexact_quotes_are_removed_and_empty_findings_dropped():
    invented = ReviewerFinding(
        requirement_ids=["R-002"],
        criterion="complete",
        severity="major",
        title="Выдуманная цитата",
        explanation="Цитаты нет в документе.",
        evidence=[ReviewerEvidence(quote="абонемент закрывается навсегда")],
    )
    mixed = _good_finding()
    mixed.evidence.append(ReviewerEvidence(quote="выдача без ограничений"))
    llm.use_backend(FakeBackend(output=ReviewerOutput(findings=[invented, mixed])))
    cfg = Cfg(panel=_models()[:1], lenses=[Lens.FUNCTIONAL])

    findings = await panel.review(_doc(), _reqs(), [], cfg)

    assert len(findings) == 1
    assert [e.quote for e in findings[0].evidence] == [
        "не более 5 книг одновременно",
        "Выдать шестую книгу",
    ]
    assert panel.stats().quotes_dropped == 2
    assert panel.stats().findings_without_evidence == 1


async def test_typographic_drift_in_a_quote_is_repaired_to_the_exact_text():
    drift = ReviewerFinding(
        requirement_ids=["R-001"],
        criterion="singular",
        severity="minor",
        title="Кавычки и разметка потеряны",
        explanation="Проверка восстановления цитаты.",
        evidence=[ReviewerEvidence(quote="К-1. Читатель может взять")],
    )
    llm.use_backend(FakeBackend(output=ReviewerOutput(findings=[drift])))
    cfg = Cfg(panel=_models()[:1], lenses=[Lens.FUNCTIONAL])

    [finding] = await panel.review(_doc(), _reqs(), [], cfg)

    quote = finding.evidence[0].quote
    assert quote in " ".join(_LINES)
    assert "Читатель может взять" in quote
    assert panel.stats().quotes_repaired == 1


async def test_unknown_requirement_ids_are_mapped_or_stripped():
    item = ReviewerFinding(
        requirement_ids=["К-2", "R-999"],  # a document id the reviewer should not have used, and a made-up one
        criterion="complete",
        severity="major",
        title="Срок приостановки не задан",
        explanation="Не сказано, на сколько приостанавливается абонемент.",
        evidence=[ReviewerEvidence(quote="абонемент приостанавливается", requirement_id="R-777")],
    )
    only_unknown = ReviewerFinding(
        requirement_ids=["R-404"],
        criterion="complete",
        severity="major",
        title="Неизвестный идентификатор",
        explanation="Требование определяется по цитате.",
        evidence=[ReviewerEvidence(quote="Читатель может взять не более 5 книг одновременно.")],
    )
    bad_enum = _good_finding()
    bad_enum.criterion = "nice_to_have"
    llm.use_backend(FakeBackend(output=ReviewerOutput(findings=[item, only_unknown, bad_enum])))
    cfg = Cfg(panel=_models()[:1], lenses=[Lens.DEVELOPER])

    findings = await panel.review(_doc(), _reqs(), [], cfg)

    assert len(findings) == 2
    assert findings[0].requirement_ids == ["R-002"]
    assert findings[0].evidence[0].requirement_id is None
    assert findings[1].requirement_ids == ["R-001"]
    assert panel.stats().findings_malformed == 1


def test_document_text_never_appears_in_the_system_prompt():
    doc = _doc()
    for lens in Lens:
        system = system_prompt(lens)
        for block in doc.blocks:
            assert block.text not in system
        assert doc.name not in system
    user = user_prompt(doc, _reqs(), [])
    assert "<document>" in user and "</document>" in user
    assert "<requirements>" in user and "R-002 | К-2 | functional" in user
    for block in doc.blocks:
        assert f"[{block.id}] {block.text}" in user


async def test_system_prompt_is_sent_as_system_and_document_as_user():
    backend = FakeBackend()
    llm.use_backend(backend)
    await panel.review(_doc(), _reqs(), [], Cfg(panel=_models()[:1], lenses=[Lens.TESTER]))
    [(_, system, user)] = backend.seen
    assert "embedded_instruction" in system
    assert "Шаг 1" in system  # simulation lens method
    assert _LINES[2] not in system
    assert _LINES[2] in user


def test_each_lens_has_its_own_method():
    prompts = {lens: system_prompt(lens) for lens in Lens}
    assert len(set(prompts.values())) == len(Lens)
    for lens in (Lens.TESTER, Lens.DEVELOPER):
        assert "worksheet" in prompts[lens]
        assert "question обязательно" in prompts[lens]
