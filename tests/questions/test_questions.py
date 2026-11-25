"""Question writer tests: offline, with a fake model backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tekti import llm, questions
from tekti.contracts import (
    Criterion,
    Evidence,
    Finding,
    FindingStatus,
    Requirement,
    RequirementKind,
    ReviewerId,
    Role,
    Severity,
)
from tekti.llm import LLMError, ModelSpec
from tekti.prompts.questions import DraftQuestion, QuestionsAnswer

WRITER = ModelSpec(family="anthropic", model_id="writer")
CFG = SimpleNamespace(panel=[WRITER], question_writer=WRITER, max_parallel=2, max_output_tokens=4000)

REQS = [
    Requirement(
        id=f"R-00{i}",
        source_id=f"П-{i}",
        kind=RequirementKind.FUNCTIONAL,
        summary=f"Правило {i}",
        quote=f"Правило номер {i}.",
        block_ids=[f"B-000{i}"],
    )
    for i in range(1, 6)
]


def finding(fid, sev, reqs, title="Замечание", quote="Правило номер 1.", crit=Criterion.SET_CONSISTENT):
    return Finding(
        id=fid,
        requirement_ids=reqs,
        criterion=crit,
        severity=sev,
        title=title,
        explanation="Пояснение.",
        evidence=[Evidence(quote=quote)],
        raised_by=[ReviewerId(model="rules")],
        families=1,
        status=FindingStatus.CONFIRMED,
    )


def dq(ids, text, role=Role.PRODUCT_OWNER, options=(), blocks="Расчёт бонуса нельзя реализовать."):
    return DraftQuestion(finding_ids=list(ids), text=text, addressee=role, blocks=blocks, options=list(options))


class FakeBackend:
    def __init__(self, main, followup=None):
        self.main = main
        self.followup = followup or (lambda user: QuestionsAnswer(questions=[]))
        self.users: list[str] = []

    async def complete(self, model, *, system, user, schema, max_tokens):
        assert schema is QuestionsAnswer
        self.users.append(user)
        handler = self.followup if "ещё нет вопроса" in user else self.main
        out = handler(user)
        if isinstance(out, Exception):
            raise out
        return out, 100, 50


@pytest.fixture
def findings():
    return [
        finding("F-001", Severity.CRITICAL, ["R-001", "R-002"], title="П-1 и П-2 противоречат"),
        finding("F-002", Severity.CRITICAL, ["R-003"], title="Нельзя проверить П-3", quote="Правило номер 3."),
        finding("F-003", Severity.MAJOR, ["R-004"], title="Неясный срок"),
        finding("F-004", Severity.MAJOR, ["R-004", "R-005"], title="Не указана единица"),
        finding("F-005", Severity.MINOR, [], title="Нет идентификаторов", crit=Criterion.TRACEABLE),
    ]


async def test_invalid_ids_stripped_and_ranking(findings):
    backend = FakeBackend(
        main=lambda user: QuestionsAnswer(
            questions=[
                dq(["F-005"], "Кто добавит идентификаторы к требованиям раздела 3?", Role.ANALYST),
                dq(
                    ["F-003", "F-004", "F-999"],
                    "В каких единицах задан срок в «Правило номер 4»?",
                    Role.DEVELOPER,
                    options=["в секундах", "в минутах", ""],
                ),
                dq(["F-777"], "Вопрос с выдуманным замечанием, который надо выбросить?"),
                dq(["F-002"], "Какой ожидаемый результат у проверки «Правило номер 3»?", Role.TESTER, options=["один"]),
                dq(
                    ["F-001"], "Какое правило главнее: «Правило номер 1» или «Правило номер 2»?", options=["П-1", "П-2"]
                ),
            ]
        )
    )
    llm.use_backend(backend)
    qs = await questions.write(findings, REQS, CFG, drafts={})
    assert len(backend.users) == 1  # every critical covered: no follow-up
    assert [q.finding_ids for q in qs] == [["F-001"], ["F-002"], ["F-003", "F-004"], ["F-005"]]
    assert [q.id for q in qs] == ["Q-001", "Q-002", "Q-003", "Q-004"]
    assert [q.priority for q in qs] == [1, 2, 3, 4]
    assert qs[0].options == ["П-1", "П-2"]
    assert qs[1].options == []  # a single option is not a choice
    assert qs[2].options == ["в секундах", "в минутах"]
    assert questions.last_stats["invalid_ids_stripped"] == 2
    assert questions.last_stats["questions_dropped"] == 1
    # findings and requirements were passed as data inside tags
    assert "<findings>" in backend.users[0] and "[F-001]" in backend.users[0] and "<requirements>" in backend.users[0]


async def test_ranking_prefers_more_findings_then_requirements(findings):
    backend = FakeBackend(
        main=lambda user: QuestionsAnswer(
            questions=[
                dq(["F-002"], "Вопрос только про одно критичное замечание П-3?"),
                dq(["F-001", "F-002"], "Вопрос про два критичных замечания сразу, П-1 и П-3?"),
                dq(["F-003"], "Вопрос про срок в правиле номер 4, один?"),
                dq(["F-004"], "Вопрос про единицы в правилах номер 4 и 5?"),
            ]
        )
    )
    llm.use_backend(backend)
    qs = await questions.write(findings, REQS, CFG, drafts={})
    assert [q.finding_ids for q in qs] == [["F-001", "F-002"], ["F-002"], ["F-004"], ["F-003"]]


async def test_followup_covers_missing_critical(findings):
    backend = FakeBackend(
        main=lambda user: QuestionsAnswer(questions=[dq(["F-001"], "Какое правило главнее, П-1 или П-2 в документе?")]),
        followup=lambda user: QuestionsAnswer(
            questions=[dq(["F-002", "F-001"], "Какой ожидаемый результат у проверки правила П-3?", Role.TESTER)]
        ),
    )
    llm.use_backend(backend)
    qs = await questions.write(findings, REQS, CFG, drafts={})
    assert len(backend.users) == 2
    assert "[F-002]" in backend.users[1] and "[F-001]" not in backend.users[1]
    covered = {i for q in qs for i in q.finding_ids}
    assert {"F-001", "F-002"} <= covered
    followup_q = next(q for q in qs if "П-3" in q.text)
    assert followup_q.finding_ids == ["F-002"]  # a follow-up may only claim what it was asked about
    assert questions.last_stats["followup_calls"] == 1


async def test_fallback_to_drafts_then_template(findings):
    backend = FakeBackend(main=lambda user: LLMError("down"), followup=lambda user: LLMError("down"))
    llm.use_backend(backend)
    drafts = {
        "F-001": [("Какое правило главнее: П-1 или П-2?", None)],
        "F-003": [("Какой срок имеется в виду?", Role.PRODUCT_OWNER)],
    }
    qs = await questions.write(findings, REQS, CFG, drafts=drafts)
    by_ids = {tuple(q.finding_ids): q for q in qs}
    assert by_ids[("F-001",)].text == "Какое правило главнее: П-1 или П-2?"
    assert by_ids[("F-001",)].addressee == Role.PRODUCT_OWNER  # default for set_consistent
    t = by_ids[("F-002",)]
    assert "«Правило номер 3.»" in t.text and "нельзя проверить П-3" in t.text
    assert t.blocks
    assert ("F-003",) in by_ids  # major with a draft kept when the writer failed outright
    assert ("F-004",) not in by_ids and ("F-005",) not in by_ids
    assert questions.last_stats["fallback_drafts"] == 2
    assert questions.last_stats["fallback_templates"] == 1
    assert [q.priority for q in qs] == list(range(1, len(qs) + 1))


async def test_duplicates_merged(findings):
    backend = FakeBackend(
        main=lambda user: QuestionsAnswer(
            questions=[
                dq(["F-001"], "Какое правило главнее: «Правило номер 1» или «Правило номер 2»?"),
                dq(["F-002"], "Какое правило главнее: «Правило номер 1» или «Правило номер 2» ?"),
            ]
        )
    )
    llm.use_backend(backend)
    qs = await questions.write(findings, REQS, CFG, drafts={})
    assert len(qs) == 1
    assert qs[0].finding_ids == ["F-001", "F-002"]
    assert questions.last_stats["duplicates_merged"] == 1


async def test_batching_for_many_findings():
    many = [finding(f"F-{i:03d}", Severity.MAJOR, [f"R-00{1 + i % 5}"]) for i in range(1, 41)]

    def main(user):
        ids = [line[1:6] for line in user.splitlines() if line.startswith("[F-")]
        return QuestionsAnswer(questions=[dq(ids[:1], "Один и тот же вопрос про все правила документа?")])

    backend = FakeBackend(main=main)
    llm.use_backend(backend)
    qs = await questions.write(many, REQS, CFG, drafts={})
    assert len(backend.users) == 2
    assert len(qs) == 1  # identical questions from two batches merged
    assert len(qs[0].finding_ids) == 2


async def test_no_findings_no_call():
    backend = FakeBackend(main=lambda user: QuestionsAnswer(questions=[]))
    llm.use_backend(backend)
    assert await questions.write([], REQS, CFG) == []
    assert backend.users == []


async def test_failed_batch_is_retried_in_halves(findings):
    def main(user):
        ids = [line[1:6] for line in user.splitlines() if line.startswith("[F-")]
        if len(ids) == 5:
            return LLMError("too long")
        return QuestionsAnswer(questions=[dq(ids, f"Вопрос про замечания {', '.join(ids)} в документе?")])

    backend = FakeBackend(main=main)
    llm.use_backend(backend)
    qs = await questions.write(findings, REQS, CFG, drafts={})
    assert len(backend.users) == 3
    assert questions.last_stats["batches_split"] == 1
    assert {i for q in qs for i in q.finding_ids} == {"F-001", "F-002", "F-003", "F-004", "F-005"}
