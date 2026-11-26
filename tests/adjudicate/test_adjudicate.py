"""Adjudication tests: offline, with a fake model backend and a tiny hand-made document."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tekti import adjudicate, llm
from tekti.adjudicate import merge, quotes_overlap, vote_severity
from tekti.contracts import (
    Block,
    BlockKind,
    Criterion,
    Document,
    Evidence,
    FindingStatus,
    Lens,
    RawFinding,
    Requirement,
    RequirementKind,
    ReviewerId,
    Role,
    Severity,
)
from tekti.llm import LLMError, ModelSpec
from tekti.prompts.challenge import ChallengeAnswer, FinalDedupAnswer, FinalGroup, MergeAnswer, PairAnswer

LINES = [
    (BlockKind.HEADING, "# Бонусная программа", []),
    (BlockKind.HEADING, "## Правила", ["Бонусная программа"]),
    (
        BlockKind.LIST_ITEM,
        "* **П-1.** Бонус начисляется в размере 1% от суммы покупки по дате оплаты.",
        ["Бонусная программа", "Правила"],
    ),
    (
        BlockKind.LIST_ITEM,
        "* **П-2.** Клиент получает не более одного уведомления о бонусе в день.",
        ["Бонусная программа", "Правила"],
    ),
    (
        BlockKind.LIST_ITEM,
        "* **П-3.** При смене тарифа уведомление отправляется сразу независимо от предыдущих уведомлений.",
        ["Бонусная программа", "Правила"],
    ),
    (BlockKind.HEADING, "## Нефункциональные требования", ["Бонусная программа"]),
    (
        BlockKind.PARAGRAPH,
        "Система должна работать быстро и удобно.",
        ["Бонусная программа", "Нефункциональные требования"],
    ),
    (BlockKind.HEADING, "## Критерии приёмки", ["Бонусная программа"]),
    (
        BlockKind.TABLE_ROW,
        "| Покупка 31 марта в 23:50 | Оплата 1 апреля | Бонус учтён в апреле |",
        ["Бонусная программа", "Критерии приёмки"],
    ),
    (BlockKind.HEADING, "## Требования к тестированию", ["Бонусная программа"]),
    (BlockKind.PARAGRAPH, "заполняются тестировщиками", ["Бонусная программа", "Требования к тестированию"]),
]


def make_doc() -> Document:
    blocks = [
        Block(id=f"B-{i + 1:04d}", kind=k, text=t, section_path=p, line_start=i * 2 + 1, line_end=i * 2 + 1)
        for i, (k, t, p) in enumerate(LINES)
    ]
    raw = "\n\n".join(t for _, t, _ in LINES) + "\n"
    return Document(name="bonus.md", source_path="bonus.md", format="markdown", sha256="0", raw_text=raw, blocks=blocks)


def make_reqs() -> list[Requirement]:
    return [
        Requirement(
            id="R-001",
            source_id="П-1",
            kind=RequirementKind.FUNCTIONAL,
            summary="Начисление 1%",
            quote="Бонус начисляется в размере 1% от суммы покупки по дате оплаты.",
            block_ids=["B-0003"],
        ),
        Requirement(
            id="R-002",
            source_id="П-2",
            kind=RequirementKind.FUNCTIONAL,
            summary="Одно уведомление в день",
            quote="Клиент получает не более одного уведомления о бонусе в день.",
            block_ids=["B-0004"],
        ),
        Requirement(
            id="R-003",
            source_id="П-3",
            kind=RequirementKind.FUNCTIONAL,
            summary="Уведомление при смене тарифа",
            quote="При смене тарифа уведомление отправляется сразу",
            block_ids=["B-0005"],
        ),
        Requirement(
            id="R-004",
            kind=RequirementKind.NONFUNCTIONAL,
            summary="Скорость",
            quote="Система должна работать быстро и удобно.",
            block_ids=["B-0007"],
        ),
        Requirement(
            id="R-005",
            kind=RequirementKind.ACCEPTANCE,
            summary="Граница месяца",
            quote="Бонус учтён в апреле",
            block_ids=["B-0009"],
        ),
    ]


PANEL = [
    ModelSpec(family="anthropic", model_id="claude"),
    ModelSpec(family="openai", model_id="gpt"),
    ModelSpec(family="xai", model_id="grok"),
]


def make_cfg(panel=None):
    panel = panel or PANEL
    return SimpleNamespace(panel=panel, question_writer=panel[0], max_parallel=3, max_output_tokens=2000)


class FakeBackend:
    """Routes by schema; records (model_id, schema name, user) for every call."""

    def __init__(self, challenge=None, merge=None, pair=None, final=None):
        self.calls: list[tuple[str, str, str]] = []
        self.challenge = challenge or (
            lambda model, user: ChallengeAnswer(reasoning="Дефект подтверждён.", upheld=True)
        )
        self.merge = merge or (lambda model, user: MergeAnswer(groups=[]))
        self.pair = pair or (lambda model, user: PairAnswer(same=[]))
        self.final = final or (lambda model, user: FinalDedupAnswer(groups=[]))

    async def complete(self, model, *, system, user, schema, max_tokens):
        self.calls.append((model.model_id, schema.__name__, user))
        handler = {ChallengeAnswer: self.challenge, PairAnswer: self.pair, FinalDedupAnswer: self.final}.get(
            schema, self.merge
        )
        out = handler(model, user)
        if isinstance(out, Exception):
            raise out
        return out, 100, 20

    def by_schema(self, name: str) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[1] == name]

    def without_final(self) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[1] != "FinalDedupAnswer"]


@pytest.fixture
def backend():
    b = FakeBackend()
    llm.use_backend(b)
    return b


def rf(
    model,
    crit,
    sev,
    reqs,
    quotes,
    title="Замечание",
    expl="Пояснение.",
    lens=Lens.FUNCTIONAL,
    question=None,
    addressee=None,
):
    return RawFinding(
        reviewer=ReviewerId(lens=None if model == "rules" else lens, model=model),
        requirement_ids=reqs,
        criterion=crit,
        severity=sev,
        title=title,
        explanation=expl,
        evidence=[Evidence(quote=q) for q in quotes],
        question=question,
        addressee=addressee,
    )


Q_P2 = "Клиент получает не более одного уведомления о бонусе в день."
Q_P3 = "уведомление отправляется сразу независимо от предыдущих уведомлений"
Q_FAST = "Система должна работать быстро и удобно."


# --------------------------------------------------------------------------- unit helpers


def test_vote_severity_tie_goes_higher():
    assert vote_severity([Severity.MAJOR, Severity.MINOR]) == Severity.MAJOR
    assert vote_severity([Severity.MINOR, Severity.MINOR, Severity.CRITICAL]) == Severity.MINOR
    assert vote_severity([Severity.CRITICAL, Severity.MINOR]) == Severity.CRITICAL


def test_quotes_overlap():
    assert quotes_overlap("быстро и удобно", "Система должна работать быстро и удобно.")
    assert quotes_overlap(
        "aaaa Клиент получает не более одного уведомления", "Клиент получает не более одного уведомления bbb"
    )
    assert not quotes_overlap("Клиент получает", "Бонус начисляется")


# --------------------------------------------------------------------------- quote check


async def test_quote_check_rejects_and_drops_evidence(backend):
    raw = [
        rf("rules", Criterion.SET_COMPLETE, Severity.MAJOR, [], ["заполняются тестировщиками"]),
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["Система должна отвечать за 2 секунды"]),
        # one good quote, one invented: evidence is trimmed, finding survives
        rf("rules", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-004"], [Q_FAST, "работать мгновенно"]),
    ]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert [f.reject_reason for f in rejected] == ["quote not found"]
    assert rejected[0].id == "X-001"
    assert rejected[0].status == FindingStatus.REJECTED
    assert len(confirmed) == 2
    fast = next(f for f in confirmed if f.requirement_ids == ["R-004"])
    assert [e.quote for e in fast.evidence] == [Q_FAST]
    assert fast.evidence[0].block_id == "B-0007"
    assert adjudicate.last_stats["evidence_dropped"] == 2
    assert backend.without_final() == []  # everything raised by rules: no challenge


async def test_quote_without_markup_is_recovered_exactly(backend):
    raw = [rf("rules", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-001"], ["«П-1. Бонус начисляется»"])]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed[0].evidence[0].quote == "П-1.** Бонус начисляется"
    assert confirmed[0].evidence[0].block_id == "B-0003"


# --------------------------------------------------------------------------- bucketing and merging


async def test_folding_merges_correct_with_set_consistent_and_confirms_by_two_families(backend):
    raw = [
        rf("claude", Criterion.CORRECT, Severity.CRITICAL, ["R-002", "R-003"], [Q_P2, Q_P3], title="Коротко"),
        rf(
            "gpt",
            Criterion.SET_CONSISTENT,
            Severity.CRITICAL,
            ["R-003", "R-002"],
            [Q_P3, Q_P2],
            title="П-2 и П-3 противоречат друг другу",
            expl="Длинное пояснение про лимит уведомлений и смену тарифа.",
        ),
    ]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert rejected == []
    assert len(confirmed) == 1
    f = confirmed[0]
    assert f.criterion == Criterion.SET_CONSISTENT
    assert f.families == 2
    assert f.challenge is None
    assert f.title == "П-2 и П-3 противоречат друг другу"  # equal evidence: longer explanation wins
    assert f.requirement_ids == ["R-002", "R-003"]
    assert len(f.evidence) == 2
    assert backend.without_final() == []  # two families: no challenge, overlapping quotes: no merge call


async def test_complete_folds_to_set_complete_without_requirements(backend):
    raw = [
        rf("claude", Criterion.COMPLETE, Severity.MAJOR, [], ["заполняются тестировщиками"]),
        rf(
            "grok",
            Criterion.SET_COMPLETE,
            Severity.MAJOR,
            [],
            ["## Требования к тестированию\n\nзаполняются тестировщиками"],
        ),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(confirmed) == 1
    assert confirmed[0].criterion == Criterion.SET_COMPLETE
    assert confirmed[0].families == 2
    assert len(confirmed[0].evidence) == 1  # the shorter quote is inside the longer one


async def test_same_bucket_without_overlap_uses_one_merge_call(backend):
    backend.merge = lambda model, user: MergeAnswer(groups=[[0, 1], [5, 0]])  # second group invalid
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["быстро"]),
        rf("gpt", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["удобно"]),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(backend.by_schema("MergeAnswer")) == 1
    assert len(confirmed) == 1 and confirmed[0].families == 2


async def test_merge_call_failure_keeps_findings_distinct(backend):
    backend.merge = lambda model, user: LLMError("boom")
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["быстро"]),
        rf("gpt", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["удобно"]),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(confirmed) == 2
    assert adjudicate.last_stats["merge_calls_failed"] == 1
    assert len(backend.by_schema("ChallengeAnswer")) == 2


async def test_cross_bucket_different_criteria_asks_the_model(backend):
    backend.pair = lambda model, user: PairAnswer(same=[0, 7])
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST]),
        rf("gpt", Criterion.VERIFIABLE, Severity.CRITICAL, ["R-004"], ["Система должна работать быстро и удобно"]),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(confirmed) == 1
    assert confirmed[0].families == 2
    assert confirmed[0].severity == Severity.CRITICAL  # 1-1 tie goes to the higher
    assert adjudicate.last_stats["cross_bucket_merges"] == 1
    assert len(backend.by_schema("PairAnswer")) == 1


async def test_cross_bucket_pair_call_failure_falls_back_to_related_criteria(backend):
    backend.pair = lambda model, user: LLMError("down")
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST]),
        rf("gpt", Criterion.VERIFIABLE, Severity.MAJOR, ["R-004"], ["Система должна работать быстро и удобно"]),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(confirmed) == 1
    assert adjudicate.last_stats["pair_calls_failed"] == 1


async def test_cross_bucket_keeps_conflict_and_ambiguity_apart(backend):
    q1 = "Бонус начисляется в размере 1% от суммы покупки по дате оплаты."
    raw = [
        rf("claude", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-001", "R-005"], [q1, "Бонус учтён в апреле"]),
        rf("gpt", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-001"], [q1]),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert len(confirmed) == 2
    assert backend.by_schema("PairAnswer") == []  # different classes and requirement sets: not even asked


# --------------------------------------------------------------------------- challenge


async def test_single_family_finding_challenged_by_other_family(backend):
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    calls = backend.by_schema("ChallengeAnswer")
    assert [c[0] for c in calls] == ["gpt"]
    user = calls[0][2]
    assert "<finding>" in user and "<excerpts>" in user
    assert "Однозначность" in user  # criterion with its Russian name and definition
    assert "Нефункциональные требования" in user  # section path of the quoted block
    assert "## Критерии приёмки" in user  # neighbour block
    assert confirmed[0].challenge is not None and confirmed[0].challenge.challenger == "gpt"
    assert confirmed[0].families == 1
    assert rejected == []


async def test_custom_challenger_for_is_used(backend):
    raw = [rf("gpt", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    seen = {}

    def pick(cfg, fams):
        seen["fams"] = fams
        return PANEL[2]

    await merge(make_doc(), make_reqs(), raw, make_cfg(), challenger_for=pick)
    assert seen["fams"] == {"openai"}
    assert backend.by_schema("ChallengeAnswer")[0][0] == "grok"


async def test_challenger_rejects(backend):
    backend.challenge = lambda model, user: ChallengeAnswer(
        reasoning="Это пожелание добавить SMS, а не дефект текста.", upheld=False
    )
    raw = [
        rf("claude", Criterion.SET_COMPLETE, Severity.MAJOR, [], ["# Бонусная программа"], title="Нет SMS"),
        rf("rules", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-004"], ["быстро"]),
    ]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert [f.title for f in rejected] == ["Нет SMS"]
    r = rejected[0]
    assert r.id == "X-001"
    assert r.status == FindingStatus.REJECTED
    assert r.challenge is not None and r.challenge.upheld is False
    assert "пожелание" in r.reject_reason
    assert [f.id for f in confirmed] == ["F-001"]


async def test_challenger_changes_severity(backend):
    backend.challenge = lambda model, user: ChallengeAnswer(reasoning="Мелочь.", upheld=True, severity=Severity.MINOR)
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.CRITICAL, ["R-004"], [Q_FAST])]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed[0].severity == Severity.MINOR
    assert confirmed[0].challenge.severity == Severity.MINOR


async def test_failed_challenge_keeps_finding(backend):
    backend.challenge = lambda model, user: LLMError("timeout")
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert rejected == []
    assert confirmed[0].challenge is None
    assert confirmed[0].severity == Severity.MAJOR
    assert adjudicate.last_stats["challenges_failed"] == 1


async def test_fast_mode_same_family_challenges(backend):
    panel = [ModelSpec(family="anthropic", model_id="claude")]
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg(panel))
    assert backend.by_schema("ChallengeAnswer")[0][0] == "claude"
    assert adjudicate.last_stats["same_family_challenges"] == 1
    assert confirmed[0].challenge is not None


# --------------------------------------------------------------------------- votes, order, drafts


async def test_severity_vote_in_merge(backend):
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-004"], [Q_FAST], lens=Lens.FUNCTIONAL),
        rf("gpt", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-004"], [Q_FAST], lens=Lens.TESTER),
        rf("grok", Criterion.UNAMBIGUOUS, Severity.CRITICAL, ["R-004"], [Q_FAST], lens=Lens.DEVELOPER),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed[0].severity == Severity.MINOR
    assert confirmed[0].families == 3
    assert len(confirmed[0].raised_by) == 3


async def test_ordering_and_numbering(backend):
    raw = [
        # minor, rules (1 family), position late
        rf("rules", Criterion.SET_COMPLETE, Severity.MINOR, [], ["заполняются тестировщиками"], title="minor"),
        # major, one family + rules = 2 families, position: R-004
        rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], ["быстро"], title="major-2fam"),
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST], title="major-2fam"),
        # major, rules only, earlier position (P-1)
        rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-001"], ["по дате оплаты"], title="major-1fam-early"),
        # critical, later position than P-1
        rf("rules", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-002", "R-003"], [Q_P2, Q_P3], title="crit-b"),
        rf(
            "rules",
            Criterion.SET_CONSISTENT,
            Severity.CRITICAL,
            ["R-001", "R-005"],
            ["по дате оплаты.", "Бонус учтён в апреле"],
            title="crit-a",
        ),
    ]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert [f.title for f in confirmed] == ["crit-a", "crit-b", "major-2fam", "major-1fam-early", "minor"]
    assert [f.id for f in confirmed] == ["F-001", "F-002", "F-003", "F-004", "F-005"]


async def test_question_drafts_are_collected(backend):
    raw = [
        rf(
            "rules",
            Criterion.UNAMBIGUOUS,
            Severity.MAJOR,
            ["R-004"],
            [Q_FAST],
            question="Какое время ответа?",
            addressee=Role.PRODUCT_OWNER,
        ),
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST], question="Какое  время ответа?"),
        rf(
            "gpt",
            Criterion.UNAMBIGUOUS,
            Severity.MAJOR,
            ["R-004"],
            [Q_FAST],
            question="Что значит «удобно»?",
            addressee=Role.CUSTOMER,
        ),
    ]
    await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert adjudicate.last_question_drafts["F-001"] == [
        ("Какое время ответа?", Role.PRODUCT_OWNER),
        ("Что значит «удобно»?", Role.CUSTOMER),
    ]


async def test_unknown_requirement_ids_dropped(backend):
    raw = [rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004", "R-999"], [Q_FAST])]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed[0].requirement_ids == ["R-004"]
    assert adjudicate.last_stats["unknown_requirement_ids"] == 1


# --------------------------------------------------------------------------- requirement id normalisation

P4_ANY = "После смены тарифа следующее начисление вызывает уведомление независимо от истории."
NORM_LINES = [
    (BlockKind.HEADING, "## Правила", []),
    (
        BlockKind.LIST_ITEM,
        "* **П-3.** Когда бонус начислен, клиент получает уведомление. Клиенту приходит не более одного уведомления в месяц.",
        ["Правила"],
    ),
    (
        BlockKind.LIST_ITEM,
        "* **П-4.** При смене тарифа баланс сохраняется. " + P4_ANY,
        ["Правила"],
    ),
    (BlockKind.HEADING, "## API", []),
    (BlockKind.TABLE_ROW, "| **Description** | Сохранение тарифа клиента |", ["API"]),
    (BlockKind.TABLE_ROW, "| **Method** | GET |", ["API"]),
    (BlockKind.LIST_ITEM, "1. Сохранить тариф и вернуть результат сохранения.", ["API"]),
    (BlockKind.LIST_ITEM, "2. Вернуть код 200 при успехе.", ["API"]),
    (BlockKind.LIST_ITEM, "3. Вернуть код 400 при ошибке.", ["API"]),
]

P3_CAP = "Клиенту приходит не более одного уведомления в месяц."
DESC = "**Description** | Сохранение тарифа клиента"
METHOD = "**Method** | GET"
STEP1 = "1. Сохранить тариф и вернуть результат сохранения."


def norm_doc() -> Document:
    blocks = [
        Block(id=f"B-{i + 1:04d}", kind=k, text=t, section_path=p, line_start=i + 1, line_end=i + 1)
        for i, (k, t, p) in enumerate(NORM_LINES)
    ]
    raw = "\n\n".join(t for _, t, _ in NORM_LINES) + "\n"
    return Document(name="n.md", source_path="n.md", format="markdown", sha256="0", raw_text=raw, blocks=blocks)


def norm_reqs() -> list[Requirement]:
    def r(rid, quote, parent=None, source=None):
        return Requirement(
            id=rid,
            source_id=source,
            kind=RequirementKind.FUNCTIONAL,
            summary=rid,
            quote=quote,
            block_ids=[],
            parent_id=parent,
        )

    return [
        r("R-001", "Когда бонус начислен, клиент получает уведомление. " + P3_CAP, source="П-3"),
        r("R-002", "Когда бонус начислен, клиент получает уведомление.", parent="R-001"),
        r("R-003", P3_CAP, parent="R-001"),
        r("R-004", "При смене тарифа баланс сохраняется. " + P4_ANY, source="П-4"),
        r("R-005", "При смене тарифа баланс сохраняется.", parent="R-004"),
        r("R-006", P4_ANY, parent="R-004"),
        r("R-007", DESC),
        r("R-008", METHOD),
        r("R-009", STEP1),
        r("R-010", "2. Вернуть код 200 при успехе."),
        r("R-011", "3. Вернуть код 400 при ошибке."),
    ]


async def run_norm(raw):
    return await merge(norm_doc(), norm_reqs(), raw, make_cfg())


async def test_parent_and_child_collapse_to_child(backend):
    raw = [rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-001", "R-003"], ["не более одного уведомления"])]
    confirmed, _ = await run_norm(raw)
    assert confirmed[0].requirement_ids == ["R-003"]
    assert not confirmed[0].is_set_level
    assert confirmed[0].evidence[0].requirement_id == "R-003"


async def test_individual_finding_with_many_ids_gets_the_one_its_evidence_sits_in(backend):
    raw = [rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-002", "R-005", "R-007", "R-010"], [METHOD])]
    confirmed, _ = await run_norm(raw)
    assert confirmed[0].requirement_ids == ["R-008"]


async def test_individual_finding_with_evidence_in_two_requirements_keeps_both(backend):
    raw = [rf("rules", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-008"], [METHOD, STEP1])]
    confirmed, _ = await run_norm(raw)
    assert confirmed[0].requirement_ids == ["R-008", "R-009"]


async def test_conflict_keeps_one_id_per_side(backend):
    raw = [
        rf("rules", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-001", "R-004", "R-010"], [P3_CAP, P4_ANY]),
    ]
    confirmed, _ = await run_norm(raw)
    assert confirmed[0].requirement_ids == ["R-003", "R-006"]


async def test_reviewer_ids_kept_when_no_evidence_sits_in_a_requirement(backend):
    raw = [rf("rules", Criterion.SET_COMPLETE, Severity.MAJOR, ["R-001", "R-002"], ["## API"])]
    confirmed, _ = await run_norm(raw)
    assert confirmed[0].requirement_ids == ["R-002"]  # collapsed parent -> child, not dropped


async def test_parent_level_and_child_level_conflict_merge(backend):
    # Same conflict cited once at parent level and once at child level (two families).
    raw = [
        rf("claude", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-001", "R-004"], [P3_CAP, P4_ANY], title="A"),
        rf("gpt", Criterion.CORRECT, Severity.CRITICAL, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="B"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 1
    assert confirmed[0].requirement_ids == ["R-003", "R-006"]
    assert confirmed[0].families == 2
    assert backend.without_final() == []


async def test_cross_class_needs_equal_requirements_and_mutual_evidence(backend):
    # "GET used to save state": once as ambiguity of one row, once as a conflict across rows.
    # Different classes and different requirement sets: kept apart even if the model says same.
    backend.pair = lambda model, user: PairAnswer(same=list(range(10)))
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-007"], [DESC, METHOD], title="GET для сохранения"),
        rf(
            "gpt",
            Criterion.SET_CONSISTENT,
            Severity.CRITICAL,
            ["R-007", "R-008", "R-009", "R-010", "R-011", "R-001"],
            ["Сохранение тарифа клиента", METHOD, STEP1],
            title="GET меняет состояние",
        ),
    ]
    confirmed, _ = await run_norm(raw)
    assert backend.by_schema("PairAnswer") == []
    assert len(confirmed) == 2


async def test_cross_class_merge_takes_the_more_severe_member(backend):
    # Same ids, same quotes, different classes, model says same: merged, and the merge never
    # lowers what was reported.
    backend.pair = lambda model, user: PairAnswer(same=[0])
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-008"], [METHOD], title="Неясный метод"),
        rf("grok", Criterion.UNAMBIGUOUS, Severity.MINOR, ["R-008"], [METHOD], title="Неясный метод 2"),
        rf("gpt", Criterion.CORRECT, Severity.CRITICAL, ["R-008"], [METHOD], title="GET меняет состояние"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(backend.by_schema("PairAnswer")) == 1
    assert len(confirmed) == 1
    f = confirmed[0]
    assert (f.criterion, f.severity, f.title) == (Criterion.CORRECT, Severity.CRITICAL, "GET меняет состояние")
    assert f.families == 3 and len(f.raised_by) == 3


async def test_conflict_is_not_absorbed_by_gap_findings_on_one_side(backend):
    # Regression (board-3): a two-family critical conflict and three major "not defined" findings
    # quoting one of its sentences; the fake grouping model says "same" to everything.
    backend.pair = lambda model, user: PairAnswer(same=list(range(50)))
    backend.merge = lambda model, user: MergeAnswer(groups=[list(range(10))])
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-001", "R-004"], [P3_CAP, P4_ANY], title="Конфликт"),
        rf(
            "claude",
            Criterion.SET_CONSISTENT,
            Severity.CRITICAL,
            ["R-003", "R-006"],
            [P3_CAP, P4_ANY],
            lens=Lens.TESTER,
            title="Противоречие П-3 и П-4",
        ),
        rf("grok", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP], title="Не определено хранение флага"),
        rf("claude", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP], title="Не указан сброс счётчика"),
        rf(
            "gpt",
            Criterion.COMPLETE,
            Severity.MAJOR,
            ["R-001"],
            ["не более одного уведомления в месяц"],
            lens=Lens.DEVELOPER,
            title="Не определён календарный месяц",
        ),
    ]
    confirmed, _ = await run_norm(raw)
    conflict = [f for f in confirmed if f.criterion == Criterion.SET_CONSISTENT]
    assert len(conflict) == 1
    c = conflict[0]
    assert c.severity == Severity.CRITICAL
    assert c.requirement_ids == ["R-003", "R-006"]
    assert c.families >= 2
    assert c.id == "F-001"
    gaps = [f for f in confirmed if f.criterion == Criterion.COMPLETE]
    assert gaps and all(f.requirement_ids == ["R-003"] for f in gaps)


async def test_cap_on_requirement_ids(backend):
    raw = [
        rf("rules", Criterion.SET_COMPLETE, Severity.MAJOR, ["R-002", "R-005", "R-007", "R-008", "R-010"], ["## API"])
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed[0].requirement_ids) == 4


async def test_individual_merge_keeps_majority_requirements(backend):
    raw = [
        rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-003"], [P3_CAP]),
        rf("gpt", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-003", "R-008"], [P3_CAP, METHOD]),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 1
    assert confirmed[0].requirement_ids == ["R-003"]
    assert len(confirmed[0].evidence) == 2  # the context quote stays as evidence


async def test_pair_merges_do_not_chain(backend):
    # Same class (gap). A~B and B~C judged same, A~C never compared and about unrelated
    # requirements: only one of the merges happens.
    q_short = "не более одного уведомления"
    backend.pair = lambda model, user: PairAnswer(same=list(range(10)))
    raw = [
        rf("claude", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [q_short], title="A"),
        rf("gpt", Criterion.SET_COMPLETE, Severity.MAJOR, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="B"),
        rf("grok", Criterion.COMPLETE, Severity.MAJOR, ["R-006"], [P4_ANY], title="C"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 2
    assert adjudicate.last_stats["pairs_blocked_chain"] == 1
    assert len(adjudicate.last_merge_log) == 1


# --------------------------------------------------------------------------- final dedup pass


def _final_all(model, user):
    n = sum(1 for line in user.splitlines() if line.startswith("["))
    return FinalDedupAnswer(groups=[FinalGroup(indices=list(range(n)), reason="одна правка: заменить метод")])


async def test_final_pass_merges_cross_class_group_and_never_lowers(backend):
    backend.final = _final_all
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-008", "R-009"], [METHOD, STEP1], title="GET меняет"),
        rf("claude", Criterion.APPROPRIATE, Severity.MAJOR, ["R-008"], [METHOD], title="GET вместо POST"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(backend.by_schema("FinalDedupAnswer")) == 1
    assert len(confirmed) == 1
    f = confirmed[0]
    assert (f.criterion, f.severity, f.title) == (Criterion.SET_CONSISTENT, Severity.CRITICAL, "GET меняет")
    assert f.requirement_ids == ["R-008", "R-009"]
    assert f.families == 2
    assert backend.by_schema("ChallengeAnswer") == []  # combined families: no challenge needed
    assert adjudicate.last_stats["final_merges"] == 1
    assert any("final:" in line and "заменить метод" in line for line in adjudicate.last_merge_log)


async def test_final_pass_keeps_conflict_and_gap_apart_without_model_group(backend):
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="Конфликт"),
        rf("claude", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="К2"),
        rf("grok", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP], title="Не определён сброс"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(backend.by_schema("FinalDedupAnswer")) == 1
    assert sorted((f.criterion.value, f.severity.value) for f in confirmed) == [
        ("complete", "major"),
        ("set_consistent", "critical"),
    ]


async def test_final_pass_skipped_on_failure(backend):
    backend.final = lambda model, user: LLMError("down")
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-008", "R-009"], [METHOD, STEP1]),
        rf("claude", Criterion.APPROPRIATE, Severity.MAJOR, ["R-008"], [METHOD]),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 2
    assert adjudicate.last_stats["final_calls_failed"] == 1


# --------------------------------------------------------------------------- two-vote rejection


def _by_model(answers):
    def handler(model, user):
        return answers[model.model_id]

    return handler


async def test_single_rejection_then_uphold_confirms(backend):
    backend.challenge = _by_model(
        {
            "gpt": ChallengeAnswer(reasoning="Нет дефекта.", upheld=False, severity=Severity.MINOR),
            "grok": ChallengeAnswer(reasoning="Дефект есть: «быстро» не измеримо.", upheld=True),
        }
    )
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert [c[0] for c in backend.by_schema("ChallengeAnswer")] == ["gpt", "grok"]  # second family differs
    assert rejected == []
    f = confirmed[0]
    assert f.severity == Severity.MAJOR  # only one challenger proposed a lower severity
    assert f.challenge.challenger == "grok" and f.challenge.upheld
    assert "«быстро»" in f.challenge.reasoning
    assert adjudicate.last_stats["second_opinions"] == 1
    assert adjudicate.last_stats["rejections_overturned"] == 1
    assert len(adjudicate.last_overturned) == 1


async def test_overturn_lowers_severity_only_when_both_agree(backend):
    backend.challenge = _by_model(
        {
            "gpt": ChallengeAnswer(reasoning="Нет.", upheld=False, severity=Severity.MINOR),
            "grok": ChallengeAnswer(reasoning="Да, но мелочь.", upheld=True, severity=Severity.MINOR),
        }
    )
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, _ = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed[0].severity == Severity.MINOR


async def test_two_rejections_reject(backend):
    backend.challenge = _by_model(
        {
            "gpt": ChallengeAnswer(reasoning="Это пожелание.", upheld=False),
            "grok": ChallengeAnswer(reasoning="Документ этого не обещал.", upheld=False),
        }
    )
    raw = [rf("claude", Criterion.SET_COMPLETE, Severity.MAJOR, [], ["# Бонусная программа"], title="Нет SMS")]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg())
    assert confirmed == []
    r = rejected[0]
    assert "пожелание" in r.reject_reason and "не обещал" in r.reject_reason
    assert r.challenge.challenger == "gpt, grok" and not r.challenge.upheld


async def test_single_family_config_keeps_single_rejection(backend):
    backend.challenge = lambda model, user: ChallengeAnswer(reasoning="Нет дефекта.", upheld=False)
    panel = [ModelSpec(family="anthropic", model_id="claude")]
    raw = [rf("claude", Criterion.UNAMBIGUOUS, Severity.MAJOR, ["R-004"], [Q_FAST])]
    confirmed, rejected = await merge(make_doc(), make_reqs(), raw, make_cfg(panel))
    assert confirmed == [] and len(rejected) == 1
    assert len(backend.by_schema("ChallengeAnswer")) == 1
    assert adjudicate.last_stats["second_opinions"] == 0


async def test_final_conflict_plus_gap_needs_a_second_model(backend):
    # The final pass groups a conflict with an "undefined case" about one of its sentences;
    # a model of another family must agree, otherwise they stay apart.
    backend.final = _final_all
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="Конфликт"),
        rf("claude", Criterion.SET_CONSISTENT, Severity.CRITICAL, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="К2"),
        rf("grok", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP], title="Не определён сброс"),
    ]
    confirmed, _ = await run_norm(raw)
    pair_calls = backend.by_schema("PairAnswer")
    assert [c[0] for c in pair_calls] == ["gpt"]  # verifier: not the grouping model's family
    assert len(confirmed) == 2
    assert adjudicate.last_stats["final_rejected_by_verifier"] == 1

    backend.calls.clear()
    backend.pair = lambda model, user: PairAnswer(same=[0])
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 1
    assert (confirmed[0].criterion, confirmed[0].severity) == (Criterion.SET_CONSISTENT, Severity.CRITICAL)


async def test_final_merge_tie_prefers_the_conflict(backend):
    backend.final = _final_all
    backend.pair = lambda model, user: PairAnswer(same=[0])
    raw = [
        rf("gpt", Criterion.SET_CONSISTENT, Severity.MAJOR, ["R-003", "R-006"], [P3_CAP, P4_ANY], title="Конфликт"),
        rf("claude", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP, "## Правила", "## API"], title="Пробел"),
        rf("grok", Criterion.COMPLETE, Severity.MAJOR, ["R-003"], [P3_CAP], title="Пробел 2"),
    ]
    confirmed, _ = await run_norm(raw)
    assert len(confirmed) == 1
    assert (confirmed[0].criterion, confirmed[0].title) == (Criterion.SET_CONSISTENT, "Конфликт")
    assert confirmed[0].requirement_ids == ["R-003", "R-006"]
    assert confirmed[0].families == 3  # the gap had more families; the conflict still leads
