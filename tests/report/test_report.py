"""Offline unit tests for tekti.report.render."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from tekti.contracts import (
    Block,
    BlockKind,
    Challenge,
    Criterion,
    Document,
    Evidence,
    Finding,
    FindingStatus,
    Lens,
    Mode,
    Passport,
    Question,
    Requirement,
    RequirementKind,
    ReviewerId,
    Role,
    Run,
    Severity,
)
from tekti.criteria import NAME_RU, SOURCE
from tekti.report import render
from tekti.verdict import decide

EXPECTED_H2 = [
    "## 1. Паспорт прогона",
    "## 2. Как агент нарезал требования",
    "## 3. Проверка каждого требования",
    "## 4. Проверка всего набора требований",
    "## 5. Уточняющие вопросы",
    "## 6. Сводка",
    "## 7. Итог",
]


def _make_document() -> Document:
    blocks = [
        Block(
            id="B-0001",
            kind=BlockKind.HEADING,
            text="## 1. Цели",
            section_path=[],
            line_start=1,
            line_end=1,
        ),
        Block(
            id="B-0002",
            kind=BlockKind.LIST_ITEM,
            text="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
            section_path=["2. Функциональные требования"],
            line_start=10,
            line_end=11,
        ),
        Block(
            id="B-0003",
            kind=BlockKind.PARAGRAPH,
            text="Ф-2. Порог задаётся пользователем. При превышении порога отправляется push.",
            section_path=["2. Функциональные требования"],
            line_start=15,
            line_end=15,
        ),
        Block(
            id="B-0004",
            kind=BlockKind.PARAGRAPH,
            text="Время отклика не должно превышать 1 секунду.",
            section_path=["3. Нефункциональные требования"],
            line_start=20,
            line_end=20,
            page=2,
        ),
        Block(
            id="B-0005",
            kind=BlockKind.TABLE_ROW,
            text="| amount | integer | да | Сумма бонуса |",
            section_path=["4. API", "Запрос"],
            line_start=30,
            line_end=30,
        ),
        Block(
            id="B-0006",
            kind=BlockKind.TABLE_ROW,
            text="| amount | string | да | Сумма бонуса в ответе |",
            section_path=["4. API", "Ответ"],
            line_start=42,
            line_end=42,
        ),
        Block(
            id="B-0007",
            kind=BlockKind.HEADING,
            text="## 5. Обработка ошибок",
            section_path=["4. API"],
            line_start=50,
            line_end=50,
        ),
    ]
    raw_text = "\n".join(b.text for b in blocks)
    return Document(
        name="bonus-spec.md",
        source_path="/tmp/bonus-spec.md",
        format="markdown",
        sha256="abc123deadbeef",
        raw_text=raw_text,
        blocks=blocks,
    )


def _make_passport(doc: Document, *, models: list[str] | None = None, dropped: list[str] | None = None) -> Passport:
    return Passport(
        document_name=doc.name,
        document_sha256=doc.sha256,
        started_at=datetime(2026, 11, 26, 10, 30, 0, tzinfo=UTC),
        agent_version="0.1.0-test",
        mode=Mode.STANDARD,
        models=["claude-haiku-4-5", "gpt-oss-120b"] if models is None else models,
        dropped_reviewers=[] if dropped is None else dropped,
    )


def test_render_full_report_structure_counts_and_evidence() -> None:
    doc = _make_document()
    # Deliberately place child R-003 after R-004 to verify parent-child grouping.
    reqs = [
        Requirement(
            id="R-001",
            source_id="Ф-1",
            kind=RequirementKind.FUNCTIONAL,
            summary="Расчёт бонуса в тенге",
            quote="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
            block_ids=["B-0002"],
        ),
        Requirement(
            id="R-002",
            source_id="Ф-2",
            kind=RequirementKind.FUNCTIONAL,
            summary="Порог и push-уведомление",
            quote="Ф-2. Порог задаётся пользователем. При превышении порога отправляется push.",
            block_ids=["B-0003"],
        ),
        Requirement(
            id="R-004",
            source_id=None,
            kind=RequirementKind.NONFUNCTIONAL,
            summary="Время отклика до 1 секунды",
            quote="Время отклика не должно превышать 1 секунду.",
            block_ids=["B-0004"],
        ),
        Requirement(
            id="R-003",
            source_id=None,
            kind=RequirementKind.DATA,
            summary="Отправка push при превышении порога",
            quote="При превышении порога отправляется push.",
            block_ids=["B-0003"],
            parent_id="R-002",
        ),
        Requirement(
            id="R-005",
            source_id="API-REQ",
            kind=RequirementKind.API,
            summary="Поле amount в запросе",
            quote="| amount | integer | да | Сумма бонуса |",
            block_ids=["B-0005"],
        ),
        Requirement(
            id="R-006",
            source_id=None,
            kind=RequirementKind.API,
            summary="Поле amount в ответе",
            quote="| amount | string | да | Сумма бонуса в ответе |",
            block_ids=["B-0006"],
        ),
    ]
    findings = [
        Finding(
            id="F-001",
            requirement_ids=["R-001"],
            criterion=Criterion.VERIFIABLE,
            severity=Severity.CRITICAL,
            title="Размытый срок «быстро» в Ф-1",
            explanation="Нельзя написать тест на слово «быстро» без числового порога.",
            evidence=[Evidence(quote="Начисление выполняется быстро.", block_id="B-0002", requirement_id="R-001")],
            raised_by=[ReviewerId(lens=Lens.TESTER, model="claude-haiku-4-5")],
            families=1,
            challenge=Challenge(
                challenger="gpt-oss-120b",
                upheld=True,
                severity=Severity.CRITICAL,
                reasoning="В разделе нет численного норматива для «быстро».",
            ),
            status=FindingStatus.CONFIRMED,
        ),
        Finding(
            id="F-002",
            requirement_ids=["R-005", "R-006"],
            criterion=Criterion.SET_CONSISTENT,
            severity=Severity.CRITICAL,
            title="Тип поля amount расходится между запросом и ответом",
            explanation="В таблице запроса указан integer, а в таблице ответа — string.",
            evidence=[
                Evidence(
                    quote="| amount | integer | да | Сумма бонуса |",
                    block_id=None,  # test fallback quote lookup
                    requirement_id="R-005",
                ),
                Evidence(
                    quote="| amount | string | да | Сумма бонуса в ответе |",
                    block_id="B-0006",
                    requirement_id="R-006",
                ),
            ],
            raised_by=[
                ReviewerId(lens=Lens.API_CONTRACT, model="claude-haiku-4-5"),
                ReviewerId(lens=None, model="rules"),
            ],
            families=2,
            status=FindingStatus.CONFIRMED,
        ),
        Finding(
            id="F-003",
            requirement_ids=[],
            criterion=Criterion.SET_COMPLETE,
            severity=Severity.MAJOR,
            title="Раздел «Обработка ошибок» не заполнен",
            explanation="После заголовка «## 5. Обработка ошибок» отсутствуют требования.",
            evidence=[Evidence(quote="## 5. Обработка ошибок", block_id="B-0007")],
            raised_by=[ReviewerId(lens=Lens.DEVELOPER, model="gpt-oss-120b")],
            families=1,
            challenge=Challenge(
                challenger="claude-haiku-4-5",
                upheld=True,
                reasoning="Раздел действительно пуст.",
            ),
            status=FindingStatus.CONFIRMED,
        ),
        Finding(
            id="F-004",
            requirement_ids=["R-003"],
            criterion=Criterion.CONFORMING,
            severity=Severity.MINOR,
            title="У дочернего требования нет идентификатора в документе",
            explanation="Формулировка в тексте не выделена отдельным идентификатором.",
            evidence=[Evidence(quote="При превышении порога отправляется push.", block_id="B-0003")],
            raised_by=[ReviewerId(lens=None, model="rules")],
            families=1,
            status=FindingStatus.CONFIRMED,
        ),
    ]
    # Intentionally unsorted questions to test priority sorting.
    questions = [
        Question(
            id="Q-002",
            text="Какие коды ошибок и сценарии должны быть описаны в разделе «5. Обработка ошибок»?",
            addressee=Role.ANALYST,
            blocks="обработка сбоев и формат ошибок сервиса",
            finding_ids=["F-003"],
            priority=2,
            options=[],
        ),
        Question(
            id="Q-001",
            text="Какой тип данных должен использоваться для поля «amount» — integer или string?",
            addressee=Role.ARCHITECT,
            blocks="контракт API запроса и ответа для amount",
            finding_ids=["F-002"],
            priority=1,
            options=["integer", "string"],
        ),
    ]
    verdict = decide(findings)
    run = Run(
        passport=_make_passport(doc, dropped=["tester/kimi-k3.1: timeout"]),
        document=doc,
        requirements=reqs,
        segmentation_dropped=["Выдуманный кандидат: отсутствует в тексте"],
        facts=[],
        findings=findings,
        rejected=[],
        questions=questions,
        verdict=verdict,
        calls=[],
    )

    md = render(run)

    # Top title and exact 7 H2 sections in exact order
    assert md.startswith("# Отчёт о проверке требований: bonus-spec.md\n")
    h2_headings = re.findall(r"^## .+$", md, flags=re.MULTILINE)
    assert h2_headings == EXPECTED_H2

    # Section 1: Passport details
    assert "abc123deadbeef" in md
    assert "`standard`" in md
    assert "`claude-haiku-4-5`, `gpt-oss-120b`" in md
    assert "`0.1.0-test`" in md
    assert "tester/kimi-k3.1: timeout" in md
    assert "Выдуманный кандидат: отсутствует в тексте" in md

    # Section 2: Segmentation table ordering and no-source_id quote inclusion
    sec2 = md.split("## 2. Как агент нарезал требования")[1].split("## 3. Проверка каждого требования")[0]
    assert "| ↳ R-003 (часть R-002) | — | Данные | Отправка push при превышении порога — «При превышении порога отправляется push.» |" in sec2
    assert sec2.index("R-002") < sec2.index("↳ R-003 (часть R-002)") < sec2.index("R-004")
    # Pipe escaping in Section 2 table for R-006 (source_id=None)
    assert r"\| amount \| string \| да \| Сумма бонуса в ответе \|" in sec2

    # Section 3: Per-requirement checks, clean requirement line, challenge reasoning
    sec3 = md.split("## 3. Проверка каждого требования")[1].split("## 4. Проверка всего набора требований")[0]
    assert "### R-001 (Ф-1)" in sec3
    assert "### R-003 [часть R-002]" in sec3
    assert "Для требования R-004 подтверждено 0 индивидуальных замечаний." in sec3
    assert "стр. 2 [B-0004]" in sec3
    assert f"{NAME_RU[Criterion.VERIFIABLE]} ({SOURCE[Criterion.VERIFIABLE]})" in sec3
    assert "gpt-oss-120b" in sec3
    assert "В разделе нет численного норматива для «быстро»." in sec3
    assert "2. Функциональные требования, строки 10–11 [B-0002]" in sec3
    assert "правила" in sec3

    # Section 4: Set-level checks, side-by-side table for conflict, document-level gap
    sec4 = md.split("## 4. Проверка всего набора требований")[1].split("## 5. Уточняющие вопросы")[0]
    assert f"### {NAME_RU[Criterion.SET_CONSISTENT]} ({SOURCE[Criterion.SET_CONSISTENT]})" in sec4
    assert f"### {NAME_RU[Criterion.SET_COMPLETE]} ({SOURCE[Criterion.SET_COMPLETE]})" in sec4
    # Side-by-side table row with escaped pipes and both locations in headers
    assert r"| «\| amount \| integer \| да \| Сумма бонуса \|» | «\| amount \| string \| да \| Сумма бонуса в ответе \|» |" in sec4
    assert "4. API › Запрос, строка 30 [B-0005]" in sec4
    assert "4. API › Ответ, строка 42 [B-0006]" in sec4
    assert "**Затронутые требования:** весь документ" in sec4

    # Section 5: Questions ordered by priority
    sec5 = md.split("## 5. Уточняющие вопросы")[1].split("## 6. Сводка")[0]
    assert sec5.index("Q-001") < sec5.index("Q-002")
    assert "Архитектор" in sec5
    assert "Аналитик" in sec5
    assert "«integer»; «string»" in sec5
    assert "документ не сужает выбор до фиксированных вариантов" in sec5

    # Section 6: Summary counts & critical list
    sec6 = md.split("## 6. Сводка")[1].split("## 7. Итог")[0]
    assert "**Всего требований:** 6 (верхнеуровневых / самостоятельных — 5, дочерних после разбивки — 1)" in sec6
    assert "**Всего подтверждённых замечаний:** 4" in sec6
    assert "| Критичное | 2 |" in sec6
    assert "| Существенное | 1 |" in sec6
    assert "| Незначительное | 1 |" in sec6
    assert "**F-001**" in sec6
    assert "**F-002**" in sec6

    # Section 7: Verdict
    sec7 = md.split("## 7. Итог")[1]
    assert "**Можно ли начинать разработку:** Нет" in sec7
    assert "**Можно ли начинать тестирование:** Нет" in sec7
    assert verdict.rationale in sec7
    assert "**F-001**" in sec7
    assert "**F-002**" in sec7


def test_rejected_findings_never_appear_anywhere_in_report() -> None:
    doc = _make_document()
    reqs = [
        Requirement(
            id="R-001",
            source_id="Ф-1",
            kind=RequirementKind.FUNCTIONAL,
            summary="Расчёт бонуса",
            quote="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
            block_ids=["B-0002"],
        )
    ]
    confirmed = Finding(
        id="F-001",
        requirement_ids=["R-001"],
        criterion=Criterion.CONFORMING,
        severity=Severity.MINOR,
        title="Подтверждённое незначительное замечание",
        explanation="Незначительный недочёт оформления.",
        evidence=[Evidence(quote="Система рассчитывает бонус в тенге.", block_id="B-0002")],
        raised_by=[ReviewerId(model="rules")],
        families=1,
        status=FindingStatus.CONFIRMED,
    )
    rejected_in_findings = Finding(
        id="F-999",
        requirement_ids=["R-001"],
        criterion=Criterion.FEASIBLE,
        severity=Severity.CRITICAL,
        title="УТЕЧКА_ИЗ_FINDINGS_REJECTED",
        explanation="СЕКРЕТНОЕ_ПОЯСНЕНИЕ_999",
        evidence=[Evidence(quote="СЕКРЕТНАЯ_ЦИТАТА_999", block_id="B-0002")],
        raised_by=[ReviewerId(lens=Lens.DEVELOPER, model="bad-model")],
        families=1,
        challenge=Challenge(challenger="other-model", upheld=False, reasoning="СЕКРЕТНОЕ_ОПРОВЕРЖЕНИЕ_999"),
        status=FindingStatus.REJECTED,
        reject_reason="ПРИЧИНА_ОТКЛОНЕНИЯ_999",
    )
    rejected_list = [
        Finding(
            id="X-001",
            requirement_ids=["R-001"],
            criterion=Criterion.COMPLETE,
            severity=Severity.CRITICAL,
            title="ОТКЛОНЁННЫЙ_ДЕФЕКТ_X001",
            explanation="ОБЪЯСНЕНИЕ_X001",
            evidence=[Evidence(quote="ЦИТАТА_X001", block_id="B-0002")],
            raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="m1")],
            families=1,
            status=FindingStatus.REJECTED,
            reject_reason="quote not found X001",
        ),
        Finding(
            id="X-002",
            requirement_ids=[],
            criterion=Criterion.SET_COMPLETE,
            severity=Severity.MAJOR,
            title="ОТКЛОНЁННЫЙ_ДЕФЕКТ_X002",
            explanation="ОБЪЯСНЕНИЕ_X002",
            evidence=[Evidence(quote="ЦИТАТА_X002")],
            raised_by=[ReviewerId(lens=Lens.TESTER, model="m2")],
            families=1,
            status=FindingStatus.REJECTED,
            reject_reason="challenger rejected X002",
        ),
    ]
    all_findings = [confirmed, rejected_in_findings]
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=reqs,
        facts=[],
        findings=all_findings,
        rejected=rejected_list,
        questions=[],
        verdict=decide(all_findings),
        calls=[],
    )

    md = render(run)

    for forbidden in (
        "X-001",
        "X-002",
        "F-999",
        "УТЕЧКА_ИЗ_FINDINGS_REJECTED",
        "СЕКРЕТНОЕ_ПОЯСНЕНИЕ_999",
        "СЕКРЕТНАЯ_ЦИТАТА_999",
        "СЕКРЕТНОЕ_ОПРОВЕРЖЕНИЕ_999",
        "ПРИЧИНА_ОТКЛОНЕНИЯ_999",
        "ОТКЛОНЁННЫЙ_ДЕФЕКТ_X001",
        "ОБЪЯСНЕНИЕ_X001",
        "ЦИТАТА_X001",
        "quote not found X001",
        "ОТКЛОНЁННЫЙ_ДЕФЕКТ_X002",
        "ОБЪЯСНЕНИЕ_X002",
        "ЦИТАТА_X002",
        "challenger rejected X002",
    ):
        assert forbidden not in md

    assert "**Всего подтверждённых замечаний:** 1" in md
    assert "| Критичное | 0 |" in md
    assert "| Незначительное | 1 |" in md
    assert "**Можно ли начинать разработку:** Да" in md
    assert "**Можно ли начинать тестирование:** Да" in md


def test_empty_run_renders_all_sections_cleanly() -> None:
    doc = _make_document()
    run = Run(
        passport=_make_passport(doc, models=[], dropped=[]),
        document=doc,
        requirements=[],
        segmentation_dropped=[],
        facts=[],
        findings=[],
        rejected=[],
        questions=[],
        verdict=decide([]),
        calls=[],
    )

    md = render(run)
    h2_headings = re.findall(r"^## .+$", md, flags=re.MULTILINE)
    assert h2_headings == EXPECTED_H2

    assert "не использовались (детерминированный прогон)" in md
    assert "Отпавшие рецензенты:** нет" in md
    assert "Отброшенные кандидаты нарезки:** нет" in md
    assert "bonus-spec.md" in md
    assert "блоков: 7" in md
    assert "выделено 0 требований" in md
    assert "| Критичное | 0 |" in md
    assert "| Существенное | 0 |" in md
    assert "| Незначительное | 0 |" in md
    assert "**Можно ли начинать разработку:** Да" in md
    assert "**Можно ли начинать тестирование:** Да" in md


def test_dev_ready_but_testing_blocked_shows_blockers_in_section_7() -> None:
    doc = _make_document()
    req = Requirement(
        id="R-001",
        source_id="Ф-1",
        kind=RequirementKind.FUNCTIONAL,
        summary="Расчёт бонуса",
        quote="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
        block_ids=["B-0002"],
    )
    finding = Finding(
        id="F-001",
        requirement_ids=["R-001"],
        criterion=Criterion.VERIFIABLE,
        severity=Severity.MAJOR,
        title="Не указан точный лимит времени",
        explanation="Формулировка «быстро» не позволяет задать критерий прохождения теста.",
        evidence=[Evidence(quote="Начисление выполняется быстро.", block_id="B-0002")],
        raised_by=[ReviewerId(lens=Lens.TESTER, model="claude-haiku-4-5")],
        families=2,
        status=FindingStatus.CONFIRMED,
    )
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=[req],
        facts=[],
        findings=[finding],
        rejected=[],
        questions=[],
        verdict=decide([finding]),
        calls=[],
    )

    md = render(run)
    sec7 = md.split("## 7. Итог")[1]
    assert "**Можно ли начинать разработку:** Да" in sec7
    assert "**Можно ли начинать тестирование:** Нет" in sec7
    assert "Критичных дефектов для первоочередного исправления нет (0)." in sec7
    assert "**F-001**" in sec7
    assert NAME_RU[Criterion.VERIFIABLE] in sec7


def test_orphan_parent_multiline_quote_and_three_evidences_preserved() -> None:
    blocks = [
        Block(
            id="B-0001",
            kind=BlockKind.PARAGRAPH,
            text="Строка один\nстрока два",
            section_path=["Раздел А"],
            line_start=1,
            line_end=2,
        ),
        Block(
            id="B-0002",
            kind=BlockKind.PARAGRAPH,
            text="Пункт два с | символом",
            section_path=["Раздел Б"],
            line_start=5,
            line_end=5,
        ),
        Block(
            id="B-0003",
            kind=BlockKind.PARAGRAPH,
            text="Пункт три документа",
            section_path=["Раздел В"],
            line_start=9,
            line_end=9,
        ),
    ]
    doc = Document(
        name="edge.md",
        source_path="/tmp/edge.md",
        format="markdown",
        sha256="deadbeef",
        raw_text="\n".join(b.text for b in blocks),
        blocks=blocks,
    )
    orphan_req = Requirement(
        id="R-010",
        source_id=None,
        kind=RequirementKind.FUNCTIONAL,
        summary="Сиротское дочернее требование",
        quote="Строка один\nстрока два",
        block_ids=["B-0001"],
        parent_id="R-999",
    )
    set_finding = Finding(
        id="F-001",
        requirement_ids=["R-010"],
        criterion=Criterion.SET_CONSISTENT,
        severity=Severity.MAJOR,
        title="Несогласованность трёх фрагментов",
        explanation="Три фрагмента противоречат друг другу.",
        evidence=[
            Evidence(quote="Строка один\nстрока два", block_id="B-0001", requirement_id="R-010"),
            Evidence(quote="Пункт два с | символом", block_id="B-0002"),
            Evidence(quote="Пункт три документа", block_id="B-0003"),
        ],
        raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="model-a")],
        families=2,
        status=FindingStatus.CONFIRMED,
    )
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=[orphan_req],
        facts=[],
        findings=[set_finding],
        rejected=[],
        questions=[],
        verdict=decide([set_finding]),
        calls=[],
    )
    md = render(run)

    # Orphan parent_id requirement is still rendered in Section 2 and Section 3
    assert "| R-010 (часть R-999) | — | Функциональное | Сиротское дочернее требование — «Строка один строка два» |" in md
    assert "### R-010 [часть R-999]" in md
    assert "см. замечания к набору: F-001" in md
    # Multiline quote in Section 4 side-by-side table uses <br> and escapes |
    assert r"| «Строка один<br>строка два» | «Пункт два с \| символом» |" in md
    # Third evidence item is rendered below the table
    assert "**Дополнительная цитата (Цитата 3 — Раздел В, строка 9 [B-0003]):**" in md
    assert "> «Пункт три документа»" in md

