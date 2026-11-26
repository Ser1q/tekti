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
    RawFinding,
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
    raw_findings = [
        RawFinding(
            reviewer=ReviewerId(lens=Lens.TESTER, model="claude-haiku-4-5"),
            requirement_ids=["R-001"],
            criterion=Criterion.VERIFIABLE,
            severity=Severity.CRITICAL,
            title="RAW_ONLY_TITLE_DO_NOT_LEAK",
            explanation="RAW_ONLY_EXPLANATION_DO_NOT_LEAK",
            evidence=[Evidence(quote="RAW_ONLY_QUOTE_DO_NOT_LEAK", block_id="B-0002")],
        ),
        RawFinding(
            reviewer=ReviewerId(lens=Lens.FUNCTIONAL, model="gpt-oss-120b"),
            requirement_ids=["R-002"],
            criterion=Criterion.COMPLETE,
            severity=Severity.MINOR,
            title="RAW_ONLY_SECOND_DO_NOT_LEAK",
            explanation="RAW_ONLY_SECOND_EXP",
            evidence=[Evidence(quote="RAW_ONLY_SECOND_QUOTE", block_id="B-0003")],
        ),
    ]
    run = Run(
        passport=_make_passport(doc, dropped=["tester/kimi-k3.1: timeout"]),
        document=doc,
        requirements=reqs,
        segmentation_dropped=["Выдуманный кандидат: отсутствует в тексте"],
        facts=[],
        raw_findings=raw_findings,
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

    # Section 1: Passport details + raw_findings count (no raw leakage) + first-screen summary
    sec1 = md.split("## 1. Паспорт прогона")[1].split("## 2. Как агент нарезал требования")[0]
    assert "abc123deadbeef" in sec1
    assert "`standard`" in sec1
    assert "`claude-haiku-4-5`, `gpt-oss-120b`" in sec1
    assert "`0.1.0-test`" in sec1
    assert "- **Исходных замечаний до сведения:** 2" in sec1
    assert "- **Отпавшие рецензенты (1):**" in sec1
    assert "  - тестировщик / kimi-k3.1: timeout" in sec1
    assert "Выдуманный кандидат: отсутствует в тексте" in sec1
    for raw_sentinel in (
        "RAW_ONLY_TITLE_DO_NOT_LEAK",
        "RAW_ONLY_EXPLANATION_DO_NOT_LEAK",
        "RAW_ONLY_QUOTE_DO_NOT_LEAK",
        "RAW_ONLY_SECOND_DO_NOT_LEAK",
    ):
        assert raw_sentinel not in md

    assert "### Краткий итог на первом экране" in sec1
    assert "- **Готовность к разработке:** Нет" in sec1
    assert "- **Готовность к тестированию:** Нет" in sec1
    assert "- **Подтверждено замечаний:** 4 (критичных — 2, существенных — 1, незначительных — 1)" in sec1
    assert "**F-001**" in sec1
    assert "**F-002**" in sec1
    assert "1. **Q-001** (Архитектор):" in sec1
    assert "2. **Q-002** (Аналитик):" in sec1
    assert sec1.index("Q-001") < sec1.index("Q-002")
    # Brief location in opening block uses first quote only and omits [B-xxxx]
    f002_open_line = next(line for line in sec1.splitlines() if "**F-002**" in line)
    assert "Запрос, строка 30" in f002_open_line
    assert "Ответ, строка 42" not in f002_open_line
    assert "[B-0005]" not in f002_open_line

    # Section 2: Segmentation table ordering and no-source_id quote inclusion
    sec2 = md.split("## 2. Как агент нарезал требования")[1].split("## 3. Проверка каждого требования")[0]
    assert "| ↳ R-003 (часть R-002) | — | Данные | Отправка push при превышении порога — «При превышении порога отправляется push.» |" in sec2
    assert sec2.index("R-002") < sec2.index("↳ R-003 (часть R-002)") < sec2.index("R-004")
    # Pipe escaping in Section 2 table for R-006 (source_id=None)
    assert r"\| amount \| string \| да \| Сумма бонуса в ответе \|" in sec2

    # Section 3: Compact clean requirements (1 line) and split child without repeating parent location
    sec3 = md.split("## 3. Проверка каждого требования")[1].split("## 4. Проверка всего набора требований")[0]
    assert "### R-001 (Ф-1)" in sec3
    assert "### R-003 [часть R-002]" in sec3
    assert "### R-002" not in sec3
    assert "### R-004" not in sec3
    assert "### R-005" not in sec3
    assert "### R-006" not in sec3
    assert "- **R-002** (Ф-2): «Ф-2. Порог задаётся пользователем. При превышении порога отправляется push.» — индивидуальных замечаний нет." in sec3
    assert "- **R-004**: «Время отклика не должно превышать 1 секунду.» — индивидуальных замечаний нет." in sec3
    assert "- **R-005** (API-REQ): «| amount | integer | да | Сумма бонуса |» — индивидуальных замечаний нет (см. замечания к набору: F-002)." in sec3
    r003_block = sec3.split("### R-003 [часть R-002]")[1]
    assert "- **Расположение:**" not in r003_block
    assert sec3.index("**R-002**") < sec3.index("### R-003 [часть R-002]") < sec3.index("**R-004**")
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

    # Section 6: Summary counts & critical list (first quote location only)
    sec6 = md.split("## 6. Сводка")[1].split("## 7. Итог")[0]
    assert "**Всего требований:** 6 (верхнеуровневых / самостоятельных — 5, дочерних после разбивки — 1)" in sec6
    assert "**Всего подтверждённых замечаний:** 4" in sec6
    assert "| Критичное | 2 |" in sec6
    assert "| Существенное | 1 |" in sec6
    assert "| Незначительное | 1 |" in sec6
    assert "**F-001**" in sec6
    f002_sec6 = next(line for line in sec6.splitlines() if "**F-002**" in line)
    assert "4. API › Запрос, строка 30 [B-0005]" in f002_sec6
    assert "B-0006" not in f002_sec6

    # Section 7: Verdict (first quote location only)
    sec7 = md.split("## 7. Итог")[1]
    assert "**Можно ли начинать разработку:** Нет" in sec7
    assert "**Можно ли начинать тестирование:** Нет" in sec7
    assert verdict.rationale in sec7
    assert "**F-001**" in sec7
    f002_sec7 = next(line for line in sec7.splitlines() if "**F-002**" in line)
    assert "4. API › Запрос, строка 30 [B-0005]" in f002_sec7
    assert "B-0006" not in f002_sec7


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
    assert "- **Исходных замечаний до сведения:** 0" in md
    assert "Отпавшие рецензенты:** нет" in md
    assert "Отброшенные кандидаты нарезки:** нет" in md
    assert "### Краткий итог на первом экране" in md
    assert "- **Готовность к разработке:** Да" in md
    assert "- **Готовность к тестированию:** Да" in md
    assert "- **Подтверждено замечаний:** 0 (критичных — 0, существенных — 0, незначительных — 0)" in md
    assert "Критичных дефектов: 0" in md
    assert "Уточняющих вопросов: 0" in md
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
        evidence=[
            Evidence(quote="Начисление выполняется быстро.", block_id="B-0002"),
            Evidence(quote="Время отклика не должно превышать 1 секунду.", block_id="B-0004"),
        ],
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
    blocker_line = next(line for line in sec7.splitlines() if "**F-001**" in line)
    assert "[B-0002]" in blocker_line
    assert "B-0004" not in blocker_line


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

    # Orphan parent_id requirement is rendered in Section 2 and as a 1-line bullet in Section 3
    assert "| R-010 (часть R-999) | — | Функциональное | Сиротское дочернее требование — «Строка один строка два» |" in md
    sec3 = md.split("## 3. Проверка каждого требования")[1].split("## 4. Проверка всего набора требований")[0]
    assert "### R-010" not in sec3
    assert "- **R-010** [часть R-999]: «Строка один строка два» — индивидуальных замечаний нет (см. замечания к набору: F-001)." in sec3
    # Multiline quote in Section 4 side-by-side table uses <br> and escapes |
    assert r"| «Строка один<br>строка два» | «Пункт два с \| символом» |" in md
    # Third evidence item is rendered below the table
    assert "**Дополнительная цитата (Цитата 3 — Раздел В, строка 9 [B-0003]):**" in md
    assert "> «Пункт три документа»" in md


def test_opening_summary_caps_at_10_criticals_and_5_questions_in_priority_order() -> None:
    doc = _make_document()
    req = Requirement(
        id="R-001",
        source_id="Ф-1",
        kind=RequirementKind.FUNCTIONAL,
        summary="Расчёт бонуса",
        quote="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
        block_ids=["B-0002"],
    )
    findings = [
        Finding(
            id=f"F-{idx:03d}",
            requirement_ids=["R-001"],
            criterion=Criterion.COMPLETE,
            severity=Severity.CRITICAL,
            title=f"Критичный дефект {idx}",
            explanation="Пояснение",
            evidence=[Evidence(quote="Система рассчитывает бонус в тенге.", block_id="B-0002")],
            raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="m1")],
            families=1,
            status=FindingStatus.CONFIRMED,
        )
        for idx in range(1, 13)
    ]
    questions = [
        Question(
            id=f"Q-{idx:03d}",
            text=f"Вопрос номер {idx}?",
            addressee=Role.PRODUCT_OWNER,
            blocks="блокирует разработку",
            finding_ids=[f"F-{idx:03d}"],
            priority=10 - idx,  # Q-007 has lowest priority number (3) -> first
        )
        for idx in range(1, 8)
    ]
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=[req],
        facts=[],
        findings=findings,
        rejected=[],
        questions=questions,
        verdict=decide(findings),
        calls=[],
    )

    md = render(run)
    assert re.findall(r"^## .+$", md, flags=re.MULTILINE) == EXPECTED_H2
    sec1 = md.split("## 1. Паспорт прогона")[1].split("## 2. Как агент нарезал требования")[0]
    first_screen = sec1.split("### Краткий итог на первом экране")[1]
    assert "показаны 10 из 12" in first_screen
    assert "**F-010**" in first_screen
    assert "**F-011**" not in first_screen
    assert "**F-012**" not in first_screen
    assert "первые 5 из 7" in first_screen
    # Priorities: Q-007(3), Q-006(4), Q-005(5), Q-004(6), Q-003(7) kept; Q-002(8), Q-001(9) omitted
    assert "1. **Q-007** (Владелец продукта): Вопрос номер 7?" in first_screen
    assert "5. **Q-003** (Владелец продукта): Вопрос номер 3?" in first_screen
    assert "Q-002" not in first_screen
    assert "Q-001" not in first_screen


def test_section_3_split_children_grouping_and_quote_truncation() -> None:
    doc = _make_document()
    long_quote = (
        "* **Ф-1.** Система рассчитывает бонус в тенге и проверяет все условия начисления за календарный месяц "
        "с учётом возвратов и ограничений на переводы и снятие наличных средств."
    )
    reqs = [
        # Clean parent + clean child
        Requirement(
            id="R-001",
            source_id="Ф-1",
            kind=RequirementKind.FUNCTIONAL,
            summary="Длинное чистое требование",
            quote=long_quote,
            block_ids=["B-0002"],
        ),
        Requirement(
            id="R-002",
            source_id=None,
            kind=RequirementKind.FUNCTIONAL,
            summary="Чистый потомок",
            quote="Система рассчитывает бонус в тенге.",
            block_ids=["B-0002"],
            parent_id="R-001",
        ),
        # Dirty parent + clean child + dirty child (same location) + dirty child (different location)
        Requirement(
            id="R-010",
            source_id="Ф-2",
            kind=RequirementKind.FUNCTIONAL,
            summary="Родитель с дефектом",
            quote="Ф-2. Порог задаётся пользователем. При превышении порога отправляется push.",
            block_ids=["B-0003"],
        ),
        Requirement(
            id="R-011",
            source_id=None,
            kind=RequirementKind.FUNCTIONAL,
            summary="Чистый потомок грязного родителя",
            quote="Порог задаётся пользователем.",
            block_ids=["B-0003"],
            parent_id="R-010",
        ),
        Requirement(
            id="R-012",
            source_id=None,
            kind=RequirementKind.FUNCTIONAL,
            summary="Грязный потомок в том же блоке",
            quote="При превышении порога отправляется push.",
            block_ids=["B-0003"],
            parent_id="R-010",
        ),
        Requirement(
            id="R-013",
            source_id=None,
            kind=RequirementKind.NONFUNCTIONAL,
            summary="Грязный потомок в другом блоке",
            quote="Время отклика не должно превышать 1 секунду.",
            block_ids=["B-0004"],
            parent_id="R-010",
        ),
    ]
    findings = [
        Finding(
            id="F-001",
            requirement_ids=["R-010"],
            criterion=Criterion.SINGULAR,
            severity=Severity.MINOR,
            title="Смешаны два правила",
            explanation="Нужно разделить.",
            evidence=[Evidence(quote="Ф-2. Порог задаётся пользователем.", block_id="B-0003")],
            raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="m1")],
            families=1,
            status=FindingStatus.CONFIRMED,
        ),
        Finding(
            id="F-002",
            requirement_ids=["R-012"],
            criterion=Criterion.COMPLETE,
            severity=Severity.MAJOR,
            title="Не указан канал повтора push",
            explanation="Нет политики повторов.",
            evidence=[Evidence(quote="При превышении порога отправляется push.", block_id="B-0003")],
            raised_by=[ReviewerId(lens=Lens.DEVELOPER, model="m1")],
            families=1,
            status=FindingStatus.CONFIRMED,
        ),
        Finding(
            id="F-003",
            requirement_ids=["R-013"],
            criterion=Criterion.VERIFIABLE,
            severity=Severity.MAJOR,
            title="Не указаны процентили времени отклика",
            explanation="Нет p95/p99.",
            evidence=[Evidence(quote="Время отклика не должно превышать 1 секунду.", block_id="B-0004")],
            raised_by=[ReviewerId(lens=Lens.TESTER, model="m1")],
            families=1,
            status=FindingStatus.CONFIRMED,
        ),
    ]
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=reqs,
        facts=[],
        findings=findings,
        rejected=[],
        questions=[],
        verdict=decide(findings),
        calls=[],
    )
    md = render(run)
    sec3 = md.split("## 3. Проверка каждого требования")[1].split("## 4. Проверка всего набора требований")[0]

    # Long clean quote truncated cleanly with ellipsis
    r001_line = next(line for line in sec3.splitlines() if line.startswith("- **R-001**"))
    assert "…" in r001_line
    quoted_part = r001_line.split("«", 1)[1].rsplit("»", 1)[0]
    assert len(quoted_part) <= 120

    # Clean child nested directly under clean parent without repeating location
    assert "  - ↳ **R-002** [часть R-001]: «Система рассчитывает бонус в тенге.» — индивидуальных замечаний нет." in sec3

    # Clean child under dirty parent rendered as top-level ↳ bullet
    assert "- ↳ **R-011** [часть R-010]: «Порог задаётся пользователем.» — индивидуальных замечаний нет." in sec3

    # Dirty child in SAME block omits - **Расположение:**; dirty child in DIFFERENT block keeps it
    r012_section = sec3.split("### R-012 [часть R-010]")[1].split("### R-013 [часть R-010]")[0]
    assert "- **Расположение:**" not in r012_section
    r013_section = sec3.split("### R-013 [часть R-010]")[1]
    assert "- **Расположение:** 3. Нефункциональные требования, строка 20, стр. 2 [B-0004]" in r013_section


def test_challenge_reasoning_two_sentence_cap_and_abbreviations() -> None:
    doc = _make_document()
    req = Requirement(
        id="R-001",
        source_id="Ф-1",
        kind=RequirementKind.FUNCTIONAL,
        summary="Расчёт бонуса",
        quote="* **Ф-1.** Система рассчитывает бонус в тенге. Начисление выполняется быстро.",
        block_ids=["B-0002"],
    )
    reasoning = (
        "Согласно п. 5.2.5 и ст. 4.3.2 (см. рис. 1, табл. 2, и т.д., др., напр., ч. 1) "
        "формулировка «быстро… без числа» нарушает стандарт 29148.1. "
        "Также пп. 3 и т.п. подтверждают дефект. "
        "Третье предложение не должно отображаться в отчёте. "
        "Четвёртое тоже лишнее."
    )
    finding = Finding(
        id="F-001",
        requirement_ids=["R-001"],
        criterion=Criterion.VERIFIABLE,
        severity=Severity.CRITICAL,
        title="Размытая формулировка",
        explanation="Нельзя проверить слово «быстро».",
        evidence=[Evidence(quote="Начисление выполняется быстро.", block_id="B-0002")],
        raised_by=[ReviewerId(lens=Lens.TESTER, model="m1")],
        families=1,
        challenge=Challenge(challenger="m2", upheld=True, reasoning=reasoning),
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
    assert "п. 5.2.5" in md
    assert "ст. 4.3.2" in md
    assert "29148.1." in md
    assert "Также пп. 3 и т.п. подтверждают дефект." in md
    assert "Третье предложение" not in md
    assert "Четвёртое" not in md


def test_dropped_reviewers_truncated_to_120_chars_per_line() -> None:
    doc = _make_document()
    long_err = "MaxTokensReachedException: " + ("x" * 200)
    run = Run(
        passport=_make_passport(
            doc,
            dropped=[
                f"functional/us.moonshotai.kimi-k3.1: {long_err}",
                "перекрёстная проверка: не выполнено вызовов — 2",
            ],
        ),
        document=doc,
        requirements=[],
        facts=[],
        findings=[],
        rejected=[],
        questions=[],
        verdict=decide([]),
        calls=[],
    )

    md = render(run)
    sec1 = md.split("## 1. Паспорт прогона")[1].split("## 2. Как агент нарезал требования")[0]
    assert "- **Отпавшие рецензенты (2):**" in sec1
    dropped_line = next(line for line in sec1.splitlines() if "us.moonshotai.kimi-k3.1" in line)
    assert dropped_line.startswith("  - функциональный анализ / us.moonshotai.kimi-k3.1: MaxTokensReachedException: ")
    err_shown = dropped_line.split(": ", 1)[1]
    assert len(err_shown) <= 120
    assert err_shown.endswith("…")
    assert "  - перекрёстная проверка: не выполнено вызовов — 2" in sec1


def test_cyrillic_r_normalized_in_prose_never_in_quotes() -> None:
    block = Block(
        id="B-0001",
        kind=BlockKind.PARAGRAPH,
        text="Р-005. Система обязана хранить сумму в тенге и Р-005 не менять.",
        section_path=["Требования"],
        line_start=1,
        line_end=1,
    )
    doc = Document(
        name="cyr.md",
        source_path="/tmp/cyr.md",
        format="markdown",
        sha256="cafebabe",
        raw_text=block.text,
        blocks=[block],
    )
    req = Requirement(
        id="R-005",
        source_id="Р-005",
        kind=RequirementKind.FUNCTIONAL,
        summary="Хранение суммы в тенге",
        quote="Р-005. Система обязана хранить сумму в тенге и Р-005 не менять.",
        block_ids=["B-0001"],
    )
    finding = Finding(
        id="F-001",
        requirement_ids=["R-005"],
        criterion=Criterion.COMPLETE,
        severity=Severity.CRITICAL,
        title="В Р-005 не указана точность округления (см. р–012)",
        explanation="Требование Р-005 и связанный пункт р — 012 не описывают округление.",
        evidence=[Evidence(quote="Р-005. Система обязана хранить сумму в тенге и Р-005 не менять.", block_id="B-0001")],
        raised_by=[ReviewerId(lens=Lens.FUNCTIONAL, model="m1")],
        families=1,
        challenge=Challenge(
            challenger="m2",
            upheld=True,
            reasoning="В Р-005 действительно нет правил округления.",
        ),
        status=FindingStatus.CONFIRMED,
    )
    question = Question(
        id="Q-001",
        text="Какое округление действует для Р-005?",
        addressee=Role.PRODUCT_OWNER,
        blocks="реализация Р-005",
        finding_ids=["F-001"],
        priority=1,
        options=["по Р-005 до целых", "до тиынов"],
    )
    run = Run(
        passport=_make_passport(doc),
        document=doc,
        requirements=[req],
        facts=[],
        findings=[finding],
        rejected=[],
        questions=[question],
        verdict=decide([finding]),
        calls=[],
    )

    md = render(run)
    # All prose references normalized to Latin R-005 / R-012
    assert "В R-005 не указана точность округления (см. R-012)" in md
    assert "Требование R-005 и связанный пункт R-012 не описывают округление." in md
    assert "В R-005 действительно нет правил округления." in md
    assert "Какое округление действует для R-005?" in md
    assert "реализация R-005" in md
    assert "«по R-005 до целых»; «до тиынов»" in md
    # Verbatim quote preserved untouched with Cyrillic Р-005
    assert "> «Р-005. Система обязана хранить сумму в тенге и Р-005 не менять.»" in md


