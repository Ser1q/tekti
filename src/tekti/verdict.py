"""Verdict computation (P7): confirmed findings -> development and testing readiness.

Implements "Verdict (computed in code)" in docs/review-design.md:
- Rejected findings are ignored completely.
- Development can start iff there is no confirmed critical finding.
- Testing can start iff development can start and there is no confirmed critical or
  major finding under `verifiable` or `set_validatable`.
- `fix_first` lists confirmed critical finding IDs ordered by descending `families`
  (preserving input order on ties), capped at 10.
- `rationale` is built deterministically in Russian from exact counts and finding IDs.
"""

from __future__ import annotations

from collections.abc import Sequence

from tekti.contracts import Criterion, Finding, FindingStatus, Severity, Verdict
from tekti.criteria import NAME_RU

_FIX_FIRST_CAP = 10
_TESTING_BLOCKING_CRITERIA = frozenset({Criterion.VERIFIABLE, Criterion.SET_VALIDATABLE})


def _plural_ru(count: int, one: str, few: str, many: str) -> str:
    """Select the Russian noun form for a non-negative integer."""
    mod100 = count % 100
    mod10 = count % 10
    if 11 <= mod100 <= 19:
        return many
    if mod10 == 1:
        return one
    if 2 <= mod10 <= 4:
        return few
    return many


def decide(findings: Sequence[Finding]) -> Verdict:
    """Compute the final go/no-go verdict from confirmed findings."""
    confirmed = [f for f in findings if f.status == FindingStatus.CONFIRMED]
    critical = [(idx, f) for idx, f in enumerate(confirmed) if f.severity == Severity.CRITICAL]
    major_count = sum(1 for f in confirmed if f.severity == Severity.MAJOR)
    minor_count = sum(1 for f in confirmed if f.severity == Severity.MINOR)

    major_test_blockers = [
        f
        for f in confirmed
        if f.severity == Severity.MAJOR and f.criterion in _TESTING_BLOCKING_CRITERIA
    ]
    any_test_blockers = [
        f
        for f in confirmed
        if f.severity in (Severity.CRITICAL, Severity.MAJOR)
        and f.criterion in _TESTING_BLOCKING_CRITERIA
    ]

    ready_for_development = len(critical) == 0
    ready_for_testing = ready_for_development and len(any_test_blockers) == 0

    critical_sorted = sorted(critical, key=lambda pair: (-pair[1].families, pair[0]))
    fix_first = [f.id for _, f in critical_sorted[:_FIX_FIRST_CAP]]

    verifiable_ru = NAME_RU[Criterion.VERIFIABLE]
    validatable_ru = NAME_RU[Criterion.SET_VALIDATABLE]

    if not ready_for_development:
        crit_count = len(critical)
        crit_word = _plural_ru(crit_count, "критичный дефект", "критичных дефекта", "критичных дефектов")
        fix_ids = ", ".join(fix_first)
        cap_note = f" (показаны первые {len(fix_first)} из {crit_count})" if crit_count > len(fix_first) else ""
        blocker_note = ""
        if major_test_blockers:
            blocker_count = len(major_test_blockers)
            blocker_ids = ", ".join(f.id for f in major_test_blockers)
            blocker_note = (
                f" Дополнительно тестирование заблокировано по критериям "
                f"«{verifiable_ru}» / «{validatable_ru}» "
                f"(подтверждено существенных дефектов: {blocker_count} — {blocker_ids})."
            )
        rationale = (
            f"Начинать разработку и тестирование нельзя до устранения критичных дефектов: "
            f"подтверждено {crit_count} {crit_word}. В первую очередь необходимо исправить: "
            f"{fix_ids}{cap_note}.{blocker_note}"
        )
    elif not ready_for_testing:
        blocker_count = len(major_test_blockers)
        blocker_phrase = _plural_ru(
            blocker_count,
            "подтверждённым существенным замечанием",
            "подтверждёнными существенными замечаниями",
            "подтверждёнными существенными замечаниями",
        )
        blocker_ids = ", ".join(f.id for f in major_test_blockers)
        rationale = (
            f"Разработку можно начинать: подтверждённых критичных дефектов — 0. "
            f"Тестирование заблокировано {blocker_count} {blocker_phrase} "
            f"по критериям «{verifiable_ru}» / «{validatable_ru}» ({blocker_ids})."
        )
    elif not confirmed:
        rationale = (
            "Разработку и тестирование можно начинать: подтверждённых дефектов не выявлено "
            "(критичных дефектов — 0, блокирующих замечаний по критериям "
            f"«{verifiable_ru}» и «{validatable_ru}» — 0)."
        )
    else:
        rationale = (
            "Разработку и тестирование можно начинать: подтверждённых критичных дефектов — 0, "
            f"блокирующих замечаний по критериям «{verifiable_ru}» и «{validatable_ru}» — 0. "
            f"Остаются прочие подтверждённые замечания ({len(confirmed)}: "
            f"существенных — {major_count}, незначительных — {minor_count})."
        )

    return Verdict(
        ready_for_development=ready_for_development,
        ready_for_testing=ready_for_testing,
        fix_first=fix_first,
        rationale=rationale,
    )
