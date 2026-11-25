"""Offline unit tests for tekti.verdict."""

from __future__ import annotations

from tekti.contracts import (
    Criterion,
    Evidence,
    Finding,
    FindingStatus,
    ReviewerId,
    Severity,
)
from tekti.criteria import NAME_RU
from tekti.verdict import decide


def _finding(
    fid: str,
    severity: Severity,
    criterion: Criterion = Criterion.UNAMBIGUOUS,
    *,
    families: int = 1,
    status: FindingStatus = FindingStatus.CONFIRMED,
    req_ids: list[str] | None = None,
) -> Finding:
    return Finding(
        id=fid,
        requirement_ids=["R-001"] if req_ids is None else req_ids,
        criterion=criterion,
        severity=severity,
        title=f"Замечание {fid}",
        explanation="Конкретное пояснение со ссылкой на текст.",
        evidence=[Evidence(quote="Точная цитата из документа.")],
        raised_by=[ReviewerId(model="rules")],
        families=families,
        status=status,
    )


def test_empty_findings_ready_for_dev_and_testing() -> None:
    v = decide([])
    assert v.ready_for_development is True
    assert v.ready_for_testing is True
    assert v.fix_first == []
    assert "0" in v.rationale
    assert "можно начинать" in v.rationale.lower()


def test_critical_blocks_both_orders_fix_first_and_caps_at_ten() -> None:
    findings = [
        _finding("F-001", Severity.CRITICAL, Criterion.SET_CONSISTENT, families=1, req_ids=["R-001", "R-002"]),
        _finding("F-002", Severity.CRITICAL, Criterion.VERIFIABLE, families=3),
        _finding("F-003", Severity.CRITICAL, Criterion.COMPLETE, families=2),
        _finding("F-004", Severity.CRITICAL, Criterion.CORRECT, families=3),
        _finding("F-005", Severity.CRITICAL, Criterion.FEASIBLE, families=1),
        _finding("F-006", Severity.CRITICAL, Criterion.SET_COMPLETE, families=2, req_ids=[]),
        _finding("F-007", Severity.CRITICAL, Criterion.VERIFIABLE, families=1),
        _finding("F-008", Severity.CRITICAL, Criterion.SET_VALIDATABLE, families=2, req_ids=[]),
        _finding("F-009", Severity.CRITICAL, Criterion.UNAMBIGUOUS, families=1),
        _finding("F-010", Severity.CRITICAL, Criterion.SET_CONSISTENT, families=3, req_ids=["R-002", "R-003"]),
        _finding("F-011", Severity.CRITICAL, Criterion.COMPLETE, families=2),
        _finding("F-012", Severity.CRITICAL, Criterion.CORRECT, families=1),
        _finding("F-013", Severity.MAJOR, Criterion.VERIFIABLE, families=2),
    ]
    v = decide(findings)
    assert v.ready_for_development is False
    assert v.ready_for_testing is False
    # Sorted by (-families, input_order), capped at 10:
    # families=3: F-002, F-004, F-010
    # families=2: F-003, F-006, F-008, F-011
    # families=1: F-001, F-005, F-007 (F-009 and F-012 cut off by cap 10)
    assert v.fix_first == [
        "F-002",
        "F-004",
        "F-010",
        "F-003",
        "F-006",
        "F-008",
        "F-011",
        "F-001",
        "F-005",
        "F-007",
    ]
    assert "12" in v.rationale
    assert "F-002" in v.rationale
    assert "F-013" in v.rationale
    assert "нельзя" in v.rationale.lower()


def test_major_verifiable_allows_dev_blocks_testing() -> None:
    findings = [
        _finding("F-001", Severity.MAJOR, Criterion.VERIFIABLE, families=2),
        _finding("F-002", Severity.MINOR, Criterion.CONFORMING, families=1),
    ]
    v = decide(findings)
    assert v.ready_for_development is True
    assert v.ready_for_testing is False
    assert v.fix_first == []
    assert "0" in v.rationale
    assert "F-001" in v.rationale
    assert NAME_RU[Criterion.VERIFIABLE] in v.rationale


def test_major_set_validatable_allows_dev_blocks_testing() -> None:
    findings = [_finding("F-001", Severity.MAJOR, Criterion.SET_VALIDATABLE, families=1, req_ids=[])]
    v = decide(findings)
    assert v.ready_for_development is True
    assert v.ready_for_testing is False
    assert v.fix_first == []
    assert "F-001" in v.rationale
    assert NAME_RU[Criterion.SET_VALIDATABLE] in v.rationale


def test_minor_verifiable_and_other_majors_allow_both() -> None:
    findings = [
        _finding("F-001", Severity.MAJOR, Criterion.UNAMBIGUOUS),
        _finding("F-002", Severity.MAJOR, Criterion.COMPLETE),
        _finding("F-003", Severity.MAJOR, Criterion.SET_CONSISTENT, req_ids=["R-001", "R-002"]),
        _finding("F-004", Severity.MINOR, Criterion.VERIFIABLE),
        _finding("F-005", Severity.MINOR, Criterion.SET_VALIDATABLE, req_ids=[]),
    ]
    v = decide(findings)
    assert v.ready_for_development is True
    assert v.ready_for_testing is True
    assert v.fix_first == []
    assert "0" in v.rationale
    assert "5" in v.rationale
    assert "3" in v.rationale
    assert "2" in v.rationale


def test_rejected_findings_are_ignored() -> None:
    findings = [
        _finding("X-001", Severity.CRITICAL, Criterion.SET_CONSISTENT, families=3, status=FindingStatus.REJECTED),
        _finding("X-002", Severity.MAJOR, Criterion.VERIFIABLE, families=2, status=FindingStatus.REJECTED),
        _finding("F-001", Severity.MINOR, Criterion.CONFORMING, families=1, status=FindingStatus.CONFIRMED),
    ]
    v = decide(findings)
    assert v.ready_for_development is True
    assert v.ready_for_testing is True
    assert v.fix_first == []

    v_only_rejected = decide(findings[:2])
    assert v_only_rejected.ready_for_development is True
    assert v_only_rejected.ready_for_testing is True
    assert v_only_rejected.fix_first == []
