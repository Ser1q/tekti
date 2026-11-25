"""Deterministic rule layer (P3): pattern checks, fact extraction, and cross-document checks."""

from __future__ import annotations

from tekti.contracts import Document, Fact, RawFinding, Requirement
from tekti.rules.checks import (
    check_embedded_instructions,
    check_empty_and_placeholder_sections,
    check_traceable,
    check_weak_wording,
)
from tekti.rules.common import DocVerifier, extract_tables
from tekti.rules.facts import extract_and_compare


def run(doc: Document, requirements: list[Requirement]) -> tuple[list[Fact], list[RawFinding]]:
    """Run all deterministic rules and fact comparisons on a `Document`."""
    verifier = DocVerifier(doc, requirements)
    tables = extract_tables(doc)

    findings: list[RawFinding] = []
    findings.extend(check_embedded_instructions(doc, verifier))
    findings.extend(check_empty_and_placeholder_sections(doc, tables, verifier))
    findings.extend(check_weak_wording(doc, verifier))
    findings.extend(check_traceable(doc, requirements, tables, verifier))

    facts, comparison_findings = extract_and_compare(doc, tables, verifier)
    findings.extend(comparison_findings)
    return facts, findings


__all__ = ["run"]
