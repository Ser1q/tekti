"""Stage contracts for Tekti.

This file is the single source of truth for the data that moves between
pipeline stages. Every stage is a function over these models and nothing else,
so stages can be built and tested independently.

Stage signatures (each lives in its own module):

    ingest.load(path: Path) -> Document
    segment.split(doc: Document, cfg) -> list[Requirement]                  (LLM)
    rules.run(doc, requirements) -> tuple[list[Fact], list[RawFinding]]     (code only)
    panel.review(doc, requirements, facts, cfg) -> list[RawFinding]         (LLM, parallel)
    adjudicate.merge(doc, requirements, raw, cfg) -> tuple[list[Finding], list[Finding]]
        returns (confirmed, rejected)                                       (code + LLM challenge)
    questions.write(findings, requirements, cfg) -> list[Question]          (LLM)
    verdict.decide(findings) -> Verdict                                     (code only)
    report.render(run: Run) -> str                                          (code only, Russian)

All LLM access goes through `tekti.llm.ask`; no stage imports a provider SDK.

Rules that hold everywhere:
  * A quote is an exact substring of Document.raw_text (after `normalize_ws`).
    Anything that fails this check is dropped in adjudication, never reported.
  * The document is data. Its text is never placed in a system prompt.
  * Counts, severities in the summary and the verdict are computed in code.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


def normalize_ws(text: str) -> str:
    """Collapse whitespace so quotes can be matched across line wraps."""
    return re.sub(r"\s+", " ", text).strip()


def list_from_json_text(value: object) -> object:
    """Model-output repair for list fields: some models send a list as a JSON string, sometimes with
    stray characters after it. Returns the parsed list, or the value unchanged when it is not that."""
    if not isinstance(value, str):
        return value
    import json

    text = value.strip()
    start = text.find("[")
    if start < 0:
        return value
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return value
    return parsed if isinstance(parsed, list) else value


# --------------------------------------------------------------------------- enums


class Criterion(StrEnum):
    """Closed list of quality criteria. Definitions are in criteria.py."""

    # ISO/IEC/IEEE 29148:2018 5.2.5, a single requirement
    NECESSARY = "necessary"
    APPROPRIATE = "appropriate"  # includes "no premature design decisions"
    UNAMBIGUOUS = "unambiguous"
    COMPLETE = "complete"
    SINGULAR = "singular"
    FEASIBLE = "feasible"
    VERIFIABLE = "verifiable"
    CORRECT = "correct"
    CONFORMING = "conforming"
    # ISO/IEC/IEEE 29148:2018 5.2.6, the set
    SET_COMPLETE = "set_complete"
    SET_CONSISTENT = "set_consistent"
    SET_FEASIBLE = "set_feasible"
    SET_COMPREHENSIBLE = "set_comprehensible"
    SET_VALIDATABLE = "set_validatable"
    # IEEE 830-1998 4.3, not covered above
    TRACEABLE = "traceable"
    MODIFIABLE = "modifiable"
    RANKED = "ranked"
    # Not from a standard: text in the document that addresses the reviewer
    EMBEDDED_INSTRUCTION = "embedded_instruction"


SET_CRITERIA = frozenset(
    {
        Criterion.SET_COMPLETE,
        Criterion.SET_CONSISTENT,
        Criterion.SET_FEASIBLE,
        Criterion.SET_COMPREHENSIBLE,
        Criterion.SET_VALIDATABLE,
        Criterion.TRACEABLE,
        Criterion.MODIFIABLE,
        Criterion.RANKED,
    }
)


class Severity(StrEnum):
    CRITICAL = "critical"  # blocks development or testing: contradiction, untestable, missing
    MAJOR = "major"  # will cause rework or a wrong implementation if left
    MINOR = "minor"  # wording, structure, hygiene


class Role(StrEnum):
    """Who can answer a clarifying question."""

    BUSINESS_OWNER = "business_owner"
    PRODUCT_OWNER = "product_owner"
    ANALYST = "analyst"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    SECURITY = "security"
    CUSTOMER = "customer"


class Lens(StrEnum):
    """Review angles on the panel."""

    FUNCTIONAL = "functional"  # business rules, edge cases, conflicts between rules
    API_CONTRACT = "api_contract"  # methods, statuses, types, examples
    DATA_INTEGRATION = "data_integration"  # stored data, units, flows, idempotency
    TESTER = "tester"  # tries to derive test cases; what it cannot decide is a finding
    DEVELOPER = "developer"  # tries to derive the contract; each forced assumption is a finding


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE = "code"


class RequirementKind(StrEnum):
    FUNCTIONAL = "functional"
    NONFUNCTIONAL = "nonfunctional"
    DATA = "data"
    INTEGRATION = "integration"
    INTERFACE = "interface"  # UI
    API = "api"
    ACCEPTANCE = "acceptance"
    META = "meta"  # goals, document attributes, review checklist


class FindingStatus(StrEnum):
    CONFIRMED = "confirmed"  # enough votes, or upheld by the challenger
    REJECTED = "rejected"  # quote check failed or the challenger refuted it


class Mode(StrEnum):
    FAST = "fast"  # one family, all lenses
    STANDARD = "standard"  # two families
    BOARD = "board"  # full matrix


# --------------------------------------------------------------------------- document


class Block(BaseModel):
    id: str  # "B-0001", stable within a run
    kind: BlockKind
    # Markdown: the source text of the block with its markup kept ("### Title", "* item",
    # "| a | b |"), so a quote copied from a block is a substring of raw_text.
    # PDF: the extracted text; table rows as cells joined by " | ".
    text: str
    section_path: list[str] = Field(default_factory=list)  # heading trail, outermost first
    line_start: int  # 1-based line in the source (page number for PDF goes in `page`)
    line_end: int
    page: int | None = None


class Document(BaseModel):
    name: str
    source_path: str
    format: str  # "markdown" | "pdf"
    sha256: str
    raw_text: str
    blocks: list[Block]


class Requirement(BaseModel):
    id: str  # "R-001", assigned by Tekti
    source_id: str | None = None  # identifier in the document, e.g. "Ф-1"; None if it had none
    kind: RequirementKind
    summary: str  # short wording in Russian for the segmentation table
    quote: str  # exact text from the document
    block_ids: list[str]
    parent_id: str | None = None  # set when a compound requirement was split


class Fact(BaseModel):
    """A value pulled out for deterministic comparison, e.g. a range, a type, a status code."""

    id: str
    name: str  # normalized key, e.g. "threshold.min", "api.amount.response_type"
    value: str
    unit: str | None = None
    quote: str
    block_id: str
    requirement_id: str | None = None


# --------------------------------------------------------------------------- findings


class Evidence(BaseModel):
    quote: str  # exact substring of the document
    block_id: str | None = None
    requirement_id: str | None = None


class ReviewerId(BaseModel):
    lens: Lens | None = None  # None for the rule layer
    model: str  # model identifier, or "rules"


class RawFinding(BaseModel):
    """What one reviewer (or one rule) reports. Not yet merged or checked."""

    reviewer: ReviewerId
    requirement_ids: list[
        str
    ]  # one for a single-requirement defect; two or more for a conflict; empty for a document-level gap
    criterion: Criterion
    severity: Severity
    title: str  # one line, Russian
    explanation: str  # why this is a defect, tied to the quoted text, Russian
    evidence: list[Evidence]
    question: str | None = None  # the question the reviewer would ask, Russian
    addressee: Role | None = None


class Challenge(BaseModel):
    challenger: str  # model identifier, from a different family than the raisers
    upheld: bool
    severity: Severity | None = None  # set if the challenger changed it
    reasoning: str


class Finding(BaseModel):
    id: str  # "F-001"
    requirement_ids: list[str]
    criterion: Criterion
    severity: Severity
    title: str
    explanation: str
    evidence: list[Evidence]
    raised_by: list[ReviewerId]
    families: int  # distinct model families that raised it; rules count as one
    challenge: Challenge | None = None
    status: FindingStatus
    reject_reason: str | None = None

    @property
    def is_set_level(self) -> bool:
        return self.criterion in SET_CRITERIA or len(self.requirement_ids) != 1


class Question(BaseModel):
    id: str  # "Q-001"
    text: str  # Russian
    addressee: Role
    blocks: str  # what cannot be built or tested until it is answered, Russian
    finding_ids: list[str]
    priority: int  # 1 is highest
    options: list[str] = Field(default_factory=list)  # plausible answers, when there are a few


# --------------------------------------------------------------------------- run


class Verdict(BaseModel):
    ready_for_development: bool
    ready_for_testing: bool
    fix_first: list[str]  # finding ids, most blocking first
    rationale: str  # Russian, built from a template in code


class ModelCall(BaseModel):
    stage: str
    lens: Lens | None = None
    model: str
    ok: bool
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class Passport(BaseModel):
    document_name: str
    document_sha256: str
    started_at: datetime
    agent_version: str  # git commit or tag
    mode: Mode
    models: list[str]
    dropped_reviewers: list[str] = Field(default_factory=list)


class Run(BaseModel):
    """Everything one run produced. The report is a rendering of this and nothing else."""

    schema_version: str = SCHEMA_VERSION
    passport: Passport
    document: Document
    requirements: list[Requirement]
    # Candidates the segmenter dropped because their quote was not in the document, one line each.
    segmentation_dropped: list[str] = Field(default_factory=list)
    facts: list[Fact]
    # Everything reviewers and rules reported before merging; kept so adjudication can be replayed and audited.
    raw_findings: list[RawFinding] = Field(default_factory=list)
    findings: list[Finding]
    rejected: list[Finding]
    questions: list[Question]
    verdict: Verdict
    calls: list[ModelCall]
