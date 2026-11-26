"""The output schema a panel reviewer fills in.

Enumerated fields are plain strings with an `enum` hint in the JSON schema, so the
model is steered to the closed lists while one odd value costs only that finding
(the runner drops it) instead of failing the whole reviewer's answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from tekti.contracts import Criterion, Role, Severity, list_from_json_text

_CRITERIA = [c.value for c in Criterion]
_SEVERITIES = [s.value for s in Severity]
_ROLES = [r.value for r in Role]


class ReviewerEvidence(BaseModel):
    quote: str = Field(
        description=(
            "Точная цитата из документа, скопированная символ в символ, включая знаки разметки. "
            "Только текст блока, без метки [B-...] и без многоточий."
        )
    )
    requirement_id: str | None = Field(
        default=None, description="Идентификатор требования вида R-001, к которому относится эта цитата, если есть."
    )


class ReviewerFinding(BaseModel):
    requirement_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Идентификаторы R-... из списка требований: один для дефекта внутри требования, "
            "два и более для противоречия, пустой список для пробела во всём документе."
        ),
    )
    criterion: str = Field(description="Один критерий из закрытого списка.", json_schema_extra={"enum": _CRITERIA})
    severity: str = Field(
        description="critical, major или minor по тестам серьёзности.", json_schema_extra={"enum": _SEVERITIES}
    )
    title: str = Field(description="Одна строка по-русски: суть дефекта.")
    explanation: str = Field(
        description="По-русски, 1-4 предложения: что именно в процитированном тексте делает его дефектным и к чему это приведёт."
    )
    evidence: list[ReviewerEvidence] = Field(
        description="Хотя бы одна точная цитата; для противоречия по одной цитате с каждой стороны."
    )
    question: str | None = Field(
        default=None, description="Вопрос по-русски, ответ на который устраняет дефект; null, если вопрос не нужен."
    )
    addressee: str | None = Field(
        default=None,
        description="Роль, которая может ответить на вопрос; null, если вопроса нет.",
        json_schema_extra={"enum": [*_ROLES, None]},
    )

    @field_validator("requirement_ids", "evidence", mode="before")
    @classmethod
    def _lists(cls, value: object) -> object:
        # Same repair one level down; a single evidence object is taken as a list of one.
        return [value] if isinstance(value, dict) else list_from_json_text(value)


class ReviewerOutput(BaseModel):
    findings: list[ReviewerFinding] = Field(
        default_factory=list,
        description="Все найденные дефекты в пределах вашего угла. Пустой список, если дефектов нет.",
    )

    @field_validator("findings", mode="before")
    @classmethod
    def _findings_as_list(cls, value: object) -> object:
        # Claude Haiku 5.5 sometimes sends the whole list as one JSON string; accept it instead of
        # paying for a second attempt (and losing the reviewer when that fails too).
        return list_from_json_text(value)
