"""The output schema a panel reviewer fills in.

Enumerated fields are plain strings with an `enum` hint in the JSON schema, so the
model is steered to the closed lists while one odd value costs only that finding
(the runner drops it) instead of failing the whole reviewer's answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tekti.contracts import Criterion, Role, Severity

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


class ReviewerOutput(BaseModel):
    # Scratch space so models without built-in reasoning actually do the cross-checks before
    # writing findings. Code discards it; it never reaches the report.
    worksheet: str = Field(
        default="",
        description=(
            "Рабочий черновик сверки по методу вашего угла: коротко, по строке на проверку, не более 40 строк. "
            "Заполняется ДО списка находок. В отчёт не попадает."
        ),
    )
    findings: list[ReviewerFinding] = Field(
        default_factory=list,
        description="Все найденные дефекты в пределах вашего угла. Пустой список, если дефектов нет.",
    )
