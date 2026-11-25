"""Russian names and one-line definitions for the closed criterion list.

Paraphrased from ISO/IEC/IEEE 29148:2018 (5.2.5, 5.2.6) and IEEE 830-1998 (4.3);
not quoted from the standards. Used in prompts and in the report.
"""

from __future__ import annotations

from tekti.contracts import Criterion

SOURCE: dict[Criterion, str] = {
    Criterion.NECESSARY: "ISO 29148 п. 5.2.5",
    Criterion.APPROPRIATE: "ISO 29148 п. 5.2.5",
    Criterion.UNAMBIGUOUS: "ISO 29148 п. 5.2.5; IEEE 830 п. 4.3.2",
    Criterion.COMPLETE: "ISO 29148 п. 5.2.5",
    Criterion.SINGULAR: "ISO 29148 п. 5.2.5",
    Criterion.FEASIBLE: "ISO 29148 п. 5.2.5",
    Criterion.VERIFIABLE: "ISO 29148 п. 5.2.5; IEEE 830 п. 4.3.6",
    Criterion.CORRECT: "ISO 29148 п. 5.2.5; IEEE 830 п. 4.3.1",
    Criterion.CONFORMING: "ISO 29148 п. 5.2.5",
    Criterion.SET_COMPLETE: "ISO 29148 п. 5.2.6; IEEE 830 п. 4.3.3",
    Criterion.SET_CONSISTENT: "ISO 29148 п. 5.2.6; IEEE 830 п. 4.3.4",
    Criterion.SET_FEASIBLE: "ISO 29148 п. 5.2.6",
    Criterion.SET_COMPREHENSIBLE: "ISO 29148 п. 5.2.6",
    Criterion.SET_VALIDATABLE: "ISO 29148 п. 5.2.6",
    Criterion.TRACEABLE: "IEEE 830 п. 4.3.8",
    Criterion.MODIFIABLE: "IEEE 830 п. 4.3.7",
    Criterion.RANKED: "IEEE 830 п. 4.3.5",
    Criterion.EMBEDDED_INSTRUCTION: "вне стандартов: защита проверки",
}

NAME_RU: dict[Criterion, str] = {
    Criterion.NECESSARY: "Необходимость",
    Criterion.APPROPRIATE: "Уместность уровня",
    Criterion.UNAMBIGUOUS: "Однозначность",
    Criterion.COMPLETE: "Полнота требования",
    Criterion.SINGULAR: "Единичность",
    Criterion.FEASIBLE: "Осуществимость",
    Criterion.VERIFIABLE: "Проверяемость",
    Criterion.CORRECT: "Корректность",
    Criterion.CONFORMING: "Соответствие шаблону",
    Criterion.SET_COMPLETE: "Полнота набора",
    Criterion.SET_CONSISTENT: "Непротиворечивость набора",
    Criterion.SET_FEASIBLE: "Осуществимость набора",
    Criterion.SET_COMPREHENSIBLE: "Понятность набора",
    Criterion.SET_VALIDATABLE: "Валидируемость набора",
    Criterion.TRACEABLE: "Трассируемость",
    Criterion.MODIFIABLE: "Модифицируемость",
    Criterion.RANKED: "Приоритизация",
    Criterion.EMBEDDED_INSTRUCTION: "Встроенная инструкция",
}

DEFINITION_RU: dict[Criterion, str] = {
    Criterion.NECESSARY: (
        "Требование задаёт нужное свойство: если его убрать, останется пробел, который не закрыт другими требованиями."
    ),
    Criterion.APPROPRIATE: (
        "Требование сформулировано на своём уровне: говорит, что нужно, и не навязывает проектное решение без причины."
    ),
    Criterion.UNAMBIGUOUS: (
        "Требование допускает одно прочтение. Нет размытых слов, неопределённых терминов и фраз, которые читаются двояко."
    ),
    Criterion.COMPLETE: (
        "Требование само содержит всё нужное для реализации: условия, значения, единицы, исключения. Нет «уточнить позже»."
    ),
    Criterion.SINGULAR: ("Требование описывает одно свойство или одно ограничение, а не несколько, склеенных вместе."),
    Criterion.FEASIBLE: ("Требование можно выполнить в заданных ограничениях по системам, срокам и технологиям."),
    Criterion.VERIFIABLE: (
        "Можно построить проверку с однозначным результатом: есть измеримый критерий, условия и ожидаемый исход."
    ),
    Criterion.CORRECT: (
        "Требование верно отражает потребность, из которой выведено, и не противоречит собственным примерам и критериям приёмки."
    ),
    Criterion.CONFORMING: (
        "Требование и документ оформлены по принятому шаблону: заполнены обязательные поля, есть идентификаторы."
    ),
    Criterion.SET_COMPLETE: (
        "Набор покрывает всё нужное: нет пропущенных сценариев, методов, данных, незаполненных разделов."
    ),
    Criterion.SET_CONSISTENT: (
        "Требования не противоречат друг другу; термины, единицы и типы используются одинаково во всём документе."
    ),
    Criterion.SET_FEASIBLE: ("Весь набор можно выполнить вместе в заданных ограничениях."),
    Criterion.SET_COMPREHENSIBLE: ("По набору понятно, что ожидается от системы и как требования связаны между собой."),
    Criterion.SET_VALIDATABLE: (
        "Можно показать, что выполнение набора приводит к заявленной цели: есть критерии приёмки, покрывающие требования."
    ),
    Criterion.TRACEABLE: (
        "У требования есть идентификатор и связь с источником и с зависимыми материалами: API, данными, тестами."
    ),
    Criterion.MODIFIABLE: (
        "Набор можно менять без рассогласования: нет дублирования одного правила в нескольких местах."
    ),
    Criterion.RANKED: ("У требований указана важность или стабильность."),
    Criterion.EMBEDDED_INSTRUCTION: (
        "В тексте документа есть указание, адресованное проверяющему или модели, а не требование к системе."
    ),
}
