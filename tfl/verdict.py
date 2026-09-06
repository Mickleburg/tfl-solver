"""Трёхзначный вердикт: доказано / опровергнуто / не выяснено.

Вынесен в отдельный модуль, потому что нужен везде, где свойство
неразрешимо в общем случае: SRS (`tfl/srs.py`), магазинные автоматы
(`tfl/pda.py`), беспрефиксность и эквивалентность КС-языков
(`tfl/parse.py`).

Смысл `__bool__` — запретить самую опасную опечатку в этом проекте:
`if check(...)` там, где `check` вернул «не выяснено», молча превращает
незнание в отрицательный ответ.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Verdict", "proved", "refuted", "unknown"]


@dataclass
class Verdict:
    """Результат проверки, которая может и не завершиться выводом.

    `value` — True (доказано), False (опровергнуто) или None (не выяснено
    в пределах бюджета). `witness` — то, что предъявляется в отчёте:
    цикл, критическая пара, порядок, контрпример.
    """

    value: bool | None
    reason: str
    witness: object = None

    def __bool__(self) -> bool:  # pragma: no cover - защита от опечатки
        raise TypeError(
            "Verdict нельзя использовать как bool: у него три исхода. "
            "Проверяйте .value is True / is False / is None"
        )

    def __str__(self) -> str:
        mark = {True: "да", False: "нет", None: "не выяснено"}[self.value]
        return f"{mark}: {self.reason}"


def proved(reason: str, witness: object = None) -> Verdict:
    return Verdict(True, reason, witness)


def refuted(reason: str, witness: object = None) -> Verdict:
    return Verdict(False, reason, witness)


def unknown(reason: str, witness: object = None) -> Verdict:
    return Verdict(None, reason, witness)
