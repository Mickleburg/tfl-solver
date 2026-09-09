"""Удаление правил: завершимость по частям.

Прямая интерпретация (`tfl/matrix.py`, `tfl/arctic.py`) требует, чтобы
**каждое** правило строго убывало под одной и той же мерой. Требование
жёсткое: у варианта 11 ЛР1 правила `aaa → bab` и `bbb → aaa` требуют
одновременно `[a] > [b]` и `[b] > [a]`, и никакой одной меры быть
не может — а система при этом вполне может оказаться завершимой.

Удаление правил меняет требование на посильное:

* **все** правила убывают нестрого;
* **хотя бы одно** — строго.

Строго убывающие правила из системы выбрасываются, и всё повторяется
на остатке. Когда правил не осталось, завершимость доказана.

## Почему это законно

Пусть `μ` — мера интерпретации (`M(w)[0][d−1]`). Нестрогое убывание
переносится в контекст, значит вдоль любого вывода `μ` не растёт;
на шаге строго убывающего правила `μ` уменьшается, причём начиная
с конечного значения. Значения лежат в фундированном множестве,
поэтому строгих шагов в выводе конечное число. Отрежем начало вывода
до последнего строгого шага: хвост — бесконечный вывод системы **без**
удалённых правил. Индукцией по числу шагов удаления: если остаток
завершим, завершима и исходная система.

Это то же рассуждение, что стоит за парами зависимостей
(`tfl/deppair.py`), только там нестрогость требуется от правил,
а строгость — от пар в компоненте графа; здесь всё происходит
на самих правилах, зато шагов может быть несколько.

## Что здесь считается, а что проверяется

Шаги ищет решатель (`matrix.relative_step`, `arctic.relative_step`),
и его ответ никуда наружу не идёт напрямую: `RemovalProof.check`
пересобирает всю цепочку заново и **сам** решает, какие правила на каждом
шаге убывают строго, целочисленной арифметикой. Наружу выдаётся только
то, что арбитр принял.
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl.verdict import Verdict, proved, unknown

__all__ = ["RemovalStep", "RemovalProof", "prove_by_removal"]


@dataclass(frozen=True)
class RemovalStep:
    """Один шаг: интерпретация и правила, которые она позволяет убрать."""

    kind: str
    interpretation: object
    removed: tuple

    def __str__(self) -> str:
        rules = ", ".join(f"«{rule}»" for rule in self.removed)
        return f"{self.kind} интерпретация {self.interpretation} убирает {rules}"


@dataclass(frozen=True)
class RemovalProof:
    """Цепочка шагов удаления. Проверяется целиком, а не по одному."""

    steps: tuple[RemovalStep, ...]

    def check(self, system) -> Verdict:
        """Арбитр: пересобирает цепочку и считает всё заново."""
        if not self.steps:
            return unknown("цепочка удаления пуста")
        remaining = list(system.rules)
        for number, step in enumerate(self.steps, start=1):
            interpretation = step.interpretation
            if not interpretation.is_monotone():
                return unknown(
                    f"шаг {number}: интерпретация не монотонна, "
                    f"убывание не переносится в контекст"
                )
            missing = {
                letter
                for rule in remaining
                for letter in rule.lhs + rule.rhs
                if letter not in interpretation.matrices
            }
            if missing:
                return unknown(
                    f"шаг {number}: нет матриц для букв {', '.join(sorted(missing))}"
                )
            weak = [
                rule
                for rule in remaining
                if not interpretation.weakly_decreases(rule.lhs, rule.rhs)
            ]
            if weak:
                return unknown(
                    f"шаг {number}: правило «{weak[0]}» возрастает, "
                    f"а обязано хотя бы не возрастать"
                )
            strict = [
                rule
                for rule in remaining
                if interpretation.decreases(rule.lhs, rule.rhs)
            ]
            if not strict:
                return unknown(f"шаг {number}: ни одно правило не убывает строго")
            extra = [rule for rule in step.removed if rule not in strict]
            if extra:
                return unknown(
                    f"шаг {number}: правило «{extra[0]}» объявлено удалённым, "
                    f"но строго не убывает"
                )
            if not step.removed:
                return unknown(f"шаг {number}: не удалено ни одного правила")
            remaining = [rule for rule in remaining if rule not in step.removed]
        if remaining:
            return unknown(
                f"после всех шагов осталось {len(remaining)} правил, "
                f"первое — «{remaining[0]}»"
            )
        return proved(
            f"завершимость доказана удалением правил за {len(self.steps)} "
            f"шаг(ов): " + "; ".join(str(step) for step in self.steps),
            self,
        )

    def markdown(self) -> str:
        lines = []
        for number, step in enumerate(self.steps, start=1):
            rules = ", ".join(f"`{rule}`" for rule in step.removed)
            lines.append(f"**Шаг {number}** ({step.kind}). Убираются {rules}.")
            lines.append("")
            lines.append(step.interpretation.markdown())
            lines.append("")
        return "\n".join(lines)


def prove_by_removal(
    system,
    dimensions: tuple[int, ...] = (1, 2, 3),
    max_entry: int = 3,
    timeout_ms: int = 20_000,
) -> Verdict:
    """Доказать завершимость последовательным удалением правил.

    Размерности перебираются по возрастанию, на каждой сначала обычная
    матричная интерпретация, потом арктическая: они несравнимы, и порядок
    выбран по цене, а не по силе.

    Метод **только доказывает**: «не выяснено» не значит «не завершима».
    """
    from tfl import arctic, matrix

    if not matrix.have_solver():
        return unknown(
            "SMT-решателя Z3 в окружении нет, удаление правил не выполнялось"
        )
    remaining = list(system.rules)
    steps: list[RemovalStep] = []
    while remaining:
        found = None
        for dimension in dimensions:
            for kind, module in (("матричная", matrix), ("арктическая", arctic)):
                interpretation = module.relative_step(
                    remaining, dimension, max_entry, timeout_ms
                )
                if interpretation is None:
                    continue
                if not all(
                    interpretation.weakly_decreases(rule.lhs, rule.rhs)
                    for rule in remaining
                ):
                    continue  # решатель ошибся или кодировка разошлась с арбитром
                strict = tuple(
                    rule
                    for rule in remaining
                    if interpretation.decreases(rule.lhs, rule.rhs)
                )
                if strict:
                    found = RemovalStep(kind, interpretation, strict)
                    break
            if found is not None:
                break
        if found is None:
            return unknown(
                f"удалить больше нечего: осталось {len(remaining)} правил "
                f"(первое — «{remaining[0]}»), а интерпретации размерностей "
                f"{dimensions} с элементами до {max_entry} ни одного из них "
                f"строго не роняют"
            )
        steps.append(found)
        remaining = [rule for rule in remaining if rule not in found.removed]
    return RemovalProof(tuple(steps)).check(system)
