"""Активное обучение автомата: алгоритм `L*` и минимально адекватный учитель.

ЛР2 2024 целиком про это (`lab_tfl_2024_2.pdf`), ЛР3 2023 — тоже, включая
`NL*` для НКА. Постановка со слайда 4:

> Рекомендуется использовать фреймворк алгоритма `L∗`, допускающего вывод
> произвольного регулярного языка в форме таблицы классов эквивалентности
> Майхилла–Нероде. `S` — классы эквивалентности (определяющие состояния
> ДКА); `E` — различающие суффиксы; `S.Σ` — расширенные префиксы.

Цикл со схемы: пополнить `S.Σ × E` → замкнута? → непротиворечива? →
эквивалентна? → выиграли. Иначе, соответственно, строка в `S`, столбец
в `E` либо контрпример.

Условия взяты со слайда 5 дословно:

> Условие полноты — отсутствие в `S.Σ × E` строк, которые отличаются
> от строк в `S × E`.
> Условие непротиворечивости — отсутствие в `S.Σ × E` таких позиций
> `i, j, k`, что `uᵢvₖ ≠ uⱼvₖ`, притом что `uᵢ = u′ᵢγ`, `uⱼ = u′ⱼγ`,
> и строки в таблице `S × E` для `u′ᵢ` и `u′ⱼ` совпадают. Иначе дополняем
> `E` столбцом `γvₖ`.

**Две стратегии обработки контрпримера** названы в issue #31 как разные
варианты задания: «один добавляет в таблицу классов **суффиксы**
контрпримеров, другой — **префиксы**». Реализованы обе, и разница
между ними меряется, а не предполагается.

**Два характера МАТа** оттуда же: «один благосклонный, выдающий
**минимальный по длине контрпример**, второй — издевательский, выдающий
контрпример с несколькими накачками, т.е. проходами по одному и тому же
циклическому пути». Тоже обе.

Про честность счёта. Учитель считает **все** обращения к нему, а таблица
держит свой кеш: повторно спрашивать одно и то же слово никто не мешает,
но и заслугой алгоритма это не является. В отчёт идут оба числа.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tfl.automata import DFA, State, counterexample, product

__all__ = [
    "Teacher",
    "DFATeacher",
    "ObservationTable",
    "LearningResult",
    "learn",
    "SUFFIXES",
    "PREFIXES",
    "KIND_SHORTEST",
    "KIND_PUMPED",
]

#: Стратегии обработки контрпримера (issue #31).
SUFFIXES = "суффиксы"
PREFIXES = "префиксы"

#: Характер МАТа при выдаче контрпримера.
KIND_SHORTEST = "минимальный"
KIND_PUMPED = "издевательский"

EPSILON = "ε"


def _shown(word: str) -> str:
    """Пустое слово в таблицах курса пишется словом `epsilon`."""
    return word or "epsilon"


# --------------------------------------------------------------------------
# Учитель
# --------------------------------------------------------------------------


@dataclass
class Teacher:
    """Минимально адекватный учитель: отвечает на два вида запросов.

    Считает обращения. Угадыватель кешировать ответы вправе, но учитель
    об этом ничего не знает и считает то, что до него дошло.
    """

    membership_queries: int = 0
    equivalence_queries: int = 0

    def member(self, word: str) -> bool:
        raise NotImplementedError

    def equivalent(self, hypothesis: DFA) -> str | None:
        """`None`, если языки совпали, иначе слово-контрпример."""
        raise NotImplementedError


@dataclass
class DFATeacher(Teacher):
    """Учитель, за которым стоит известный ДКА.

    Так МАТ и устроен в задании: он знает автомат, отвечает на включение
    разбором по нему, а на эквивалентность — пересечением гипотезы
    с дополнением цели (и наоборот) и выдачей слова из разности.
    """

    target: DFA | None = None
    kind: str = KIND_SHORTEST
    pump_times: int = 3

    def member(self, word: str) -> bool:
        self.membership_queries += 1
        return self.target.accepts(word)

    def equivalent(self, hypothesis: DFA) -> str | None:
        self.equivalence_queries += 1
        if self.kind == KIND_SHORTEST:
            return counterexample(self.target, hypothesis)
        witness = _pumped_counterexample(self.target, hypothesis, self.pump_times)
        return witness if witness is not None else counterexample(self.target, hypothesis)


def _pumped_counterexample(target: DFA, hypothesis: DFA, times: int) -> str | None:
    """Контрпример с несколькими проходами по одному циклу.

    Строится в автомате симметрической разности: путь до состояния `q`,
    цикл в `q`, пройденный `times` раз, и путь из `q` в принимающее.
    Если цикла не нашлось, вернётся `None` — тогда МАТ отдаёт обычный
    кратчайший контрпример, и это не обман: издевательство не обязано
    удаваться всегда.
    """
    difference = product(target, hypothesis, lambda left, right: left != right)
    if not difference.finals:
        return None

    def path(source: State, targets) -> str | None:
        if source in targets:
            return ""
        seen = {source}
        queue: deque[tuple[State, str]] = deque([(source, "")])
        while queue:
            state, word = queue.popleft()
            for letter in sorted(difference.alphabet):
                following = difference.delta.get((state, letter))
                if following is None or following in seen:
                    continue
                grown = word + letter
                if following in targets:
                    return grown
                seen.add(following)
                queue.append((following, grown))
        return None

    def cycle_at(state: State) -> str | None:
        for letter in sorted(difference.alphabet):
            following = difference.delta.get((state, letter))
            if following is None:
                continue
            if following == state:
                return letter
            back = path(following, {state})
            if back:
                return letter + back
        return None

    seen = {difference.start}
    queue: deque[tuple[State, str]] = deque([(difference.start, "")])
    while queue:
        state, prefix = queue.popleft()
        loop = cycle_at(state)
        if loop:
            tail = path(state, difference.finals)
            if tail is not None:
                return prefix + loop * times + tail
        for letter in sorted(difference.alphabet):
            following = difference.delta.get((state, letter))
            if following is not None and following not in seen:
                seen.add(following)
                queue.append((following, prefix + letter))
    return None


# --------------------------------------------------------------------------
# Таблица наблюдений
# --------------------------------------------------------------------------


@dataclass
class ObservationTable:
    """Таблица классов: строки `S` и `S·Σ`, столбцы `E`."""

    alphabet: str
    teacher: Teacher
    prefixes: list[str] = field(default_factory=lambda: [""])
    suffixes: list[str] = field(default_factory=lambda: [""])
    cache: dict[str, bool] = field(default_factory=dict)

    def ask(self, word: str) -> bool:
        if word not in self.cache:
            self.cache[word] = self.teacher.member(word)
        return self.cache[word]

    def row(self, prefix: str) -> tuple[bool, ...]:
        return tuple(self.ask(prefix + suffix) for suffix in self.suffixes)

    def extended(self) -> list[str]:
        """`S·Σ` без тех слов, что уже лежат в `S`."""
        found = []
        for prefix in self.prefixes:
            for letter in self.alphabet:
                word = prefix + letter
                if word not in self.prefixes and word not in found:
                    found.append(word)
        return found

    # ------------------------------------------------------------- условия

    def unclosed(self) -> str | None:
        """Строка из `S·Σ`, не совпадающая ни с одной строкой `S`."""
        known = {self.row(prefix) for prefix in self.prefixes}
        for word in self.extended():
            if self.row(word) not in known:
                return word
        return None

    def inconsistency(self) -> str | None:
        """Столбец `γvₖ`, которого не хватает для непротиворечивости.

        Ищутся два префикса `S` с одинаковыми строками, у которых
        продолжение одной и той же буквой строки разводит.
        """
        for i, first in enumerate(self.prefixes):
            for second in self.prefixes[i + 1 :]:
                if self.row(first) != self.row(second):
                    continue
                for letter in self.alphabet:
                    left, right = self.row(first + letter), self.row(second + letter)
                    if left == right:
                        continue
                    for index, (mine, yours) in enumerate(zip(left, right)):
                        if mine != yours:
                            return letter + self.suffixes[index]
        return None

    def close(self) -> int:
        """Замкнуть таблицу, добавляя строки. Возвращает число добавленных."""
        added = 0
        while (word := self.unclosed()) is not None:
            self.prefixes.append(word)
            added += 1
        return added

    def make_consistent(self) -> int:
        """Сделать непротиворечивой, добавляя столбцы."""
        added = 0
        while (column := self.inconsistency()) is not None:
            if column in self.suffixes:
                break  # столбец уже есть — дальше добавлять нечего
            self.suffixes.append(column)
            added += 1
        return added

    # -------------------------------------------------------------- вывод

    def to_dfa(self) -> DFA:
        """Построить ДКА по таблице — конструкция со слайда 5.

        > Состояния ДКА — кратчайшие слова из `S`, порождающие разные строки
        > в `S × E`. Начальное состояние соответствует префиксу `ε`.
        > Конечные состояния — те, которые содержат 1 в столбце, помеченном `ε`.
        > Если `uγ ≡ u′`, то `⟨u, γ⟩ → u′` добавляется в ДКА как переход.
        """
        representative: dict[tuple[bool, ...], str] = {}
        for prefix in sorted(self.prefixes, key=lambda w: (len(w), w)):
            representative.setdefault(self.row(prefix), prefix)

        empty = self.suffixes.index("")
        finals = {name for signature, name in representative.items() if signature[empty]}
        delta: dict[tuple[State, str], State] = {}
        for signature, name in representative.items():
            for letter in self.alphabet:
                following = representative.get(self.row(name + letter))
                if following is not None:
                    delta[(name, letter)] = following
        return DFA(
            frozenset(self.alphabet),
            representative[self.row("")],
            frozenset(finals),
            delta,
        )

    def markdown(self) -> str:
        """Таблица в формате запроса об эквивалентности со слайда 7.

        > Здесь `epsilon` — указание на пустую строку, имена строк — классы
        > эквивалентности, `valk` — 0 или 1.
        """
        header = " | ".join(_shown(suffix) for suffix in self.suffixes)
        lines = [f"        | {header}", "-" * (10 + len(header))]
        for group, title in ((self.prefixes, "S"), (self.extended(), "S·Σ")):
            lines.append(f"[{title}]")
            for prefix in group:
                values = " | ".join(
                    "1" if value else "0" for value in self.row(prefix)
                )
                lines.append(f"{_shown(prefix):>8}| {values}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LearningResult:
    """Итог обучения: автомат, таблица и счёт запросов."""

    dfa: DFA
    table: ObservationTable
    rounds: int
    counterexamples: tuple[str, ...]
    membership_queries: int
    equivalence_queries: int
    distinct_words: int
    converged: bool

    def summary(self) -> str:
        state = "выучен" if self.converged else "НЕ выучен (лимит раундов)"
        return (
            f"автомат {state}: {len(self.dfa)} состояний, раундов {self.rounds}, "
            f"запросов о принадлежности {self.membership_queries} "
            f"(различных слов {self.distinct_words}), "
            f"об эквивалентности {self.equivalence_queries}"
        )


def learn(
    teacher: Teacher,
    alphabet: str,
    strategy: str = SUFFIXES,
    max_rounds: int = 60,
) -> LearningResult:
    """Алгоритм `L*`: выучить ДКА запросами к учителю.

    Стратегия — что делать с контрпримером:

    * `SUFFIXES` — «дополняем `E` им и всеми его суффиксами»;
    * `PREFIXES` — «дополняем `S` им и всеми его префиксами».

    Обе названы в условии как варианты задания. Первая обычно даёт
    меньше строк и больше столбцов, вторая наоборот; какая выгоднее —
    зависит от автомата, и это меряется, а не решается заранее.

    `max_rounds` — предохранитель. Алгоритм завершается всегда, если
    учитель честен, но нечестный учитель (или ошибка в предикате) иначе
    подвесил бы цикл, а молча крутиться в проекте не принято.
    """
    if strategy not in (SUFFIXES, PREFIXES):
        raise ValueError(f"стратегия должна быть «{SUFFIXES}» либо «{PREFIXES}»")

    table = ObservationTable(alphabet, teacher)
    seen: list[str] = []
    for round_number in range(1, max_rounds + 1):
        while True:
            table.close()
            if table.make_consistent() == 0:
                break

        hypothesis = table.to_dfa()
        witness = teacher.equivalent(hypothesis)
        if witness is None:
            return LearningResult(
                hypothesis,
                table,
                round_number,
                tuple(seen),
                teacher.membership_queries,
                teacher.equivalence_queries,
                len(table.cache),
                True,
            )
        seen.append(witness)
        if strategy == SUFFIXES:
            for index in range(len(witness) + 1):
                suffix = witness[index:]
                if suffix not in table.suffixes:
                    table.suffixes.append(suffix)
        else:
            for index in range(len(witness) + 1):
                prefix = witness[:index]
                if prefix not in table.prefixes:
                    table.prefixes.append(prefix)

    return LearningResult(
        table.to_dfa(),
        table,
        max_rounds,
        tuple(seen),
        teacher.membership_queries,
        teacher.equivalence_queries,
        len(table.cache),
        False,
    )
