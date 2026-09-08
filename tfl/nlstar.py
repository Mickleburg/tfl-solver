"""Активное обучение НКА: алгоритм `NL*` и остаточные автоматы (RFSA).

ЛР3 2023 (`lab_tfl_2023_3.pdf`, слайд 10): «(Чётная последняя цифра
зачётки) алгоритм `L∗`. (Нечётная последняя цифра зачётки) алгоритм `NL∗`».
`L*` сделан в `tfl/lstar.py`; здесь вторая половина.

Определения со слайда 18, дословно:

> Алгоритм строит минимальный остаточный НКА (RFSA).
> Скажем, что строка `r` накрывается `⋃ₖ rₖ`, если `∀i (r[i] = ⋁ₖ rₖ[i])`
> (т.е. она является поэлементной дизъюнкцией строк `rₖ`).
> Строка `r₁` поглощает `r₂` (`r₂ ⊑ r₁`), если `∀i (r₁[i] ⩾ r₂[i])`.
> Условие полноты — отсутствие в `S.Σ × E` строк, которые **не
> накрываются** строками в `S × E`.
> Условие непротиворечивости — отсутствие в `S × E` таких позиций
> `i, j, k`, что `uᵢ ⊑ uⱼ`, но для некоторого `γ ∈ Σ` `uᵢγ ⋢ uⱼγ`
> в расширенной таблице.
> Состояния НКА — кратчайшие слова из `S`, базисные (т.е. не
> накрывающиеся набором других) в `S × E`. Начальные состояния включают
> строки, поглощаемые строкой `ε` (чтобы перейти к классическому НКА,
> придётся стянуть их в одно). Конечные состояния — те, которые содержат
> 1 в столбце, помеченном `ε`. Если `v ⊑ uγ`, то `⟨u, γ⟩ → v` добавляется
> в НКА как переход.

Чем `NL*` отличается от `L*` по сути, а не по буквам. `L*` ищет классы
Майхилла–Нероуда: строки таблицы сравниваются **на равенство**. `NL*`
ищет вычеты `u⁻¹L`, и сравнивает их **на включение**: состояние нужно
только тогда, когда его вычет не собирается объединением меньших. Поэтому
таблица та же, а условия на неё — про решётку, а не про разбиение.

Отсюда практическое следствие, которое стоит держать в голове при выборе
варианта: канонический RFSA **никогда не больше** минимального ДКА
(его состояния — подмножество вычетов) и бывает экспоненциально меньше.
Замер по проекту — в `docs/recipes/MAT.md`, §NL*.

Тонкость запроса об эквивалентности, названная в самом задании (слайд 11):

> Если у вас вариант с НКА, тогда придётся детерминизировать.

МАТ принимает описание регулярного языка и умеет сравнивать ДКА, поэтому
гипотеза-НКА перед запросом детерминизируется. Это не деталь реализации:
детерминизация стоит экспоненты, и **выигрыш `NL*` — в числе состояний
ответа, а не в цене запроса**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tfl.automata import DFA, NFA, State, difference
from tfl.lstar import (
    EPSILON,
    KIND_PUMPED,
    KIND_SHORTEST,
    PREFIXES,
    SUFFIXES,
    DFATeacher,
    Teacher,
    _shown,
)

__all__ = [
    "Row",
    "absorbs",
    "join",
    "covered",
    "is_prime",
    "RFSATable",
    "ZERO_COMPOSED",
    "ZERO_PRIME",
    "NFALearningResult",
    "learn_nfa",
    "canonical_rfsa",
    "residual_primes",
    "SUFFIXES",
    "PREFIXES",
    "KIND_SHORTEST",
    "KIND_PUMPED",
    "DFATeacher",
]

#: Строка таблицы наблюдений: по булеву значению на каждый суффикс из `E`.
Row = tuple[bool, ...]

#: Два прочтения слов «не накрывающиеся **набором других**» со слайда 18.
#: Разница — в нулевой строке: пустой набор её накрывает (дизъюнкция
#: пустого набора нулевая), и вопрос в том, считается ли пустой набор
#: «набором других».
#:
#: `ZERO_COMPOSED` — да, считается: нулевая строка не базисная, состояния
#: под неё нет. Это стандартное определение канонического RFSA и
#: единственное, при котором автомат **минимален** (лекция обещает
#: «минимальный остаточный НКА»).
#:
#: `ZERO_PRIME` — нет, набор обязан быть непустым: нулевая строка получает
#: состояние-ловушку. Автомат перестаёт быть минимальным, зато не рушится
#: при стратегии префиксов, см. §NL* рецепта `MAT`.
ZERO_COMPOSED = "нулевая накрывается пустым набором"
ZERO_PRIME = "нулевая базисна"

START = "→"


# --------------------------------------------------------------------------
# Решётка строк
# --------------------------------------------------------------------------


def absorbs(bigger: Row, smaller: Row) -> bool:
    """`smaller ⊑ bigger`: `∀i (bigger[i] ⩾ smaller[i])`."""
    return all(mine >= yours for mine, yours in zip(bigger, smaller))


def join(rows) -> Row:
    """Поэлементная дизъюнкция `⋁ₖ rₖ`. Пустой набор даёт нулевую строку."""
    rows = list(rows)
    if not rows:
        return ()
    return tuple(any(values) for values in zip(*rows))


def covered(row: Row, rows) -> bool:
    """Накрывается ли `row` объединением `rows`.

    Пустой набор накрывает только нулевую строку — на это опирается
    построение начальных состояний, когда `ε` не в языке.
    """
    rows = list(rows)
    if not rows:
        return not any(row)
    return join(rows) == row


def is_prime(row: Row, rows, zero_is_prime: bool = False) -> bool:
    """Базисная ли строка: не накрывается набором **строго меньших**.

    Оговорка «строго» существенна. Если сравнивать со всеми остальными
    строками таблицы, то две равные строки накроют друг друга и обе
    окажутся не базисными — состояний не останется вовсе.

    `zero_is_prime` — вторая трактовка слов «набором других» со слайда 18,
    см. `ZERO_COMPOSED` / `ZERO_PRIME`.
    """
    if not any(row):
        return zero_is_prime
    smaller = [other for other in rows if other != row and absorbs(row, other)]
    return not covered(row, smaller)


# --------------------------------------------------------------------------
# Таблица наблюдений
# --------------------------------------------------------------------------


@dataclass
class RFSATable:
    """Таблица `NL*`: те же `S`, `S·Σ`, `E`, но условия — про включение."""

    alphabet: str
    teacher: Teacher
    prefixes: list[str] = field(default_factory=lambda: [""])
    suffixes: list[str] = field(default_factory=lambda: [""])
    cache: dict[str, bool] = field(default_factory=dict)
    zero: str = ZERO_COMPOSED

    def ask(self, word: str) -> bool:
        if word not in self.cache:
            self.cache[word] = self.teacher.member(word)
        return self.cache[word]

    def row(self, prefix: str) -> Row:
        return tuple(self.ask(prefix + suffix) for suffix in self.suffixes)

    def extended(self) -> list[str]:
        """`S·Σ` без тех слов, что уже лежат в `S`."""
        found: list[str] = []
        for prefix in self.prefixes:
            for letter in self.alphabet:
                word = prefix + letter
                if word not in self.prefixes and word not in found:
                    found.append(word)
        return found

    # ------------------------------------------------------------- условия

    def unclosed(self) -> str | None:
        """Слово из `S·Σ`, чья строка не накрывается строками `S`.

        Накрытие — это объединение, поэтому проверять достаточно
        объединение строк `S`, **поглощаемых** искомой: всё, что не
        поглощается, в дизъюнкцию внести лишнюю единицу.
        """
        known = [self.row(prefix) for prefix in self.prefixes]
        for word in self.extended():
            row = self.row(word)
            below = [other for other in known if absorbs(row, other)]
            if not covered(row, below):
                return word
        return None

    def inconsistency(self) -> str | None:
        """Столбец `γvₖ`, которого не хватает для непротиворечивости.

        Ищется пара `uᵢ ⊑ uⱼ` из `S`, у которой продолжение одной и той же
        буквой включение ломает: `uᵢγ` даёт единицу там, где `uⱼγ` даёт
        ноль. Этот столбец и добавляется, с приписанной слева буквой.
        """
        for first in self.prefixes:
            for second in self.prefixes:
                if first == second:
                    continue
                if not absorbs(self.row(second), self.row(first)):
                    continue
                for letter in self.alphabet:
                    left = self.row(first + letter)
                    right = self.row(second + letter)
                    if absorbs(right, left):
                        continue
                    for index, (mine, yours) in enumerate(zip(left, right)):
                        if mine and not yours:
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
                break  # столбец уже есть — добавлять нечего
            self.suffixes.append(column)
            added += 1
        return added

    # -------------------------------------------------------------- вывод

    def representatives(self) -> dict[Row, str]:
        """Кратчайшее слово `S` на каждую различную строку."""
        found: dict[Row, str] = {}
        for prefix in sorted(self.prefixes, key=lambda w: (len(w), w)):
            found.setdefault(self.row(prefix), prefix)
        return found

    def primes(self) -> list[str]:
        """Состояния НКА: кратчайшие базисные слова из `S`."""
        names = self.representatives()
        rows = list(names)
        return [
            names[row]
            for row in sorted(rows, key=lambda r: (len(names[r]), names[r]))
            if is_prime(row, rows, self.zero == ZERO_PRIME)
        ]

    def initial(self) -> list[str]:
        """Начальные: строки, поглощаемые строкой `ε`."""
        empty = self.row("")
        return [word for word in self.primes() if absorbs(empty, self.row(word))]

    def to_nfa(self) -> NFA:
        """Построить НКА по таблице — конструкция со слайда 18.

        Начальных состояний у RFSA может быть несколько, а `tfl.automata.NFA`
        держит одно. Как и советует лекция, они стягиваются в одно: заводится
        свежая вершина `→` с ε-переходами в каждое начальное. Число состояний
        самого RFSA берётся из `primes()`, а не из `len(nfa.states)`.
        """
        primes = self.primes()
        empty = self.suffixes.index("")
        finals = {word for word in primes if self.row(word)[empty]}

        delta: dict[tuple[State, str], frozenset[State]] = {}
        for word in primes:
            for letter in self.alphabet:
                target = self.row(word + letter)
                reachable = frozenset(
                    other for other in primes if absorbs(target, self.row(other))
                )
                if reachable:
                    delta[(word, letter)] = reachable

        initial = self.initial()
        if len(initial) == 1:
            return NFA(frozenset(self.alphabet), initial[0], frozenset(finals), delta)
        eps = {START: frozenset(initial)}
        return NFA(frozenset(self.alphabet), START, frozenset(finals), delta, eps)

    def markdown(self) -> str:
        """Таблица в формате запроса об эквивалентности со слайда 7.

        Базисные строки помечены звёздочкой: в `L*` состояния — все
        различные строки, в `NL*` — только базисные, и глазами это должно
        быть видно.
        """
        names = self.representatives()
        rows = list(names)
        prime_words = set(self.primes())
        header = " | ".join(_shown(suffix) for suffix in self.suffixes)
        lines = [f"         | {header}", "-" * (11 + len(header))]
        for group, title in ((self.prefixes, "S"), (self.extended(), "S·Σ")):
            lines.append(f"[{title}]")
            for prefix in group:
                values = " | ".join("1" if value else "0" for value in self.row(prefix))
                mark = "*" if prefix in prime_words else " "
                lines.append(f"{mark}{_shown(prefix):>8}| {values}")
        lines.append("")
        lines.append(f"* — базисные строки ({len(prime_words)} из {len(rows)})")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NFALearningResult:
    """Итог обучения: НКА, таблица и счёт запросов."""

    nfa: NFA
    table: RFSATable
    states: int
    rounds: int
    counterexamples: tuple[str, ...]
    membership_queries: int
    equivalence_queries: int
    distinct_words: int
    converged: bool
    reason: str = ""

    def summary(self) -> str:
        state = "выучен" if self.converged else f"НЕ выучен ({self.reason})"
        return (
            f"НКА {state}: {self.states} состояний, раундов {self.rounds}, "
            f"запросов о принадлежности {self.membership_queries} "
            f"(различных слов {self.distinct_words}), "
            f"об эквивалентности {self.equivalence_queries}"
        )


def learn_nfa(
    teacher: Teacher,
    alphabet: str,
    strategy: str = SUFFIXES,
    max_rounds: int = 60,
    zero: str = ZERO_COMPOSED,
) -> NFALearningResult:
    """Алгоритм `NL*`: выучить НКА запросами к учителю.

    Канва та же, что у `L*` (слайд 16), и обе стратегии обработки
    контрпримера из issue #31 работают здесь без изменений. Отличаются
    только условия на таблицу и построение автомата.

    Гипотеза перед запросом об эквивалентности **детерминизируется** —
    прямое указание слайда 11.

    **Стратегия префиксов для `NL*` не работает** при стандартном прочтении
    базисности (`ZERO_COMPOSED`), и это не дефект реализации: она не
    добавляет столбцов, поэтому строка `ε` может остаться нулевой, а такая
    строка не базисна — начальных состояний не остаётся вовсе и гипотеза
    задаёт пустой язык. Учитель возвращает один и тот же контрпример
    бесконечно. Застой распознаётся и называется, а не сводится к молчаливому
    упору в `max_rounds`.
    """
    if strategy not in (SUFFIXES, PREFIXES):
        raise ValueError(f"стратегия должна быть «{SUFFIXES}» либо «{PREFIXES}»")
    if zero not in (ZERO_COMPOSED, ZERO_PRIME):
        raise ValueError(f"прочтение должно быть «{ZERO_COMPOSED}» либо «{ZERO_PRIME}»")

    table = RFSATable(alphabet, teacher, zero=zero)
    seen: list[str] = []
    for round_number in range(1, max_rounds + 1):
        while True:
            table.close()
            if table.make_consistent() == 0:
                break

        hypothesis = table.to_nfa()
        witness = teacher.equivalent(hypothesis.determinize())
        if witness is None:
            return NFALearningResult(
                hypothesis,
                table,
                len(table.primes()),
                round_number,
                tuple(seen),
                teacher.membership_queries,
                teacher.equivalence_queries,
                len(table.cache),
                True,
                "",
            )
        seen.append(witness)
        grew = 0
        if strategy == SUFFIXES:
            for index in range(len(witness) + 1):
                suffix = witness[index:]
                if suffix not in table.suffixes:
                    table.suffixes.append(suffix)
                    grew += 1
        else:
            for index in range(len(witness) + 1):
                prefix = witness[:index]
                if prefix not in table.prefixes:
                    table.prefixes.append(prefix)
                    grew += 1
        if grew == 0:
            return NFALearningResult(
                hypothesis,
                table,
                len(table.primes()),
                round_number,
                tuple(seen),
                teacher.membership_queries,
                teacher.equivalence_queries,
                len(table.cache),
                False,
                f"застой: контрпример «{_shown(witness)}» таблицу не меняет",
            )

    hypothesis = table.to_nfa()
    return NFALearningResult(
        hypothesis,
        table,
        len(table.primes()),
        max_rounds,
        tuple(seen),
        teacher.membership_queries,
        teacher.equivalence_queries,
        len(table.cache),
        False,
        "лимит раундов",
    )


# --------------------------------------------------------------------------
# Канонический остаточный автомат — эталон для сверки
# --------------------------------------------------------------------------


def _residual(dfa: DFA, state: State) -> DFA:
    """Вычет `u⁻¹L` как автомат: тот же граф, другое начальное состояние."""
    return DFA(dfa.alphabet, state, dfa.finals, dfa.delta)


def _union(dfa: DFA, states) -> DFA:
    """Объединение вычетов набора состояний."""
    states = list(states)
    if not states:
        return DFA(dfa.alphabet, START, frozenset(), {})
    eps = {START: frozenset(states)}
    delta = {key: frozenset([value]) for key, value in dfa.delta.items()}
    return NFA(dfa.alphabet, START, dfa.finals, delta, eps).determinize()


def _included(inner: DFA, outer: DFA) -> bool:
    return difference(inner, outer).is_empty()


def residual_primes(dfa: DFA) -> list[State]:
    """Базисные вычеты языка: те, что не собираются объединением меньших.

    Состояния минимального ДКА — это в точности различные вычеты `u⁻¹L`.
    Вычет базисный, если он не равен объединению строго вложенных в него.
    Из базисных и состоит канонический RFSA.
    """
    minimal = dfa.minimize().trim()
    states = sorted(minimal.states, key=repr)
    languages = {state: _residual(minimal, state) for state in states}

    primes: list[State] = []
    for state in states:
        smaller = [
            other
            for other in states
            if other != state
            and _included(languages[other], languages[state])
            and not _included(languages[state], languages[other])
        ]
        if not _included(languages[state], _union(minimal, smaller)):
            primes.append(state)
    return primes


def canonical_rfsa(dfa: DFA) -> NFA:
    """Канонический остаточный автомат языка — эталон, к которому идёт `NL*`.

    Строится **не** обучением, а прямо по определению, поэтому годится
    для сверки: `NL*` обязан выучить автомат с тем же числом состояний
    и тем же языком.
    """
    minimal = dfa.minimize().trim()
    primes = residual_primes(dfa)
    if not primes:
        return NFA(dfa.alphabet, START, frozenset(), {})

    languages = {state: _residual(minimal, state) for state in minimal.states}
    whole = _residual(minimal, minimal.start)

    delta: dict[tuple[State, str], frozenset[State]] = {}
    for state in primes:
        for letter in sorted(minimal.alphabet):
            following = minimal.delta.get((state, letter))
            if following is None:
                continue
            target = languages[following]
            reachable = frozenset(
                other for other in primes if _included(languages[other], target)
            )
            if reachable:
                delta[(state, letter)] = reachable

    finals = frozenset(state for state in primes if state in minimal.finals)
    initial = [state for state in primes if _included(languages[state], whole)]
    if len(initial) == 1:
        return NFA(minimal.alphabet, initial[0], finals, delta)
    return NFA(minimal.alphabet, START, finals, delta, {START: frozenset(initial)})
