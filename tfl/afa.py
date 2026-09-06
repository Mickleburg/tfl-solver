"""Переключающиеся (альтернирующие) конечные автоматы — ПКА.

Формализм из лекции 6 (`corpus/txt/chat_AFA.txt`): автомат
`A = (Q∃, Q∀, Σ, δ, s, F)` отличается от НКА только тем, что состояния
разделены на два вида. Из состояния `Q∃` достаточно, чтобы **какой-то**
преемник принял остаток; из состояния `Q∀` должны принять **все**.

Зачем это в ЛР2 (issue #40, `corpus/issues/lab2-2025-issue40.md`):

> «Проще всего понять ПКА из тех же lookahead-регулярок, где неявно как раз
> ПКА и строится… В персональных вариантах часто нет такой логики напрямую,
> но бывают **инварианты**, которые выполняются в языке регулярки, вот их
> и используем при построении ПКА.»

Отсюда две конструкции, которыми ПКА и строится на практике:

* `conjunction` — И-ветвление в стартовой вершине: язык распадается
  в конъюнкцию простых условий (её первый пример,
  `^(?= .* a .* $) .* b .* $`);
* `with_lookahead` — **рекурсивный** инвариант: в некоторых состояниях
  читателя дополнительно проверяется условие на весь остаток слова
  (её второй пример, `^((?=(b*ab*ab*)*$)a*b)*$`).

Тонкость, которую пришлось решать оракулом
------------------------------------------

Пустое множество переходов трактуется по-разному в разных местах лекции.
Здесь принято стандартное прочтение: пустая дизъюнкция — ложь, пустая
конъюнкция — истина. То есть заглохшая ветвь из `Q∃` губит прогон,
а заглохшая ветвь из `Q∀` просто исчезает.

Иначе ломается **теорема 5** (дополнение = поменять `Q∃` с `Q∀`
и дополнить `F`): дуальность обязана переводить пустую дизъюнкцию
в пустую конъюнкцию, а «ложь» — в «истину». Проверено исполнением,
см. `tests/test_afa.py`. Из-за этого же в теореме 6 множеством
принимающих состояний НКА взято `2^F`, а не напечатанное в лекции
`2^F \\ {∅}`: набор обязательств, схлопнувшийся в пустой, выполнен.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Hashable, Iterable, Sequence

from tfl.automata import DFA, NFA, State
from tfl.words import iter_words

__all__ = [
    "AFA",
    "Run",
    "conjunction",
    "with_lookahead",
    "disagreements",
]


@dataclass(frozen=True)
class Run:
    """Прогон ПКА — дерево, помеченное состояниями."""

    state: State
    children: tuple["Run", ...] = ()

    def leaves(self) -> list[State]:
        if not self.children:
            return [self.state]
        return [leaf for child in self.children for leaf in child.leaves()]

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        head = f"{pad}{self.state}"
        return "\n".join([head] + [c.render(indent + 1) for c in self.children])


@dataclass
class AFA:
    """Переключающийся автомат.

    `universal` — множество `Q∀`; всё остальное считается `Q∃`.
    `delta` частична: отсутствие ключа означает пустое множество преемников.
    """

    alphabet: frozenset[str]
    start: State
    finals: frozenset[State]
    universal: frozenset[State] = frozenset()
    delta: dict[tuple[State, str], frozenset[State]] = field(default_factory=dict)

    @property
    def states(self) -> set[State]:
        found: set[State] = {self.start, *self.finals, *self.universal}
        for (src, _), dsts in self.delta.items():
            found.add(src)
            found |= set(dsts)
        return found

    def __len__(self) -> int:
        return len(self.states)

    def is_universal(self, state: State) -> bool:
        return state in self.universal

    def successors(self, state: State, char: str) -> frozenset[State]:
        return self.delta.get((state, char), frozenset())

    # ------------------------------------------------------------------
    # Принятие
    # ------------------------------------------------------------------

    def _values(self, word: str) -> list[dict[State, bool]]:
        """Таблица `Acc(q, суффикс с позиции i)` — считается справа налево."""
        states = self.states
        table: list[dict[State, bool]] = [{q: q in self.finals for q in states}]
        for char in reversed(word):
            nxt = table[-1]
            current = {}
            for state in states:
                targets = self.successors(state, char)
                if self.is_universal(state):
                    current[state] = all(nxt[p] for p in targets)
                else:
                    current[state] = any(nxt[p] for p in targets)
            table.append(current)
        table.reverse()
        return table

    def accepts(self, word: str) -> bool:
        return self._values(word)[0][self.start]

    def run(self, word: str) -> Run | None:
        """Принимающий прогон-свидетель или None.

        Из `Q∃` берётся один преемник, у которого остаток принимается;
        из `Q∀` — все, как в определении прогона из лекции.
        """
        table = self._values(word)
        if not table[0][self.start]:
            return None

        def build(state: State, index: int) -> Run:
            if index == len(word):
                return Run(state)
            targets = self.successors(state, word[index])
            if self.is_universal(state):
                return Run(state, tuple(build(p, index + 1) for p in sorted(targets, key=repr)))
            chosen = next(p for p in sorted(targets, key=repr) if table[index + 1][p])
            return Run(state, (build(chosen, index + 1),))

        return build(self.start, 0)

    # ------------------------------------------------------------------
    # Дополнение и перевод в НКА
    # ------------------------------------------------------------------

    def dual(self) -> AFA:
        """Дополнение по теореме 5: поменять `Q∃` с `Q∀`, дополнить `F`."""
        states = self.states
        return AFA(
            alphabet=self.alphabet,
            start=self.start,
            finals=frozenset(states - self.finals),
            universal=frozenset(states - self.universal),
            delta=dict(self.delta),
        )

    def to_nfa(self) -> NFA:
        """Теорема 6: состояния НКА — подмножества `Q`.

        `δ′(X, a) = {X′ ⊆ δ(X, a) | ∀q ∈ X∩Q∀ δ(q,a) ⊆ X′,
        ∀q ∈ X∩Q∃ X′ ∩ δ(q,a) ≠ ∅}`.

        Число состояний растёт как `2^{|Q|}`, и это не изъян реализации:
        такова цена перехода от И-ветвления к обычному недетерминизму.
        """
        start = frozenset({self.start})
        delta: dict[tuple[State, str], frozenset[State]] = {}
        seen = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for char in sorted(self.alphabet):
                targets = self._nfa_targets(current, char)
                if not targets:
                    continue
                delta[(current, char)] = frozenset(targets)
                for target in targets:
                    if target not in seen:
                        seen.add(target)
                        queue.append(target)
        finals = frozenset(X for X in seen if X <= self.finals)
        return NFA(alphabet=self.alphabet, start=start, finals=finals, delta=delta)

    def _nfa_targets(self, current: frozenset, char: str) -> set[frozenset]:
        """Все `X′`, годные как следующий уровень прогона."""
        required: set[State] = set()
        choices: list[frozenset[State]] = []
        for state in current:
            targets = self.successors(state, char)
            if self.is_universal(state):
                required |= targets  # все преемники обязаны быть в X′
            else:
                if not targets:
                    return set()  # пустая дизъюнкция — тупик
                choices.append(targets)

        pool = set()
        for state in current:
            pool |= self.successors(state, char)
        optional = sorted(pool - required, key=repr)

        out: set[frozenset] = set()
        for size in range(len(optional) + 1):
            for extra in combinations(optional, size):
                candidate = frozenset(required | set(extra))
                if all(candidate & choice for choice in choices):
                    out.add(candidate)
        return out

    def to_dfa(self) -> DFA:
        return self.to_nfa().determinize()

    # ------------------------------------------------------------------
    # Вывод
    # ------------------------------------------------------------------

    def to_dot(self, name: str = "AFA") -> str:
        """Точки Граф: `Q∀` — квадраты, `Q∃` — круги, как в лекции."""
        lines = [f"digraph {name} {{", "  rankdir=LR;", '  "" [shape=none];']
        for state in sorted(self.states, key=repr):
            shape = "box" if self.is_universal(state) else "circle"
            peripheries = 2 if state in self.finals else 1
            lines.append(
                f'  "{state}" [shape={shape}, peripheries={peripheries}];'
            )
        lines.append(f'  "" -> "{self.start}";')
        for (src, char), dsts in sorted(self.delta.items(), key=repr):
            for dst in sorted(dsts, key=repr):
                lines.append(f'  "{src}" -> "{dst}" [label="{char}"];')
        lines.append("}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Размеры: ПКА, полученный НКА и ДКА после детерминизации.

        Преподаватель прямо разрешает ПКА не меньше ДКА: «нашли
        нетривиальный инвариант ⇒ построили ПКА, даже если формально
        он будет не меньше ДКА».
        """
        nfa = self.to_nfa()
        dfa = nfa.determinize().minimize()
        return (
            f"ПКА: {len(self)} состояний "
            f"({len(self.universal)} ∀, {len(self) - len(self.universal)} ∃); "
            f"НКА по теореме 6: {len(nfa.states)}; минимальный ДКА: {len(dfa)}"
        )


# --------------------------------------------------------------------------
# Конструкции
# --------------------------------------------------------------------------


def conjunction(parts: Sequence[DFA], alphabet: Iterable[str] | None = None) -> AFA:
    """И-ветвление в стартовой вершине: пересечение языков нескольких ДКА.

    Первый пример преподавателя: `^(?= .* a .* $) .* b .* $` — конъюнкция
    двух ДКА, один проверяет наличие `a`, другой наличие `b`.

    Части обязаны быть **детерминированными**: у состояния `Q∀` все
    преемники обязательны, поэтому смешивать в одном множестве варианты
    выбора одного автомата с ветвями другого нельзя.
    """
    if not parts:
        raise ValueError("нужен хотя бы один автомат")
    letters = frozenset(alphabet) if alphabet else frozenset().union(
        *(part.alphabet for part in parts)
    )

    start = ("⋀", 0)
    delta: dict[tuple[State, str], frozenset[State]] = {}
    finals: set[State] = set()
    completed = [part.complete() for part in parts]

    for index, complete in enumerate(completed):
        for state in complete.states:
            tagged = (index, state)
            if state in complete.finals:
                finals.add(tagged)
            for char in sorted(letters):
                target = complete.delta.get((state, char))
                if target is not None:
                    delta[(tagged, char)] = frozenset({(index, target)})

    for char in sorted(letters):
        targets = set()
        for index, complete in enumerate(completed):
            target = complete.delta.get((complete.start, char))
            if target is None:
                targets = set()
                break
            targets.add((index, target))
        if targets:
            delta[(start, char)] = frozenset(targets)

    if all(part.start in part.finals for part in completed):
        finals.add(start)

    return AFA(
        alphabet=letters,
        start=start,
        finals=frozenset(finals),
        universal=frozenset({start}),
        delta=delta,
    )


def with_lookahead(
    reader: DFA, checker: DFA, at: Iterable[State] | None = None
) -> AFA:
    """Рекурсивный инвариант: в состояниях `at` проверить остаток слова.

    Это перевод lookahead-регулярки `(?=C$)` в ПКА: в помеченных
    состояниях читателя ставится И-ветвление, одна ветвь продолжает
    читать по `reader`, вторая проверяет **весь остаток** автоматом
    `checker` с его начала.

    По умолчанию `at` — стартовое состояние читателя: именно так устроен
    второй пример преподавателя `^((?=(b*ab*ab*)*$)a*b)*$`, где по `b`
    происходит возврат в стартовую вершину.
    """
    reader = reader.complete()
    checker = checker.complete()
    marked = frozenset(at) if at is not None else frozenset({reader.start})
    letters = frozenset(reader.alphabet) | frozenset(checker.alphabet)

    delta: dict[tuple[State, str], frozenset[State]] = {}
    finals: set[State] = set()

    for state in reader.states:
        tagged = ("r", state)
        accepts_here = state in reader.finals
        if state in marked:
            accepts_here = accepts_here and checker.start in checker.finals
        if accepts_here:
            finals.add(tagged)
        for char in sorted(letters):
            target = reader.delta.get((state, char))
            if target is None:
                continue
            targets = {("r", target)}
            if state in marked:
                probe = checker.delta.get((checker.start, char))
                if probe is None:
                    continue  # проверка заглохла — ветвь невозможна
                targets.add(("c", probe))
            delta[(tagged, char)] = frozenset(targets)

    for state in checker.states:
        tagged = ("c", state)
        if state in checker.finals:
            finals.add(tagged)
        for char in sorted(letters):
            target = checker.delta.get((state, char))
            if target is not None:
                delta[(tagged, char)] = frozenset({("c", target)})

    return AFA(
        alphabet=letters,
        start=("r", reader.start),
        finals=frozenset(finals),
        universal=frozenset(("r", state) for state in marked),
        delta=delta,
    )


def disagreements(
    automaton: AFA,
    reference: Callable[[str], bool] | DFA,
    max_len: int = 8,
    limit: int = 5,
) -> list[str]:
    """Слова, где ПКА расходится с эталоном — предикатом или ДКА."""
    check = reference.accepts if isinstance(reference, DFA) else reference
    out: list[str] = []
    for word in iter_words(sorted(automaton.alphabet), max_len):
        if automaton.accepts(word) != check(word):
            out.append(word)
            if len(out) >= limit:
                break
    return out
