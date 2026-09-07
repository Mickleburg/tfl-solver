"""Позиционный автомат (автомат Глушкова) и редукция НКА.

Автомат Глушкова строится по регулярному выражению без ε-переходов и имеет
ровно `n + 1` состояние, где `n` — число вхождений символов в выражение.
Это заметно меньше автомата Томпсона (тот даёт до `2n` состояний и кучу
ε-переходов) и потому служит хорошей отправной точкой, когда в ЛР2 просят
«возможно малый НКА».

Тот же автомат в курсе называется **позиционным** и используется в ЛР3 для
построения LR(0)-аппроксимации КС-грамматики сверху — так что модуль ещё
пригодится.

Конструкция стандартная: выражение линеаризуется (каждому вхождению символа
даётся свой номер), после чего считаются множества `first`, `last` и `follow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tfl import regex as rx
from tfl.automata import NFA, DFA, State, counterexample
from tfl.verdict import Verdict, proved, unknown

__all__ = [
    "glushkov",
    "small_nfa",
    "Linearization",
    "linearize",
    "reduce_nfa",
    "Conflict",
    "nondeterminism",
    "is_one_unambiguous",
]


@dataclass
class Linearization:
    """Разметка вхождений символов номерами 1..n."""

    symbol: dict[int, str] = field(default_factory=dict)
    first: set[int] = field(default_factory=set)
    last: set[int] = field(default_factory=set)
    follow: dict[int, set[int]] = field(default_factory=dict)
    nullable: bool = False

    @property
    def size(self) -> int:
        return len(self.symbol)


def linearize(node: rx.Node) -> Linearization:
    """Посчитать first / last / follow для линеаризованного выражения."""
    lin = Linearization()
    counter = 0

    def walk(n: rx.Node) -> tuple[set[int], set[int], bool]:
        """Вернуть (first, last, nullable) для поддерева."""
        nonlocal counter
        if isinstance(n, rx.Empty):
            return set(), set(), False
        if isinstance(n, rx.Eps):
            return set(), set(), True
        if isinstance(n, rx.Sym):
            counter += 1
            pos = counter
            lin.symbol[pos] = n.char
            lin.follow.setdefault(pos, set())
            return {pos}, {pos}, False
        if isinstance(n, rx.Alt):
            first: set[int] = set()
            last: set[int] = set()
            nullable = False
            for arg in n.args:
                f, l, e = walk(arg)
                first |= f
                last |= l
                nullable = nullable or e
            return first, last, nullable
        if isinstance(n, rx.Cat):
            # Свёртка слева бинарной конкатенацией, начиная с нейтрального ε.
            # Бинарный шаг сам обеспечивает транзитивность follow через
            # аннулируемые сомножители, поэтому особых случаев не нужно.
            first, last, nullable = set(), set(), True
            for arg in n.args:
                f, l, e = walk(arg)
                for p in last:  # хвост накопленного тянется к началу текущего
                    lin.follow[p] |= f
                first = first | f if nullable else first
                last = l | last if e else l
                nullable = nullable and e
            return first, last, nullable
        if isinstance(n, rx.Star):
            f, l, _ = walk(n.arg)
            for p in l:
                lin.follow[p] |= f
            return f, l, True
        raise TypeError(f"неизвестный узел AST: {type(n).__name__}")  # pragma: no cover

    lin.first, lin.last, lin.nullable = walk(node)
    return lin


def glushkov(node: rx.Node) -> NFA:
    """Позиционный автомат: ε-переходов нет, состояний ровно n + 1."""
    lin = linearize(node)
    start = 0
    delta: dict[tuple[State, str], frozenset[State]] = {}

    def add(src: State, char: str, dst: State) -> None:
        key = (src, char)
        delta[key] = delta.get(key, frozenset()) | {dst}

    for pos in lin.first:
        add(start, lin.symbol[pos], pos)
    for src, targets in lin.follow.items():
        for dst in targets:
            add(src, lin.symbol[dst], dst)

    finals = set(lin.last)
    if lin.nullable:
        finals.add(start)
    return NFA(node.alphabet(), start, frozenset(finals), delta, {})


# --------------------------------------------------------------------------
# Редукция НКА
# --------------------------------------------------------------------------


def _right_language(nfa: NFA, state: State) -> DFA:
    """Язык, распознаваемый автоматом, если стартовать из данного состояния."""
    clone = NFA(nfa.alphabet, state, nfa.finals, nfa.delta, nfa.eps)
    return clone.determinize().minimize()


def reduce_nfa(nfa: NFA) -> NFA:
    """Склеить состояния с одинаковым правым языком.

    Это самая безопасная из редукций НКА: если из двух состояний распознаётся
    в точности один и тот же язык, их можно отождествить, не меняя языка
    автомата в целом.

    Минимальность полученного НКА **не гарантируется** — задача о минимальном
    НКА PSPACE-полна, и уже поэтому в задании просят «возможно малый», а не
    минимальный. Оценку снизу даёт `tfl.myhill.extended_fooling_set`;
    если она совпала с числом состояний, минимальность доказана.
    """
    states = sorted(nfa.states, key=repr)
    languages = {st: _right_language(nfa, st) for st in states}

    # Классы состояний с совпадающими правыми языками.
    representative: dict[State, State] = {}
    classes: list[State] = []
    for st in states:
        for rep in classes:
            if counterexample(languages[st], languages[rep]) is None:
                representative[st] = rep
                break
        else:
            classes.append(st)
            representative[st] = st

    if len(classes) == len(states):
        return nfa

    delta: dict[tuple[State, str], frozenset[State]] = {}
    for (src, char), dsts in nfa.delta.items():
        key = (representative[src], char)
        merged = frozenset(representative[d] for d in dsts)
        delta[key] = delta.get(key, frozenset()) | merged
    eps: dict[State, frozenset[State]] = {}
    for src, dsts in nfa.eps.items():
        key = representative[src]
        merged = frozenset(representative[d] for d in dsts)
        eps[key] = eps.get(key, frozenset()) | merged

    return NFA(
        nfa.alphabet,
        representative[nfa.start],
        frozenset(representative[f] for f in nfa.finals),
        delta,
        eps,
    )


def small_nfa(node: rx.Node) -> NFA:
    """Возможно малый НКА: позиционный автомат плюс склейка состояний."""
    return reduce_nfa(glushkov(node))


# --------------------------------------------------------------------------
# 1-однозначность: детерминирован ли автомат Глушкова
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Conflict:
    """Место недетерминированного разбора в позиционном автомате.

    `position` — номер вхождения буквы, после которого возникает выбор
    (`0` — стартовое состояние). `targets` — вхождения, между которыми
    выбор происходит; они помечены одной и той же буквой `char`.
    """

    position: int
    char: str
    targets: tuple[int, ...]

    def __str__(self) -> str:
        where = "в начале" if self.position == 0 else f"после позиции {self.position}"
        return f"{where} по букве «{self.char}» выбор между {list(self.targets)}"


def nondeterminism(node: rx.Node) -> list[Conflict]:
    """Все позиции регулярки, где автомат Глушкова недетерминирован.

    Ровно то, что просят предъявить в задаче 4 РК1: «указать позиции,
    где происходит недетерминированный разбор».
    """
    lin = linearize(node)
    machine = glushkov(node)
    conflicts = []
    for (src, char), targets in sorted(machine.delta.items(), key=lambda kv: kv[0]):
        if len(targets) > 1:
            conflicts.append(Conflict(src, char, tuple(sorted(targets))))
    assert all(lin.symbol[t] == c.char for c in conflicts for t in c.targets)
    return conflicts


def is_one_unambiguous(node: rx.Node | str) -> Verdict:
    """1-однозначна ли **предъявленная регулярка**.

    Регулярка называется 1-однозначной, если её автомат Глушкова
    детерминирован: читая слово слева направо, всегда понятно, какому
    вхождению буквы соответствует прочитанный символ, без заглядывания
    вперёд.

    Границы вывода. Детерминированность даёт **доказательство**: язык
    1-однозначен, и свидетель — сама регулярка. Обратный вывод неправомерен:
    недетерминированность автомата Глушкова говорит только про эту запись,
    а для языка может найтись другая, 1-однозначная. Поэтому здесь
    «не выяснено», а не «нет» — то же различие, что между свойством
    грамматики и свойством языка.

    >>> is_one_unambiguous("a*b").value
    True
    >>> is_one_unambiguous("(a|b)*a").value is None
    True
    """
    tree = rx.parse(node) if isinstance(node, str) else node
    conflicts = nondeterminism(tree)
    if not conflicts:
        return proved(
            "автомат Глушкова детерминирован, значит регулярка 1-однозначна "
            "и язык 1-однозначен",
            tree,
        )
    return unknown(
        "автомат Глушкова недетерминирован: "
        + "; ".join(str(c) for c in conflicts)
        + ". Про язык это ничего не говорит — 1-однозначной может оказаться "
        "другая запись",
        conflicts,
    )
