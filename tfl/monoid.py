"""Трансформационный и синтаксический моноиды по ДКА.

Обслуживает `RK1-*` (задача 2: +1 балл сверх НКА за моноид — issue #6),
`EXAM-3` (моноид встречается третьим вопросом билета) и `LAB-1`
(«SRS над трансформационным моноидом»: правила переписывания, которые
строит этот модуль, и есть искомая система).

Что здесь считается. Элемент трансформационного моноида ДКА — это
**функция переходов слова**: куда слово переводит каждое состояние.
Слова с одинаковой функцией неразличимы автоматом, поэтому моноид конечен
(не больше `|Q|^|Q|` элементов) и вычисляется обходом в ширину.

Побочный продукт обхода — система переписывания строк. Когда у нового слова
`w` функция совпала с функцией уже найденного представителя `r`, рождается
правило `w → r`, а `w` дальше не продолжается: любое его продолжение
переписывается через `r`. Представители остаются неприводимыми, поэтому
получается именно та SRS, которую просят в ЛР1.

Тонкость, за которую снимают баллы. **Синтаксический моноид языка — это
трансформационный моноид минимального полного ДКА**, и ловушку выбрасывать
нельзя: без неё функции становятся частичными, а моноид — другим. Поэтому
`transition_monoid` работает по полному автомату, а `syntactic_monoid`
сперва минимизирует с сохранением ловушки. Функция `transition_monoid`
с `complete=False` оставлена для случаев, когда моноид нужен именно
по предъявленному автомату, и тогда это **не** синтаксический моноид.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

from tfl.automata import DFA, State

__all__ = [
    "Element",
    "Monoid",
    "transition_monoid",
    "syntactic_monoid",
]


@dataclass(frozen=True)
class Element:
    """Элемент моноида: кратчайший представитель и его функция переходов.

    `image` — кортеж пар «состояние → куда переводит слово», отсортированный
    по исходному состоянию. Именно он задаёт равенство элементов; слово
    `word` — лишь выбранное имя.
    """

    word: str
    image: tuple[tuple[State, State], ...]

    @property
    def mapping(self) -> dict[State, State]:
        return dict(self.image)

    def is_identity(self) -> bool:
        return all(src == dst for src, dst in self.image)

    def __str__(self) -> str:
        name = self.word or "ε"
        moves = ", ".join(f"{src}→{dst}" for src, dst in self.image)
        return f"[{name}] {{{moves}}}"


@dataclass
class Monoid:
    """Трансформационный моноид ДКА вместе с задающей его SRS."""

    dfa: DFA
    states: tuple[State, ...]
    elements: tuple[Element, ...]
    rules: tuple[tuple[str, str], ...]
    complete: bool

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterator[Element]:
        return iter(self.elements)

    # -- вычисления ---------------------------------------------------------

    def image_of(self, word: str) -> tuple[tuple[State, State], ...] | None:
        """Функция переходов слова; `None` — если автомат неполон и слово
        выводит какое-то состояние за пределы функции переходов."""
        pairs = []
        for src in self.states:
            cur: State | None = src
            for ch in word:
                if cur is None:
                    break
                cur = self.dfa.delta.get((cur, ch))
            if cur is None:
                return None
            pairs.append((src, cur))
        return tuple(pairs)

    def element_of(self, word: str) -> Element | None:
        """Элемент, которому принадлежит слово."""
        image = self.image_of(word)
        if image is None:
            return None
        for element in self.elements:
            if element.image == image:
                return element
        return None

    def reduce(self, word: str) -> str | None:
        """Представитель класса слова — нормальная форма относительно SRS."""
        element = self.element_of(word)
        return None if element is None else element.word

    def accepting(self) -> tuple[Element, ...]:
        """Элементы, переводящие стартовое состояние в финальное.

        Язык автомата — это в точности множество слов, попадающих в эти
        классы; отсюда и берётся распознавание моноидом.
        """
        start, finals = self.dfa.start, self.dfa.finals
        return tuple(e for e in self.elements if e.mapping.get(start) in finals)

    def accepts(self, word: str) -> bool:
        element = self.element_of(word)
        return element is not None and element in self.accepting()

    def is_aperiodic(self) -> bool:
        """Апериодичен ли моноид: `∀x ∃n (xⁿ = xⁿ⁺¹)`.

        По теореме Шютценберже апериодичность равносильна тому, что язык
        бесзвёздочный (star-free). Перебор конечен: степени элемента
        зацикливаются не позже, чем через `|M|` шагов.
        """
        bound = len(self.elements) + 1
        for element in self.elements:
            power = element.word
            for _ in range(bound):
                nxt = self.image_of(power + element.word)
                if nxt == self.image_of(power):
                    break
                power = power + element.word
            else:
                return False
        return True

    # -- предъявление -------------------------------------------------------

    def srs(self) -> str:
        """Система переписывания, задающая моноид, в синтаксисе `tfl.srs`."""
        return "\n".join(f"{lhs} -> {rhs or 'ε'}" for lhs, rhs in self.rules)

    def markdown(self) -> str:
        """Таблица моноида для отчёта: представитель, функция, финальность."""
        accepting = set(self.accepting())
        head = " | ".join(str(s) for s in self.states)
        lines = [
            f"| Представитель | {head} | Принимающий |",
            "|---" * (len(self.states) + 2) + "|",
        ]
        for element in self.elements:
            mapping = element.mapping
            cells = " | ".join(str(mapping[s]) for s in self.states)
            mark = "да" if element in accepting else "нет"
            lines.append(f"| `{element.word or 'ε'}` | {cells} | {mark} |")
        return "\n".join(lines)

    def summary(self) -> str:
        kind = "синтаксический" if self.complete else "трансформационный"
        return (
            f"{kind} моноид: {len(self.elements)} элементов, "
            f"{len(self.rules)} правил переписывания, "
            f"принимающих классов {len(self.accepting())}"
        )


def transition_monoid(dfa: DFA, complete: bool = True) -> Monoid:
    """Построить трансформационный моноид обходом в ширину.

    Слова перебираются в шортлекс-порядке, поэтому представителем класса
    становится кратчайшее слово, а при равной длине — лексикографически
    меньшее. Кандидат отбрасывается, если содержит левую часть уже
    построенного правила: такое слово приводимо, и его продолжения тоже.

    `complete=True` (по умолчанию) сперва делает функцию переходов тотальной.
    Без этого моноид считается по частичным функциям и синтаксическим
    **не является** — см. `syntactic_monoid`.

    >>> from tfl.automata import dfa_of
    >>> m = transition_monoid(dfa_of("(aa)*"))
    >>> len(m)
    2
    >>> m.reduce("aaa")
    'a'
    """
    machine = dfa.complete() if complete else dfa
    states = tuple(sorted(machine.states, key=str))
    alphabet = sorted(machine.alphabet)

    identity = tuple((s, s) for s in states)
    elements = [Element("", identity)]
    seen = {identity: ""}
    rules: list[tuple[str, str]] = []
    queue: deque[tuple[str, tuple[tuple[State, State], ...]]] = deque(
        [("", identity)]
    )

    while queue:
        word, image = queue.popleft()
        mapping = dict(image)
        for ch in alphabet:
            candidate = word + ch
            if any(lhs in candidate for lhs, _ in rules):
                continue
            moved = []
            broken = False
            for src in states:
                nxt = machine.delta.get((mapping[src], ch))
                if nxt is None:
                    broken = True
                    break
                moved.append((src, nxt))
            if broken:
                # частичная функция: слово выводит автомат из области
                # определения, класса для него нет
                continue
            new_image = tuple(moved)
            known = seen.get(new_image)
            if known is None:
                seen[new_image] = candidate
                element = Element(candidate, new_image)
                elements.append(element)
                queue.append((candidate, new_image))
            else:
                rules.append((candidate, known))

    return Monoid(
        dfa=machine,
        states=states,
        elements=tuple(elements),
        rules=tuple(rules),
        complete=complete,
    )


def syntactic_monoid(dfa: DFA) -> Monoid:
    """Синтаксический моноид языка автомата.

    Минимизирует ДКА **с сохранением ловушки** и строит по нему
    трансформационный моноид: по теореме о синтаксическом моноиде это одно
    и то же. Ловушка обязательна — она отвечает за слова, выводящие из языка
    безвозвратно, и без неё классы склеиваются.

    >>> from tfl.automata import dfa_of
    >>> monoid = syntactic_monoid(dfa_of("a*ba*"))
    >>> [element.word or "ε" for element in monoid]
    ['ε', 'b', 'bb']
    >>> print(monoid.srs())
    a -> ε
    bbb -> bb
    """
    return transition_monoid(dfa.minimize(keep_trap=True), complete=True)
