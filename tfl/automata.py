"""Конечные автоматы: НКА, ДКА, детерминизация, минимизация, эквивалентность.

Два независимых пути «регулярка → ДКА»:

    parse(r) --thompson--> НКА --determinize--> ДКА --minimize--> мин. ДКА
    parse(r) -----------brzozowski_dfa-------> ДКА --minimize--> мин. ДКА

Они не должны совпасть «по построению» — они получаются совсем разными
алгоритмами. Поэтому их совпадение (и по языку, и по числу состояний
после минимизации) — настоящая проверка, а не тавтология.

Состояния — любые хешируемые объекты. `relabel()` приводит их к числам,
когда нужен читаемый вывод.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

from tfl import regex as rx
from tfl.words import iter_words

__all__ = [
    "NFA",
    "DFA",
    "thompson",
    "brzozowski_dfa",
    "dfa_of",
    "equivalent",
    "counterexample",
]

State = Hashable
TRAP = "⊥"


# --------------------------------------------------------------------------
# НКА
# --------------------------------------------------------------------------


@dataclass
class NFA:
    """Недетерминированный автомат с ε-переходами."""

    alphabet: frozenset[str]
    start: State
    finals: frozenset[State]
    delta: dict[tuple[State, str], frozenset[State]] = field(default_factory=dict)
    eps: dict[State, frozenset[State]] = field(default_factory=dict)

    @property
    def states(self) -> set[State]:
        found: set[State] = {self.start, *self.finals}
        for (src, _), dsts in self.delta.items():
            found.add(src)
            found |= set(dsts)
        for src, dsts in self.eps.items():
            found.add(src)
            found |= set(dsts)
        return found

    def eps_closure(self, states: Iterable[State]) -> frozenset[State]:
        """Замыкание множества состояний по ε-переходам."""
        stack = list(states)
        seen = set(stack)
        while stack:
            cur = stack.pop()
            for nxt in self.eps.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    def step(self, states: Iterable[State], char: str) -> frozenset[State]:
        """Множество состояний после чтения символа (с ε-замыканием)."""
        moved: set[State] = set()
        for st in states:
            moved |= self.delta.get((st, char), frozenset())
        return self.eps_closure(moved)

    def accepts(self, word: str) -> bool:
        current = self.eps_closure([self.start])
        for ch in word:
            current = self.step(current, ch)
            if not current:
                return False
        return bool(current & self.finals)

    def determinize(self) -> DFA:
        """Построение подмножеств. Состояния ДКА — frozenset состояний НКА."""
        start = self.eps_closure([self.start])
        delta: dict[tuple[State, str], State] = {}
        finals: set[State] = set()
        seen = {start}
        queue = deque([start])
        alphabet = sorted(self.alphabet)
        while queue:
            cur = queue.popleft()
            if cur & self.finals:
                finals.add(cur)
            for ch in alphabet:
                nxt = self.step(cur, ch)
                if not nxt:
                    continue  # ловушку добавит complete(), если понадобится
                delta[(cur, ch)] = nxt
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return DFA(frozenset(self.alphabet), start, frozenset(finals), delta)

    def to_dot(self, name: str = "NFA") -> str:
        return _to_dot(
            name,
            self.states,
            self.start,
            self.finals,
            [
                (src, dst, ch)
                for (src, ch), dsts in self.delta.items()
                for dst in dsts
            ]
            + [(src, dst, "ε") for src, dsts in self.eps.items() for dst in dsts],
        )


# --------------------------------------------------------------------------
# ДКА
# --------------------------------------------------------------------------


@dataclass
class DFA:
    """Детерминированный автомат. `delta` частична: отсутствие перехода
    означает переход в ловушку. `complete()` делает ловушку явной."""

    alphabet: frozenset[str]
    start: State
    finals: frozenset[State]
    delta: dict[tuple[State, str], State] = field(default_factory=dict)

    @property
    def states(self) -> set[State]:
        found: set[State] = {self.start, *self.finals}
        for (src, _), dst in self.delta.items():
            found.add(src)
            found.add(dst)
        return found

    def __len__(self) -> int:
        return len(self.states)

    # -- работа со словами --------------------------------------------------

    def run(self, word: str) -> State | None:
        """Состояние после чтения слова; None — если провалились в ловушку."""
        cur: State | None = self.start
        for ch in word:
            if cur is None:
                return None
            cur = self.delta.get((cur, ch))
        return cur

    def accepts(self, word: str) -> bool:
        return self.run(word) in self.finals

    # -- структурные преобразования ----------------------------------------

    def complete(self) -> DFA:
        """Сделать функцию переходов тотальной, добавив состояние-ловушку."""
        states = self.states
        missing = [
            (st, ch)
            for st in states
            for ch in self.alphabet
            if (st, ch) not in self.delta
        ]
        if not missing:
            return self
        delta = dict(self.delta)
        for st, ch in missing:
            delta[(st, ch)] = TRAP
        for ch in self.alphabet:
            delta[(TRAP, ch)] = TRAP
        return DFA(self.alphabet, self.start, self.finals, delta)

    def reachable(self) -> set[State]:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            cur = queue.popleft()
            for ch in self.alphabet:
                nxt = self.delta.get((cur, ch))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def coreachable(self) -> set[State]:
        """Состояния, из которых достижимо хотя бы одно финальное."""
        back: dict[State, set[State]] = {}
        for (src, _), dst in self.delta.items():
            back.setdefault(dst, set()).add(src)
        seen = set(self.finals)
        queue = deque(self.finals)
        while queue:
            cur = queue.popleft()
            for prev in back.get(cur, ()):
                if prev not in seen:
                    seen.add(prev)
                    queue.append(prev)
        return seen

    def trim(self) -> DFA:
        """Убрать недостижимые и тупиковые состояния (ловушку в том числе)."""
        keep = self.reachable() & self.coreachable()
        if self.start not in keep:
            # язык пуст
            return DFA(self.alphabet, self.start, frozenset(), {})
        delta = {
            (src, ch): dst
            for (src, ch), dst in self.delta.items()
            if src in keep and dst in keep
        }
        return DFA(self.alphabet, self.start, frozenset(self.finals & keep), delta)

    def relabel(self, offset: int = 0) -> DFA:
        """Переименовать состояния в 0,1,2,… в порядке обхода в ширину.

        Порядок детерминирован, поэтому имена состояний воспроизводимы
        от запуска к запуску — иначе отчёт нельзя было бы сверять.
        """
        order: dict[State, int] = {self.start: offset}
        queue = deque([self.start])
        while queue:
            cur = queue.popleft()
            for ch in sorted(self.alphabet):
                nxt = self.delta.get((cur, ch))
                if nxt is not None and nxt not in order:
                    order[nxt] = len(order) + offset
                    queue.append(nxt)
        for st in sorted(self.states, key=repr):
            if st not in order:
                order[st] = len(order) + offset
        delta = {
            (order[src], ch): order[dst] for (src, ch), dst in self.delta.items()
        }
        finals = frozenset(order[st] for st in self.finals)
        return DFA(self.alphabet, order[self.start], finals, delta)

    def minimize(self, keep_trap: bool = False) -> DFA:
        """Минимизация разбиением (Мур): итеративное дробление классов.

        `keep_trap` решает судьбу состояния-ловушки, и это не косметика.

        * `False` (по умолчанию) — ловушка убирается. Именно такой автомат
          рисуют в отчёте: «за отсутствие состояния-ловушки в минимальных
          автоматах штрафа не будет» (`2022/tasks/2022_RK1_data.pdf`).
        * `True` — ловушка остаётся. Классы Майхилла–Нероуда — это классы
          Σ*/≡_L, и слова, не продолжаемые до языка, образуют полноценный
          класс. Если его выбросить, число классов окажется на единицу
          меньше правильного.

        Расхождение ровно в эту единицу — типичная ошибка в отчётах,
        поэтому две величины разделены явно.
        """
        full = self.complete()
        reach = full.reachable()
        finals = full.finals & reach
        nonfinals = reach - finals
        blocks = [b for b in (finals, nonfinals) if b]
        alphabet = sorted(full.alphabet)

        while True:
            index = {st: i for i, block in enumerate(blocks) for st in block}
            new_blocks: list[set[State]] = []
            changed = False
            for block in blocks:
                buckets: dict[tuple[int, ...], set[State]] = {}
                for st in block:
                    sig = tuple(index[full.delta[(st, ch)]] for ch in alphabet)
                    buckets.setdefault(sig, set()).add(st)
                if len(buckets) > 1:
                    changed = True
                new_blocks.extend(buckets.values())
            blocks = new_blocks
            if not changed:
                break

        rep = {st: min(block, key=repr) for block in blocks for st in block}
        delta = {
            (rep[st], ch): rep[full.delta[(st, ch)]]
            for st in reach
            for ch in alphabet
        }
        minimal = DFA(
            full.alphabet,
            rep[full.start],
            frozenset(rep[st] for st in finals),
            delta,
        )
        if not keep_trap:
            minimal = minimal.trim()
        return minimal.relabel()

    def class_count(self) -> int:
        """Число классов эквивалентности Майхилла–Нероуда (с ловушкой)."""
        return len(self.minimize(keep_trap=True))

    # -- булевы операции ----------------------------------------------------

    def complement(self) -> DFA:
        full = self.complete()
        return DFA(
            full.alphabet,
            full.start,
            frozenset(full.states - full.finals),
            dict(full.delta),
        )

    def to_dot(self, name: str = "DFA") -> str:
        return _to_dot(
            name,
            self.states,
            self.start,
            self.finals,
            [(src, dst, ch) for (src, ch), dst in self.delta.items()],
        )

    # -- проверки -----------------------------------------------------------

    def shortest_word(self) -> str | None:
        """Кратчайшее принимаемое слово; None — язык пуст."""
        queue = deque([(self.start, "")])
        seen = {self.start}
        while queue:
            state, word = queue.popleft()
            if state in self.finals:
                return word
            for ch in sorted(self.alphabet):
                nxt = self.delta.get((state, ch))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, word + ch))
        return None

    def is_empty(self) -> bool:
        return self.shortest_word() is None


# --------------------------------------------------------------------------
# Произведение и эквивалентность
# --------------------------------------------------------------------------


def product(left: DFA, right: DFA, accept: Callable[[bool, bool], bool]) -> DFA:
    """Произведение автоматов; `accept(l_final, r_final)` задаёт операцию.

    Алфавит берётся объединением: сравнивать автоматы над разными алфавитами
    иначе некорректно — слово, невозможное для одного, должно им отвергаться,
    а не выпадать из рассмотрения.
    """
    alphabet = frozenset(left.alphabet | right.alphabet)
    a = DFA(alphabet, left.start, left.finals, dict(left.delta)).complete()
    b = DFA(alphabet, right.start, right.finals, dict(right.delta)).complete()

    start = (a.start, b.start)
    delta: dict[tuple[State, str], State] = {}
    finals: set[State] = set()
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        sa, sb = cur
        if accept(sa in a.finals, sb in b.finals):
            finals.add(cur)
        for ch in sorted(alphabet):
            nxt = (a.delta[(sa, ch)], b.delta[(sb, ch)])
            delta[(cur, ch)] = nxt
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return DFA(alphabet, start, frozenset(finals), delta)


def intersection(left: DFA, right: DFA) -> DFA:
    return product(left, right, lambda x, y: x and y)


def union(left: DFA, right: DFA) -> DFA:
    return product(left, right, lambda x, y: x or y)


def difference(left: DFA, right: DFA) -> DFA:
    return product(left, right, lambda x, y: x and not y)


def counterexample(left: DFA, right: DFA) -> str | None:
    """Кратчайшее слово, различающее языки; None — языки равны.

    Именно этот вид ответа нужен в отчёте: не «не эквивалентны», а конкретное
    слово, на котором распознаватели расходятся.
    """
    return product(left, right, lambda x, y: x != y).shortest_word()


def equivalent(left: DFA, right: DFA) -> bool:
    return counterexample(left, right) is None


# --------------------------------------------------------------------------
# Регулярка → автомат
# --------------------------------------------------------------------------


def thompson(node: rx.Node) -> NFA:
    """Построение Томпсона: ε-НКА с одним входом и одним выходом."""
    counter = iter(range(10**9))
    delta: dict[tuple[State, str], frozenset[State]] = {}
    eps: dict[State, frozenset[State]] = {}

    def add_eps(src: State, dst: State) -> None:
        eps[src] = eps.get(src, frozenset()) | {dst}

    def add_move(src: State, ch: str, dst: State) -> None:
        key = (src, ch)
        delta[key] = delta.get(key, frozenset()) | {dst}

    def build(n: rx.Node) -> tuple[State, State]:
        start, end = next(counter), next(counter)
        if isinstance(n, rx.Empty):
            pass  # переходов нет: язык пуст
        elif isinstance(n, rx.Eps):
            add_eps(start, end)
        elif isinstance(n, rx.Sym):
            add_move(start, n.char, end)
        elif isinstance(n, rx.Alt):
            for arg in n.args:
                s, e = build(arg)
                add_eps(start, s)
                add_eps(e, end)
        elif isinstance(n, rx.Cat):
            prev = start
            for arg in n.args:
                s, e = build(arg)
                add_eps(prev, s)
                prev = e
            add_eps(prev, end)
        elif isinstance(n, rx.Star):
            s, e = build(n.arg)
            add_eps(start, s)
            add_eps(start, end)
            add_eps(e, s)
            add_eps(e, end)
        else:  # pragma: no cover
            raise TypeError(f"неизвестный узел AST: {type(n).__name__}")
        return start, end

    start, end = build(node)
    return NFA(node.alphabet(), start, frozenset({end}), delta, eps)


def brzozowski_dfa(node: rx.Node, alphabet: Iterable[str] | None = None) -> DFA:
    """ДКА по производным Брзозовского: состояние = класс производной.

    Автомат получается сразу детерминированным и почти минимальным
    (минимален с точностью до тождеств, заложенных в умные конструкторы).
    """
    letters = sorted(set(alphabet) if alphabet is not None else node.alphabet())
    delta: dict[tuple[State, str], State] = {}
    finals: set[State] = set()
    seen = {node}
    queue = deque([node])
    while queue:
        cur = queue.popleft()
        if cur.nullable:
            finals.add(cur)
        for ch in letters:
            nxt = cur.derivative(ch)
            if isinstance(nxt, rx.Empty):
                continue  # ловушка остаётся неявной
            delta[(cur, ch)] = nxt
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return DFA(frozenset(letters), node, frozenset(finals), delta)


def dfa_of(pattern: str | rx.Node, alphabet: Iterable[str] | None = None) -> DFA:
    """Минимальный ДКА по регулярке (или уже разобранному AST)."""
    node = rx.parse(pattern) if isinstance(pattern, str) else pattern
    return brzozowski_dfa(node, alphabet).minimize()


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------


def _to_dot(
    name: str,
    states: Iterable[State],
    start: State,
    finals: Iterable[State],
    edges: Iterable[tuple[State, State, str]],
) -> str:
    """Graphviz. Параллельные рёбра склеиваются в одну подпись `a,b,c`."""
    label = {st: str(i) for i, st in enumerate(sorted(states, key=repr))}
    finals = set(finals)
    merged: dict[tuple[State, State], list[str]] = {}
    for src, dst, ch in edges:
        merged.setdefault((src, dst), []).append(ch)

    lines = [f"digraph {name} {{", "  rankdir=LR;", '  __start [shape=none,label=""];']
    for st in sorted(states, key=repr):
        shape = "doublecircle" if st in finals else "circle"
        lines.append(f'  q{label[st]} [shape={shape},label="{label[st]}"];')
    lines.append(f"  __start -> q{label[start]};")
    for (src, dst), chars in sorted(merged.items(), key=lambda kv: repr(kv[0])):
        text = ",".join(sorted(set(chars)))
        lines.append(f'  q{label[src]} -> q{label[dst]} [label="{text}"];')
    lines.append("}")
    return "\n".join(lines)


def disagreements(
    left: Callable[[str], bool],
    right: Callable[[str], bool],
    alphabet: Iterable[str],
    max_len: int = 12,
    limit: int = 10,
) -> list[str]:
    """Слова до длины `max_len`, на которых два распознавателя расходятся.

    Универсальный оракул уровня L1: годится для любой пары «гипотеза против
    эталона», не только для двух ДКА.
    """
    bad: list[str] = []
    for word in iter_words(alphabet, max_len):
        if left(word) != right(word):
            bad.append(word)
            if len(bad) >= limit:
                break
    return bad
