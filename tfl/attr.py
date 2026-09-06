"""Атрибутные грамматики — третья задача каждого варианта РК2.

Задача формулируется так: «язык, определяемый следующей атрибутной
грамматикой», дальше идут правила вида

    S → S S   ;  S2.attr < S1.attr, S0.attr := S1.attr − S2.attr
    S → b A   ;  S.attr := A.attr
    A → b A b ;  A0.attr := A1.attr + 2
    A → ε     ;  A.attr := 0

и требуется описать язык, а затем ответить, регулярен ли он и КС ли.

Нотация занятий:

* `X0` — вхождение символа `X` в **левой** части правила, `X1`, `X2`, … —
  вхождения в правой части слева направо. Голое `X` означает левую часть,
  если `X` совпадает с ней, иначе единственное вхождение справа.
* `:=` — присваивание атрибута, всё остальное — **условие** на вывод.
  Слово принадлежит языку, если у него есть дерево вывода, в котором
  выполнены условия **всех** применённых правил.
* Атрибут, вычисляемый в `X0`, — синтезируемый; вычисляемый в `Xi` при
  `i ≥ 1` — наследуемый. Оба встречаются в вариантах (наследуемые —
  например, там, где условие `T.inh_attr == 0` стоит на правиле `T → ε`).

Что здесь есть и чего нет
-------------------------

`AttrGrammar.accepts` разбирает слово **исчерпывающе**: перебираются все
деревья вывода с данной кроной, а не первое попавшееся. Поэтому ответ
«нет» — это ответ, а не «не нашлось». Исключение одно: грамматики
с циклами по ε- и цепным правилам; такой цикл обнаруживается, обход
помечается усечённым, и вердикт становится «не выяснено».

Языка целиком модуль не знает и знать не может: принадлежность слова
языку атрибутной грамматики разрешима (перебор конечного леса разбора),
а вот равенство двух таких языков — уже нет. Поэтому `words()` — это
перебор до заданной длины, а не описание языка.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from tfl.cfg import CFG, EPSILON_TOKENS, Production
from tfl.verdict import Verdict, proved, refuted, unknown
from tfl.words import iter_words

__all__ = [
    "Ref",
    "Rule",
    "AttrGrammar",
    "Node",
    "Evaluation",
    "parse_attr_grammar",
]

ARROWS = ("-->", "->", "→", "::=", "=>")
LHS_SLOT = -1

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


# ---------------------------------------------------------------------------
# Выражения и условия
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ref:
    """Ссылка на атрибут вхождения символа: `S1.attr`."""

    symbol: str
    index: int | None  # None — вхождение не указано явно
    attribute: str

    def __str__(self) -> str:
        index = "" if self.index is None else str(self.index)
        return f"{self.symbol}{index}.{self.attribute}"


@dataclass(frozen=True)
class Const:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class BinOp:
    op: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[object, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass(frozen=True)
class Compare:
    op: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True)
class Logic:
    op: str  # "&" или "|"
    parts: tuple[object, ...]

    def __str__(self) -> str:
        glue = " & " if self.op == "&" else " ∨ "
        return "(" + glue.join(str(p) for p in self.parts) + ")"


ARITH = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a // b if b else None,
    "%": lambda a, b: a % b if b else None,
}

COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}

FUNCTIONS = {
    "min": min,
    "max": max,
    "abs": lambda *args: abs(args[0]),
}


def refs_in(node: object) -> set[Ref]:
    """Все ссылки на атрибуты внутри выражения или условия."""
    if isinstance(node, Ref):
        return {node}
    if isinstance(node, Const):
        return set()
    if isinstance(node, (BinOp, Compare)):
        return refs_in(node.left) | refs_in(node.right)
    if isinstance(node, Call):
        return set().union(*(refs_in(a) for a in node.args)) if node.args else set()
    if isinstance(node, Logic):
        return set().union(*(refs_in(p) for p in node.parts))
    raise TypeError(f"неизвестный узел выражения: {node!r}")


# ---------------------------------------------------------------------------
# Правила
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """Правило вместе с семантикой."""

    lhs: str
    rhs: tuple[str, ...]
    assignments: tuple[tuple[Ref, object], ...] = ()
    conditions: tuple[object, ...] = ()

    def __str__(self) -> str:
        body = " ".join(self.rhs) if self.rhs else "ε"
        parts = [f"{ref} := {expr}" for ref, expr in self.assignments]
        parts += [str(c) for c in self.conditions]
        semantics = f" ; {', '.join(parts)}" if parts else ""
        return f"{self.lhs} → {body}{semantics}"

    @property
    def production(self) -> Production:
        return Production(self.lhs, self.rhs)

    def slot_of(self, ref: Ref) -> int:
        """Куда указывает ссылка: `LHS_SLOT` или позиция в правой части.

        Голое имя без индекса означает левую часть, если совпадает с ней,
        иначе — единственное вхождение справа. Если вхождений несколько,
        а индекс не указан, это ошибка записи, а не повод угадывать.
        """
        positions = [i for i, s in enumerate(self.rhs) if s == ref.symbol]
        if ref.index is None:
            if ref.symbol == self.lhs:
                return LHS_SLOT
            if len(positions) == 1:
                return positions[0]
            raise ValueError(
                f"{ref}: в правиле «{self}» вхождений {ref.symbol} — "
                f"{len(positions)}, нужен явный индекс"
            )
        if ref.index == 0:
            if ref.symbol != self.lhs:
                raise ValueError(f"{ref}: индекс 0 — это левая часть, а там {self.lhs}")
            return LHS_SLOT
        if ref.index > len(positions):
            raise ValueError(f"{ref}: в правиле «{self}» нет такого вхождения")
        return positions[ref.index - 1]

    def synthesized(self) -> set[str]:
        """Атрибуты, вычисляемые в левой части."""
        return {r.attribute for r, _ in self.assignments if self.slot_of(r) == LHS_SLOT}

    def inherited(self) -> set[str]:
        """Атрибуты, вычисляемые в правой части, — наследуемые."""
        return {r.attribute for r, _ in self.assignments if self.slot_of(r) != LHS_SLOT}


# ---------------------------------------------------------------------------
# Деревья вывода
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """Узел дерева вывода. У листа-терминала `rule is None`."""

    symbol: str
    rule: Rule | None = None
    children: tuple["Node", ...] = ()

    @property
    def word(self) -> str:
        if self.rule is None:
            return self.symbol
        return "".join(child.word for child in self.children)

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def size(self) -> int:
        """Число применённых правил."""
        return sum(1 for node in self.walk() if node.rule is not None)

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.rule is None:
            return f"{pad}{self.symbol}"
        head = f"{pad}{self.symbol} → {' '.join(self.rule.rhs) or 'ε'}"
        return "\n".join([head] + [c.render(indent + 1) for c in self.children])


@dataclass
class Evaluation:
    """Результат вычисления атрибутов на дереве."""

    tree: Node
    values: dict[tuple[int, str], int]
    ok: bool
    undefined: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def of(self, node: Node, attribute: str) -> int | None:
        return self.values.get((id(node), attribute))


# ---------------------------------------------------------------------------
# Грамматика
# ---------------------------------------------------------------------------


@dataclass
class AttrGrammar:
    """Атрибутная грамматика: правила плюс семантика."""

    start: str
    rules: tuple[Rule, ...]
    _span_cycle: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for rule in self.rules:  # ранняя проверка ссылок
            for ref, _ in rule.assignments:
                rule.slot_of(ref)
            for cond in rule.conditions:
                for ref in refs_in(cond):
                    rule.slot_of(ref)

    def __str__(self) -> str:
        return "\n".join(str(rule) for rule in self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def nonterminals(self) -> frozenset[str]:
        return frozenset(rule.lhs for rule in self.rules)

    @property
    def terminals(self) -> frozenset[str]:
        symbols = {s for rule in self.rules for s in rule.rhs}
        return frozenset(symbols - self.nonterminals)

    @property
    def attributes(self) -> frozenset[str]:
        names = {ref.attribute for rule in self.rules for ref, _ in rule.assignments}
        for rule in self.rules:
            for cond in rule.conditions:
                names |= {ref.attribute for ref in refs_in(cond)}
        return frozenset(names)

    def rules_for(self, symbol: str) -> list[Rule]:
        return [rule for rule in self.rules if rule.lhs == symbol]

    def has_span_cycle(self) -> bool:
        """Есть ли вывод `A ⇒⁺ αAβ`, где `α` и `β` пусты.

        Только такой цикл способен зациклить разбор: нетерминал вызывается
        сам на себе на том же отрезке слова. Если цикла нет, обход
        конечен, и обрыв по защите означает лишь тупиковую ветвь,
        а не потерю ответа — это и позволяет отвечать «нет», а не
        «не выяснено».
        """
        if self._span_cycle is None:
            grammar = self.cfg()
            nullable = grammar.nullable()
            live = grammar.reachable() & self.nonterminals
            edges: dict[str, set[str]] = {n: set() for n in live}
            for rule in self.rules:
                if rule.lhs not in live:
                    continue
                for i, symbol in enumerate(rule.rhs):
                    if symbol not in self.nonterminals:
                        continue
                    others = rule.rhs[:i] + rule.rhs[i + 1 :]
                    if all(s in nullable for s in others):
                        edges[rule.lhs].add(symbol)
            colour: dict[str, int] = {}

            def visit(node: str) -> bool:
                colour[node] = 1
                for nxt in edges.get(node, ()):
                    if colour.get(nxt) == 1:
                        return True
                    if colour.get(nxt) is None and visit(nxt):
                        return True
                colour[node] = 2
                return False

            self._span_cycle = any(
                colour.get(n) is None and visit(n) for n in sorted(live)
            )
        return self._span_cycle

    def cfg(self) -> CFG:
        """Нижележащая КС-грамматика — язык **без** учёта условий.

        Первый шаг разбора задачи `RK2-C`: выписать, что порождает
        грамматика, если про атрибуты забыть. Условия потом только
        отсеивают часть выводов, поэтому язык атрибутной грамматики
        всегда вложен в язык этой КС-грамматики.
        """
        return CFG(self.start, tuple(rule.production for rule in self.rules))

    # ------------------------------------------------------------------
    # Вычисление атрибутов
    # ------------------------------------------------------------------

    def evaluate(self, tree: Node) -> Evaluation:
        """Вычислить атрибуты на дереве и проверить условия.

        Порядок вычисления не задаётся заранее: присваивания применяются
        по мере готовности аргументов, пока что-то меняется. Это работает
        и для синтезируемых, и для наследуемых атрибутов, а на циклической
        зависимости просто останавливается, оставив атрибут неопределённым.
        """
        values: dict[tuple[int, str], int] = {}
        nodes = [node for node in tree.walk() if node.rule is not None]

        def slot_node(node: Node, ref: Ref) -> Node:
            slot = node.rule.slot_of(ref)
            return node if slot == LHS_SLOT else node.children[slot]

        def value(node: Node, expr: object) -> int | None:
            if isinstance(expr, Const):
                return expr.value
            if isinstance(expr, Ref):
                return values.get((id(slot_node(node, expr)), expr.attribute))
            if isinstance(expr, BinOp):
                left = value(node, expr.left)
                right = value(node, expr.right)
                if left is None or right is None:
                    return None
                return ARITH[expr.op](left, right)
            if isinstance(expr, Call):
                args = [value(node, a) for a in expr.args]
                if any(a is None for a in args):
                    return None
                return FUNCTIONS[expr.name](*args)
            raise TypeError(f"не выражение: {expr!r}")

        changed = True
        while changed:
            changed = False
            for node in nodes:
                for ref, expr in node.rule.assignments:
                    target = (id(slot_node(node, ref)), ref.attribute)
                    if target in values:
                        continue
                    computed = value(node, expr)
                    if computed is not None:
                        values[target] = computed
                        changed = True

        def truth(node: Node, cond: object) -> bool | None:
            if isinstance(cond, Compare):
                left = value(node, cond.left)
                right = value(node, cond.right)
                if left is None or right is None:
                    return None
                return COMPARISONS[cond.op](left, right)
            if isinstance(cond, Logic):
                parts = [truth(node, p) for p in cond.parts]
                if cond.op == "&":
                    if any(p is False for p in parts):
                        return False
                    return None if any(p is None for p in parts) else True
                if any(p is True for p in parts):
                    return True
                return None if any(p is None for p in parts) else False
            raise TypeError(f"не условие: {cond!r}")

        undefined: list[str] = []
        failed: list[str] = []
        for node in nodes:
            for cond in node.rule.conditions:
                verdict = truth(node, cond)
                if verdict is None:
                    undefined.append(f"{node.rule}: {cond}")
                elif not verdict:
                    failed.append(f"{node.rule}: {cond}")

        return Evaluation(tree, values, not failed and not undefined, undefined, failed)

    # ------------------------------------------------------------------
    # Разбор
    # ------------------------------------------------------------------

    def _forest(
        self, symbol: str, word: str, start: int, end: int, guard: frozenset, budget: list[int]
    ) -> tuple[list[Node], bool]:
        """Все деревья для `symbol`, дающие `word[start:end]`.

        `guard` защищает от циклов по ε- и цепным правилам: если тот же
        нетерминал вызывается на том же отрезке, обход обрывается,
        и результат помечается усечённым.
        """
        if symbol not in self.nonterminals:
            if end - start == 1 and word[start] == symbol:
                return [Node(symbol)], False
            return [], False

        key = (symbol, start, end)
        if key in guard:
            return [], self.has_span_cycle()
        guard = guard | {key}

        out: list[Node] = []
        truncated = False
        for rule in self.rules_for(symbol):
            for split, cut in self._split(rule.rhs, word, start, end, guard, budget):
                truncated |= cut
                if split is not None:
                    out.append(Node(symbol, rule, tuple(split)))
                if budget[0] <= 0:
                    return out, True
        return out, truncated

    def _split(
        self,
        symbols: Sequence[str],
        word: str,
        start: int,
        end: int,
        guard: frozenset,
        budget: list[int],
    ) -> Iterator[tuple[list[Node] | None, bool]]:
        """Разложить `word[start:end]` по символам правой части."""
        if not symbols:
            yield ([] if start == end else None), False
            return
        head, rest = symbols[0], symbols[1:]
        lower = start if head in self.nonterminals else start + 1
        upper = end if head in self.nonterminals else min(start + 1, end)
        for middle in range(lower, upper + 1):
            budget[0] -= 1
            if budget[0] <= 0:
                yield None, True
                return
            heads, cut_head = self._forest(head, word, start, middle, guard, budget)
            if not heads:
                if cut_head:
                    yield None, True
                continue
            for tail, cut_tail in self._split(rest, word, middle, end, guard, budget):
                if tail is None:
                    yield None, cut_head or cut_tail
                    continue
                for node in heads:
                    yield [node] + tail, cut_head or cut_tail

    def derivations(self, word: str, budget: int = 200_000) -> tuple[list[Node], bool]:
        """Все деревья вывода слова — и признак того, что обход усечён."""
        counter = [budget]
        trees, truncated = self._forest(
            self.start, word, 0, len(word), frozenset(), counter
        )
        return trees, truncated or counter[0] <= 0

    def synthesized_only(self) -> bool:
        """Все ли атрибуты синтезируемые (вычисляются в левой части)."""
        return not any(rule.inherited() for rule in self.rules)

    def _combinations(
        self,
        symbols: Sequence[str],
        word: str,
        start: int,
        end: int,
        table: dict,
    ) -> Iterator[list[tuple | None]]:
        """Разложить отрезок по правой части, подставляя known-значения."""
        if not symbols:
            if start == end:
                yield []
            return
        head, rest = symbols[0], symbols[1:]
        if head not in self.nonterminals:
            if start < end and word[start] == head:
                for tail in self._combinations(rest, word, start + 1, end, table):
                    yield [None] + tail
            return
        for middle in range(start, end + 1):
            for values in table.get((head, start, middle), ()):
                for tail in self._combinations(rest, word, middle, end, table):
                    yield [values] + tail

    def _fixpoint(self, word: str, rounds: int = 64, cap: int = 4096) -> Verdict:
        """Разбор через неподвижную точку по значениям атрибутов.

        Перебор деревьев спотыкается о циклы вида `S′ → T S′` при
        `T ⇒ ε`: деревьев бесконечно много. Но **значений** атрибутов
        на отрезке обычно конечное число, и множество достижимых
        наборов насыщается. Поэтому здесь считается не «какие деревья»,
        а «какие наборы атрибутов достижимы для нетерминала на отрезке».

        Годится только для синтезируемых атрибутов: у наследуемого
        значение приходит сверху, и снизу вверх его не восстановить.
        """
        names = sorted(self.attributes)
        position = {name: i for i, name in enumerate(names)}
        length = len(word)
        table: dict[tuple[str, int, int], set[tuple]] = {}
        total = 0

        for _ in range(rounds):
            changed = False
            for rule in self.rules:
                for start in range(length + 1):
                    for end in range(start, length + 1):
                        for combo in self._combinations(
                            rule.rhs, word, start, end, table
                        ):
                            values = self._apply(rule, combo, names, position)
                            if values is None:
                                continue
                            bucket = table.setdefault((rule.lhs, start, end), set())
                            if values not in bucket:
                                bucket.add(values)
                                total += 1
                                changed = True
                                if total > cap:
                                    return unknown(
                                        f"наборы атрибутов не насытились "
                                        f"за {cap} значений"
                                    )
            if not changed:
                break
        else:
            return unknown(f"неподвижная точка не достигнута за {rounds} проходов")

        if table.get((self.start, 0, length)):
            return proved(f"{word!r} выводится с выполнением условий")
        return refuted(f"{word!r} не выводится с выполнением условий")

    def _apply(
        self,
        rule: Rule,
        combo: Sequence[tuple | None],
        names: Sequence[str],
        position: dict[str, int],
    ) -> tuple | None:
        """Применить правило к набору значений детей: вернуть значения левой части."""
        own: list[int | None] = [None] * len(names)

        def read(ref: Ref) -> int | None:
            slot = rule.slot_of(ref)
            index = position[ref.attribute]
            if slot == LHS_SLOT:
                return own[index]
            child = combo[slot]
            return None if child is None else child[index]

        def value(expr: object) -> int | None:
            if isinstance(expr, Const):
                return expr.value
            if isinstance(expr, Ref):
                return read(expr)
            if isinstance(expr, BinOp):
                left, right = value(expr.left), value(expr.right)
                if left is None or right is None:
                    return None
                return ARITH[expr.op](left, right)
            if isinstance(expr, Call):
                args = [value(a) for a in expr.args]
                return None if any(a is None for a in args) else FUNCTIONS[expr.name](*args)
            raise TypeError(f"не выражение: {expr!r}")

        for _ in range(len(rule.assignments) + 1):
            progress = False
            for ref, expr in rule.assignments:
                index = position[ref.attribute]
                if own[index] is not None:
                    continue
                computed = value(expr)
                if computed is not None:
                    own[index] = computed
                    progress = True
            if not progress:
                break

        def truth(cond: object) -> bool | None:
            if isinstance(cond, Compare):
                left, right = value(cond.left), value(cond.right)
                if left is None or right is None:
                    return None
                return COMPARISONS[cond.op](left, right)
            parts = [truth(p) for p in cond.parts]
            if cond.op == "&":
                if any(p is False for p in parts):
                    return False
                return None if any(p is None for p in parts) else True
            if any(p is True for p in parts):
                return True
            return None if any(p is None for p in parts) else False

        for cond in rule.conditions:
            if truth(cond) is not True:
                return None
        return tuple(own)

    def accepts(self, word: str, budget: int = 200_000) -> Verdict:
        """Принадлежит ли слово языку грамматики.

        Перебираются **все** деревья с такой кроной, поэтому «нет» —
        это доказательство, а не «не нашлось».

        Если в грамматике есть цикл по ε- и цепным правилам, деревьев
        бесконечно много; тогда для чисто синтезируемых атрибутов
        включается счёт по неподвижной точке (`_fixpoint`), который
        цикла не боится. Свидетеля-дерева он не даёт, зато даёт ответ.
        Если же атрибуты наследуемые и цикл есть, вердикт честно
        становится «не выяснено».
        """
        if self.has_span_cycle() and self.synthesized_only():
            return self._fixpoint(word)
        trees, truncated = self.derivations(word, budget)
        for tree in trees:
            result = self.evaluate(tree)
            if result.ok:
                return proved(f"{word!r} выводится с выполнением условий", tree)
        if truncated:
            return unknown(f"обход разбора {word!r} оборван", None)
        if not trees:
            return refuted(f"{word!r} не выводится и без учёта условий", None)
        return refuted(f"ни одно из {len(trees)} деревьев не проходит условия", trees[0])

    def __contains__(self, word: str) -> bool:
        """Удобство для перебора: «не выяснено» считается «нет»."""
        return self.accepts(word).value is True

    def words(self, max_len: int, budget: int = 200_000) -> list[str]:
        """Слова языка до данной длины — перебором по алфавиту.

        Это не описание языка, а конечный срез: годится, чтобы сверить
        гипотезу с оракулом, и не годится как ответ на задачу.
        """
        alphabet = "".join(sorted(self.terminals))
        return [w for w in iter_words(alphabet, max_len) if self.accepts(w, budget).value]

    def language(self, name: str = ""):
        """Обернуть в `tfl.lang.Language`, чтобы работали `restrict` и таблицы."""
        from tfl.lang import Language

        return Language(
            lambda w: self.accepts(w).value is True,
            "".join(sorted(self.terminals)),
            name or f"L({self.start})",
        )

    # ------------------------------------------------------------------
    # Что считает атрибут
    # ------------------------------------------------------------------

    def check_attribute(
        self,
        symbol: str,
        attribute: str,
        formula: Callable[[str], int],
        max_len: int = 6,
        budget: int = 200_000,
    ) -> Verdict:
        """Проверить гипотезу «атрибут символа равен `formula(кроны)`».

        Первый шаг разбора задачи `RK2-C` — понять, что именно считает
        атрибут: число букв, число применений правила, глубину. Гипотеза
        проверяется на всех поддеревьях, выводимых из `symbol`, до
        заданной длины кроны.

        Работает только для **синтезируемых** атрибутов: у наследуемого
        значение зависит от контекста, и в отрыве от целого дерева оно
        не определено.
        """
        inherited = {a for rule in self.rules for a in rule.inherited()}
        if attribute in inherited:
            return unknown(
                f"{symbol}.{attribute} наследуемый — вне дерева значения нет"
            )

        # Цикл может быть в части грамматики, до которой из `symbol`
        # не добраться, поэтому стартовый символ подменяется целиком.
        sub = AttrGrammar(symbol, self.rules)
        alphabet = "".join(sorted(self.terminals))
        seen = 0
        for word in iter_words(alphabet, max_len):
            trees, truncated = sub.derivations(word, budget)
            if truncated:
                return unknown(f"обход разбора {word!r} оборван")
            for tree in trees:
                result = self.evaluate(tree)
                if not result.ok:
                    continue
                actual = result.of(tree, attribute)
                if actual is None:
                    return unknown(f"{word!r}: {symbol}.{attribute} не вычислен", tree)
                seen += 1
                if actual != formula(word):
                    return refuted(
                        f"{word!r}: {symbol}.{attribute} = {actual}, "
                        f"а формула даёт {formula(word)}",
                        word,
                    )
        if not seen:
            return unknown(f"из {symbol} не выведено ни одного слова длины ≤ {max_len}")
        return proved(f"совпало на {seen} выводах длины ≤ {max_len}")

    def disagreements(
        self,
        predicate: Callable[[str], bool],
        max_len: int = 6,
        limit: int = 5,
        budget: int = 200_000,
    ) -> list[str]:
        """Слова, где гипотеза о языке расходится с грамматикой."""
        alphabet = "".join(sorted(self.terminals))
        out: list[str] = []
        for word in iter_words(alphabet, max_len):
            if (self.accepts(word, budget).value is True) != predicate(word):
                out.append(word)
                if len(out) >= limit:
                    break
        return out


# ---------------------------------------------------------------------------
# Разбор текста
# ---------------------------------------------------------------------------

_REF = re.compile(r"([A-Z][′']*)(\d*)\s*\.\s*([A-Za-z_][A-Za-z_0-9]*)")
_NUMBER = re.compile(r"\d+")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class _Parser:
    """Рекурсивный спуск по выражению или условию."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self, *options: str) -> str | None:
        self.skip()
        for option in options:
            if self.text.startswith(option, self.pos):
                return option
        return None

    def eat(self, token: str) -> None:
        self.skip()
        if not self.text.startswith(token, self.pos):
            raise ValueError(f"ожидалось {token!r} в {self.text[self.pos:]!r}")
        self.pos += len(token)

    def at_end(self) -> bool:
        self.skip()
        return self.pos >= len(self.text)

    # -- условия --------------------------------------------------------

    def condition(self) -> object:
        parts = [self.conjunction()]
        while (op := self.peek("∨", "||", "|", " or ")) is not None:
            self.eat(op)
            parts.append(self.conjunction())
        return parts[0] if len(parts) == 1 else Logic("|", tuple(parts))

    def conjunction(self) -> object:
        parts = [self.comparison()]
        while (op := self.peek("&&", "&", "∧", " and ")) is not None:
            self.eat(op)
            parts.append(self.comparison())
        return parts[0] if len(parts) == 1 else Logic("&", tuple(parts))

    def comparison(self) -> object:
        if self.peek("(") and self._parenthesised_condition():
            self.eat("(")
            inner = self.condition()
            self.eat(")")
            return inner
        left = self.expression()
        for token, op in (
            ("==", "=="), ("!=", "!="), ("≠", "!="), ("<=", "<="), ("≤", "<="),
            (">=", ">="), ("≥", ">="), ("=", "=="), ("<", "<"), (">", ">"),
        ):
            if self.peek(token):
                self.eat(token)
                return Compare(op, left, self.expression())
        raise ValueError(f"условие без сравнения: {self.text!r}")

    def _parenthesised_condition(self) -> bool:
        """Отличить `(a + b) > c` от `(a > b) & c`: заглянуть до скобки."""
        depth = 0
        for i in range(self.pos, len(self.text)):
            ch = self.text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    rest = self.text[i + 1 :].lstrip()
                    return not rest.startswith(
                        ("==", "!=", "≠", "<", ">", "≤", "≥", "=")
                    )
            elif depth == 1 and ch in "<>=≠≤≥":
                return True
        return False

    # -- выражения ------------------------------------------------------

    def expression(self) -> object:
        node = self.term()
        while (op := self.peek("+", "-")) is not None:
            self.eat(op)
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> object:
        node = self.factor()
        while (op := self.peek("*", "/", "%")) is not None:
            self.eat(op)
            node = BinOp(op, node, self.factor())
        return node

    def factor(self) -> object:
        self.skip()
        if self.peek("("):
            self.eat("(")
            inner = self.expression()
            self.eat(")")
            return inner
        match = _NUMBER.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            return Const(int(match.group()))
        match = _REF.match(self.text, self.pos)
        if match:
            self.pos = match.end()
            symbol, index, attribute = match.groups()
            return Ref(symbol, int(index) if index else None, attribute)
        match = _NAME.match(self.text, self.pos)
        if match and match.group() in FUNCTIONS:
            self.pos = match.end()
            name = match.group()
            self.eat("(")
            args = [self.expression()]
            while self.peek(","):
                self.eat(",")
                args.append(self.expression())
            self.eat(")")
            return Call(name, tuple(args))
        raise ValueError(f"не разобрано выражение: {self.text[self.pos:]!r}")


def _split_top(text: str, separator: str = ",") -> list[str]:
    """Разбить по запятым верхнего уровня — внутри `min(...)` не режем."""
    out, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == separator and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    out.append("".join(current))
    return [part.strip() for part in out if part.strip()]


def _tokenize_rhs(body: str, nonterminals: set[str]) -> tuple[str, ...]:
    """Разбить правую часть на символы: заглавная со штрихами — нетерминал."""
    if body.strip() in EPSILON_TOKENS:
        return ()
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch.isspace():
            i += 1
            continue
        if ch in EPSILON_TOKENS and ch:
            i += 1
            continue
        if ch.isupper():
            j = i + 1
            while j < len(body) and body[j] in "'′":
                j += 1
            out.append(body[i:j])
            i = j
            continue
        out.append(ch)
        i += 1
    return tuple(out)


def parse_attr_grammar(text: str, start: str | None = None) -> AttrGrammar:
    """Разобрать атрибутную грамматику в нотации курса.

    Правило и семантика разделены `;`, элементы семантики — запятыми
    верхнего уровня. Пустая правая часть записывается как `ε` или `eps`.
    Альтернативы через `|` в правой части допускаются только без
    семантики — иначе непонятно, к какой из них она относится.

    >>> g = parse_attr_grammar("S -> a S b ; S0.n := S1.n + 1\\nS -> ε ; S.n := 0")
    >>> g.accepts("aabb").value
    True
    >>> g.accepts("aab").value
    False
    """
    raw_rules: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        line = line.strip().translate(_MINUS).translate(_SUBSCRIPTS)
        if not line or line.startswith("#"):
            continue
        head_part, _, semantics = line.partition(";")
        for arrow in ARROWS:
            if arrow in head_part:
                left, right = head_part.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {line!r}")
        lhs = left.strip()
        alternatives = _split_top(right, "|")
        if len(alternatives) > 1 and semantics.strip():
            raise ValueError(
                f"альтернативы с общей семантикой неоднозначны: {line!r}"
            )
        for alternative in alternatives or [""]:
            raw_rules.append((lhs, alternative, semantics))

    nonterminals = {lhs for lhs, _, _ in raw_rules}
    rules: list[Rule] = []
    for lhs, body, semantics in raw_rules:
        rhs = _tokenize_rhs(body, nonterminals)
        assignments: list[tuple[Ref, object]] = []
        conditions: list[object] = []
        for item in _split_top(semantics):
            if ":=" in item:
                target, _, expr = item.partition(":=")
                parser = _Parser(target.strip())
                ref = parser.factor()
                if not isinstance(ref, Ref) or not parser.at_end():
                    raise ValueError(f"слева от := должен быть атрибут: {item!r}")
                value_parser = _Parser(expr.strip())
                assignments.append((ref, value_parser.expression()))
                if not value_parser.at_end():
                    raise ValueError(f"лишнее в присваивании: {item!r}")
            else:
                parser = _Parser(item)
                conditions.append(parser.condition())
                if not parser.at_end():
                    raise ValueError(f"лишнее в условии: {item!r}")
        rules.append(Rule(lhs, rhs, tuple(assignments), tuple(conditions)))

    return AttrGrammar(start or rules[0].lhs, tuple(rules))
