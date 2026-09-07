"""Древесные языки: термы, контексты и восходящие древесные автоматы.

Обслуживает `RK1-C`. В условиях РК1 2025 древесные языки стоят минимум
в шести вариантах из четырнадцати («древесный язык логических формул»,
«деревья арифметических выражений», «деревья регулярных выражений»),
и со слов преподавателя их будет больше.

Главное, что здесь важно не перепутать. **Регулярность древесного языка —
не то же самое, что регулярность языка его линейных записей.** Древесный
язык регулярен, если его распознаёт детерминированный восходящий древесный
автомат: состояние приписывается каждому узлу снизу вверх, исходя только
из символа узла и состояний его детей. Роль суффикса при этом играет
**контекст** — дерево с одной дырой, куда подставляется поддерево.

Отсюда и аналог теоремы Майхилла–Нероуда: два дерева неразличимы, если
ни один контекст не разводит их по принадлежности языку. Различили `n`
деревьев — доказано, что классов не меньше `n`, значит и состояний в любом
распознающем автомате не меньше `n`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "Tree",
    "Context",
    "HOLE",
    "Signature",
    "parse_tree",
    "trees",
    "contexts",
    "TreeAutomaton",
    "at_least_classes",
    "disagreements",
]

#: Ранжированный алфавит: символ → арность.
Signature = dict[str, int]

#: Символ дыры в контексте.
HOLE = "□"

State = object


@dataclass(frozen=True)
class Tree:
    """Терм над ранжированным алфавитом. Лист — это узел без детей."""

    symbol: str
    children: tuple["Tree", ...] = ()

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def height(self) -> int:
        return 1 + max((child.height() for child in self.children), default=0)

    def subtrees(self) -> Iterator["Tree"]:
        yield self
        for child in self.children:
            yield from child.subtrees()

    def __str__(self) -> str:
        if not self.children:
            return self.symbol
        inner = ", ".join(str(child) for child in self.children)
        return f"{self.symbol}({inner})"


def parse_tree(text: str) -> Tree:
    """Разобрать запись вида `and(not(P), Q)`.

    >>> str(parse_tree("and(not(P), Q)"))
    'and(not(P), Q)'
    >>> parse_tree("P").children
    ()
    """
    pos = 0
    text = text.strip()

    def parse() -> Tree:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] not in "(),":
            pos += 1
        symbol = text[start:pos].strip()
        if not symbol:
            raise ValueError(f"пустой символ в позиции {start}")
        children: list[Tree] = []
        if pos < len(text) and text[pos] == "(":
            pos += 1
            while True:
                children.append(parse())
                if pos >= len(text):
                    raise ValueError("не хватает закрывающей скобки")
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"неожиданный символ «{text[pos]}»")
        return Tree(symbol, tuple(children))

    tree = parse()
    if pos != len(text):
        raise ValueError(f"лишний текст с позиции {pos}: «{text[pos:]}»")
    return tree


def trees(signature: Signature, max_size: int) -> list[Tree]:
    """Все деревья размера не больше `max_size`, по возрастанию размера.

    >>> [str(t) for t in trees({"a": 0, "f": 1}, 3)]
    ['a', 'f(a)', 'f(f(a))']
    """
    by_size: dict[int, list[Tree]] = {}
    for size in range(1, max_size + 1):
        level: list[Tree] = []
        for symbol, arity in sorted(signature.items()):
            if arity == 0:
                if size == 1:
                    level.append(Tree(symbol))
                continue
            for parts in _compositions(size - 1, arity):
                if any(part not in by_size for part in parts):
                    continue
                for combo in _product([by_size[part] for part in parts]):
                    level.append(Tree(symbol, tuple(combo)))
        by_size[size] = level
    return [t for size in sorted(by_size) for t in by_size[size]]


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Разбиения числа на упорядоченные положительные слагаемые."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _product(groups: Sequence[Sequence[Tree]]) -> Iterator[tuple[Tree, ...]]:
    if not groups:
        yield ()
        return
    for head in groups[0]:
        for tail in _product(groups[1:]):
            yield (head, *tail)


@dataclass(frozen=True)
class Context:
    """Дерево ровно с одной дырой — древесный аналог суффикса."""

    tree: Tree

    def fill(self, subtree: Tree) -> Tree:
        """Подставить дерево в дыру."""

        def walk(node: Tree) -> Tree:
            if node.symbol == HOLE:
                return subtree
            return Tree(node.symbol, tuple(walk(c) for c in node.children))

        return walk(self.tree)

    def __str__(self) -> str:
        return str(self.tree)


def contexts(signature: Signature, max_size: int) -> list[Context]:
    """Все контексты размера не больше `max_size`, включая пустой.

    Пустой контекст — это одна дыра; он проверяет саму принадлежность
    дерева языку, как пустой суффикс в строковом случае.

    >>> [str(c) for c in contexts({"a": 0, "f": 1}, 2)]
    ['□', 'f(□)']
    """
    with_hole = dict(signature)
    with_hole[HOLE] = 0
    result = []
    for tree in trees(with_hole, max_size):
        holes = sum(1 for node in tree.subtrees() if node.symbol == HOLE)
        if holes == 1:
            result.append(Context(tree))
    return result


@dataclass
class TreeAutomaton:
    """Детерминированный восходящий древесный автомат.

    `delta` отображает пару «символ, кортеж состояний детей» в состояние.
    Отсутствие перехода означает ловушку: дерево не принимается.
    """

    signature: Signature
    finals: frozenset[State]
    delta: dict[tuple[str, tuple[State, ...]], State] = field(default_factory=dict)

    @property
    def states(self) -> set[State]:
        found = set(self.finals)
        for (_, children), target in self.delta.items():
            found.update(children)
            found.add(target)
        return found

    def __len__(self) -> int:
        return len(self.states)

    def run(self, tree: Tree) -> State | None:
        """Состояние в корне; `None` — если попали в ловушку."""
        children = []
        for child in tree.children:
            state = self.run(child)
            if state is None:
                return None
            children.append(state)
        return self.delta.get((tree.symbol, tuple(children)))

    def accepts(self, tree: Tree) -> bool:
        return self.run(tree) in self.finals

    def summary(self) -> str:
        return (
            f"древесный автомат: {len(self)} состояний, "
            f"{len(self.delta)} переходов, финальных {len(self.finals)}"
        )


def at_least_classes(
    predicate: Callable[[Tree], bool],
    samples: Sequence[Tree],
    separators: Sequence[Context],
) -> Verdict:
    """Оценка снизу на число классов древесной эквивалентности.

    Два дерева различимы, если какой-то контекст разводит их
    по принадлежности языку. Различили `n` деревьев — **доказано**, что
    классов не меньше `n`, а значит, и состояний в любом распознающем
    восходящем автомате не меньше `n`.

    Бесконечность классов — это обобщение при `n → ∞`, и пишет его человек.
    Оракул даёт только конечную оценку, зато точную.
    """
    if not samples:
        return refuted("нечего различать: список деревьев пуст", samples)
    signatures: dict[tuple[bool, ...], Tree] = {}
    for tree in samples:
        key = tuple(predicate(ctx.fill(tree)) for ctx in separators)
        signatures.setdefault(key, tree)
    count = len(signatures)
    if count < len(samples):
        return unknown(
            f"различено {count} деревьев из {len(samples)}: предъявленных "
            "контекстов не хватило, чтобы развести остальные",
            [str(t) for t in signatures.values()],
        )
    return proved(
        f"классов не меньше {count}: контексты развели все предъявленные "
        "деревья попарно",
        [str(t) for t in samples],
    )


def disagreements(
    automaton: TreeAutomaton,
    predicate: Callable[[Tree], bool],
    max_size: int = 7,
    limit: int = 5,
) -> list[Tree]:
    """Деревья, на которых автомат расходится с условием.

    Обязательный шаг перед тем, как предъявлять автомат: красные пометки
    «автомат принимает лишнее» стоят ровно там, где эту проверку
    не сделали.
    """
    found = []
    for tree in trees(automaton.signature, max_size):
        if automaton.accepts(tree) != predicate(tree):
            found.append(tree)
            if len(found) >= limit:
                break
    return found
