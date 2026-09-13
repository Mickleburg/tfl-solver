"""Расширенные регулярные выражения ЛР4: группы захвата и ссылки.

Синтаксис из условия (`corpus/tasks/lab4-2024-task.md`):

```
[rg]  ::= [rg][rg] | [rg]|[rg] | ([rg]) | (?:[rg]) | [rg]* | (?[num]) | \\[num] | [a-z]
[num] ::= [1-9]
[rg]  ::= (?=[rg])      # в части вариантов
```

Семантика ссылок задана преподавателем в issue #35
(`corpus/issues/lab4-issue35.md`) и в двух местах расходится с интуицией:

* `(?N)` — ссылка на **выражение** группы. Она корректна **всегда**:
  подставить подвыражение можно, ещё не разобрав ни одного символа.
  Поэтому `(?1)(a|b)` — это просто `(a|b)(a|b)`, а левая рекурсия
  вроде `((?1)a|b)` синтаксически корректна, хотя разбор по ней
  в реальных машинах падает.
* `\\N` — ссылка на **строку**, захваченную группой. Корректна только
  если к моменту обращения группа дочитана до конца **по всем путям
  разбора**. Группа под звёздочкой не считается инициализированной:
  звёздочка может раскрыться в ноль итераций.

Нумерация групп — по номеру открывающей скобки слева направо, считая
только скобки захвата.

Каркас (`skeleton_cfg`) — это КС-грамматика, в которой каждая группа стала
нетерминалом. Для выражений **без** `\\N` и опережающих проверок каркас
задаёт язык точно. Со ссылками на строку и с проверками он даёт язык
**сверху**: равенство захваченных строк и опережающее условие в КС-грамматике
не выражаются. Это то же приближение сверху, что в `tfl/approx.py`,
и вывод из него односторонний.

Поздняя формулировка добавляет stateful-семантику: значение захвата может
сохраняться между итерациями, а первая успешная итерация выбирает ветвь без
ещё не инициализированной ссылки. Для неё используются
`validate(..., stateful=True)` и `exact_matches(..., stateful=True)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from tfl.cfg import CFG, EPSILON, Production
from tfl.verdict import Verdict, proved, refuted

__all__ = [
    "Node",
    "Eps",
    "Sym",
    "Anchor",
    "Concat",
    "Alt",
    "Star",
    "Group",
    "NonCapturing",
    "ExprRef",
    "StrRef",
    "Lookahead",
    "ParseError",
    "parse_extended",
    "groups",
    "validate",
    "skeleton_cfg",
    "matches",
    "exact_matches",
    "MAX_GROUPS",
]

#: Ограничение условия: групп захвата не больше девяти.
MAX_GROUPS = 9


class ParseError(ValueError):
    """Синтаксическая ошибка в записи выражения."""


@dataclass(frozen=True)
class Node:
    """Узел разобранного выражения."""

    def children(self) -> tuple["Node", ...]:
        return ()

    def walk(self) -> Iterator["Node"]:
        yield self
        for child in self.children():
            yield from child.walk()


@dataclass(frozen=True)
class Eps(Node):
    def __str__(self) -> str:
        return "ε"


@dataclass(frozen=True)
class Sym(Node):
    char: str

    def __str__(self) -> str:
        return self.char


@dataclass(frozen=True)
class Anchor(Node):
    char: str

    def __str__(self) -> str:
        return self.char


@dataclass(frozen=True)
class Concat(Node):
    parts: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.parts

    def __str__(self) -> str:
        return "".join(_wrap(p, in_concat=True) for p in self.parts)


@dataclass(frozen=True)
class Alt(Node):
    options: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.options

    def __str__(self) -> str:
        return "|".join(str(o) for o in self.options)


@dataclass(frozen=True)
class Star(Node):
    inner: Node

    def children(self) -> tuple[Node, ...]:
        return (self.inner,)

    def __str__(self) -> str:
        return _wrap(self.inner, in_concat=True) + "*"


@dataclass(frozen=True)
class Group(Node):
    number: int
    inner: Node

    def children(self) -> tuple[Node, ...]:
        return (self.inner,)

    def __str__(self) -> str:
        return f"({self.inner})"


@dataclass(frozen=True)
class NonCapturing(Node):
    inner: Node

    def children(self) -> tuple[Node, ...]:
        return (self.inner,)

    def __str__(self) -> str:
        return f"(?:{self.inner})"


@dataclass(frozen=True)
class ExprRef(Node):
    number: int

    def __str__(self) -> str:
        return f"(?{self.number})"


@dataclass(frozen=True)
class StrRef(Node):
    number: int

    def __str__(self) -> str:
        return f"\\{self.number}"


@dataclass(frozen=True)
class Lookahead(Node):
    inner: Node

    def children(self) -> tuple[Node, ...]:
        return (self.inner,)

    def __str__(self) -> str:
        return f"(?={self.inner})"


def _wrap(node: Node, in_concat: bool = False) -> str:
    if isinstance(node, (Alt,)) or (in_concat and isinstance(node, Concat)):
        return f"(?:{node})"
    return str(node)


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def parse_extended(text: str) -> Node:
    """Разобрать расширенное регулярное выражение.

    Группы нумеруются по порядку открывающих скобок захвата.

    >>> str(parse_extended("(aa|bb)(?1)"))
    '(aa|bb)(?1)'
    >>> str(parse_extended("(a(?1)b|c)"))
    '(a(?1)b|c)'
    """
    normalized = (
        "".join(text.split()).replace("ˆ", "^").replace("∗", "*")
    )
    parser = _Parser(normalized)
    node = parser.expression()
    if parser.pos != len(normalized):
        raise ParseError(
            f"лишний текст с позиции {parser.pos}: «{normalized[parser.pos:]}»"
        )
    return node


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.group_count = 0

    def peek(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def expression(self) -> Node:
        options = [self.term()]
        while self.peek() == "|":
            self.pos += 1
            options.append(self.term())
        return options[0] if len(options) == 1 else Alt(tuple(options))

    def term(self) -> Node:
        parts: list[Node] = []
        while True:
            char = self.peek()
            if char is None or char in "|)":
                break
            parts.append(self.factor())
        if not parts:
            raise ParseError(f"пустая часть выражения в позиции {self.pos}")
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))

    def factor(self) -> Node:
        node = self.atom()
        while self.peek() in ("*", "+", "?"):
            operator = self.peek()
            self.pos += 1
            if operator == "*":
                node = Star(node)
            elif operator == "+":
                node = Concat((node, Star(node)))
            else:
                node = Alt((node, Eps()))
        return node

    def atom(self) -> Node:
        char = self.peek()
        if char is None:
            raise ParseError("выражение оборвано")
        if char == "\\":
            self.pos += 1
            digit = self.peek()
            if digit is None or digit not in "123456789":
                raise ParseError(f"после \\ ожидалась цифра 1-9, позиция {self.pos}")
            self.pos += 1
            return StrRef(int(digit))
        if char == "(":
            return self.bracket()
        if char in "^$":
            self.pos += 1
            return Anchor(char)
        if char.isalpha() and char.islower():
            self.pos += 1
            return Sym(char)
        raise ParseError(f"недопустимый символ «{char}» в позиции {self.pos}")

    def bracket(self) -> Node:
        self.pos += 1  # съели '('
        char = self.peek()
        if char == "?":
            self.pos += 1
            marker = self.peek()
            if marker == ":":
                self.pos += 1
                inner = self.expression()
                self.expect(")")
                return NonCapturing(inner)
            if marker == "=":
                self.pos += 1
                inner = self.expression()
                self.expect(")")
                return Lookahead(inner)
            if marker is not None and marker in "123456789":
                self.pos += 1
                self.expect(")")
                return ExprRef(int(marker))
            raise ParseError(f"после (? ожидалось :, = или цифра, позиция {self.pos}")
        self.group_count += 1
        number = self.group_count
        inner = self.expression()
        self.expect(")")
        return Group(number, inner)

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise ParseError(f"ожидался «{char}» в позиции {self.pos}")
        self.pos += 1


def groups(node: Node) -> dict[int, Node]:
    """Тела групп захвата по номерам.

    >>> sorted(groups(parse_extended("(a|(bb))(?2)")))
    [1, 2]
    """
    return {n.number: n.inner for n in node.walk() if isinstance(n, Group)}


# --------------------------------------------------------------------------
# Проверка корректности
# --------------------------------------------------------------------------


def validate(node: Node, stateful: bool = False) -> Verdict:
    """Корректно ли выражение по всем ограничениям условия.

    Проверяются: число групп захвата, содержимое опережающих проверок
    и инициализированность ссылок на строку.

    >>> validate(parse_extended("(a|(bb))(?2)")).value
    True
    >>> validate(parse_extended("(a|(bb))\\\\2")).value
    False
    """
    table = groups(node)
    problems: list[str] = []

    if len(table) > MAX_GROUPS:
        problems.append(f"групп захвата {len(table)}, а разрешено {MAX_GROUPS}")

    for sub in node.walk():
        if isinstance(sub, Lookahead) and not stateful:
            for inner in sub.inner.walk():
                if isinstance(inner, Group):
                    problems.append(
                        f"опережающая проверка «{sub}» содержит группу захвата"
                    )
                    break
            for inner in sub.inner.walk():
                if isinstance(inner, Lookahead):
                    problems.append(
                        f"опережающая проверка «{sub}» содержит вложенную проверку"
                    )
                    break
        if isinstance(sub, (ExprRef, StrRef)) and sub.number not in table:
            problems.append(f"ссылка «{sub}» указывает на несуществующую группу")

    if not problems and not stateful:
        _initialised(node, frozenset(), table, (), problems)

    if problems:
        return refuted("; ".join(dict.fromkeys(problems)), problems)
    detail = (
        "инициализация проверяется во время stateful-разбора"
        if stateful
        else "все ссылки на строку инициализированы по всем путям разбора"
    )
    return proved(f"выражение корректно: {len(table)} групп захвата, {detail}", node)


def _initialised(
    node: Node,
    before: frozenset[int],
    table: dict[int, Node],
    stack: tuple[int, ...],
    problems: list[str],
) -> frozenset[int]:
    """Какие группы гарантированно дочитаны после этого узла.

    Альтернатива пересекает результаты ветвей, звёздочка не даёт ничего
    (ноль итераций), группа добавляет себя только после закрытия.
    Ссылка на выражение раскрывается на месте — иначе не поймать случай
    `(a|(?2))(a|(bb\\1))`, где ошибка возникает именно после подстановки.
    """
    if isinstance(node, (Eps, Sym, Anchor)):
        return before
    if isinstance(node, Concat):
        current = before
        for part in node.parts:
            current = _initialised(part, current, table, stack, problems)
        return current
    if isinstance(node, Alt):
        results = [
            _initialised(option, before, table, stack, problems)
            for option in node.options
        ]
        return frozenset.intersection(*results) if results else before
    if isinstance(node, Star):
        _initialised(node.inner, before, table, stack, problems)
        return before
    if isinstance(node, Group):
        after = _initialised(node.inner, before, table, stack, problems)
        return after | {node.number}
    if isinstance(node, NonCapturing):
        return _initialised(node.inner, before, table, stack, problems)
    if isinstance(node, Lookahead):
        _initialised(node.inner, before, table, stack, problems)
        return before
    if isinstance(node, StrRef):
        if node.number not in before:
            problems.append(
                f"ссылка «{node}» обращается к группе {node.number}, "
                "которая на этом пути разбора ещё не дочитана"
            )
        return before
    if isinstance(node, ExprRef):
        if node.number in stack:
            return before  # рекурсия: тело уже проверено выше по стеку
        body = table.get(node.number)
        if body is None:
            return before
        _initialised(body, before, table, stack + (node.number,), problems)
        return before
    return before


# --------------------------------------------------------------------------
# Каркас: КС-грамматика по выражению
# --------------------------------------------------------------------------


def skeleton_cfg(node: Node, start: str = "S") -> CFG:
    """КС-грамматика каркаса: каждая группа захвата — свежий нетерминал.

    Точна для выражений без `\\N` и без опережающих проверок. Со ссылкой
    на строку каркас **шире** языка: он допускает любое слово группы
    вместо ровно того, что она захватила. Опережающая проверка стирается
    в ε, что тоже расширяет язык.

    >>> grammar = skeleton_cfg(parse_extended("(a(?1)b|c)"))
    >>> from tfl.parse import recognize
    >>> [w for w in ("c", "acb", "aacbb", "ab", "acc") if recognize(grammar, w)]
    ['c', 'acb', 'aacbb']
    """
    productions: list[Production] = []
    counter = [0]
    emitted: set[int] = set()

    def fresh(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def emit(current: Node) -> str:
        if isinstance(current, Sym):
            return current.char
        if isinstance(current, (Eps, Anchor)):
            name = fresh("E")
            productions.append(Production(name, ()))
            return name
        if isinstance(current, Group):
            name = f"G{current.number}"
            if current.number not in emitted:
                emitted.add(current.number)
                productions.append(Production(name, (emit(current.inner),)))
            return name
        if isinstance(current, NonCapturing):
            return emit(current.inner)
        if isinstance(current, (ExprRef, StrRef)):
            return f"G{current.number}"
        if isinstance(current, Lookahead):
            name = fresh("L")
            productions.append(Production(name, ()))
            return name
        if isinstance(current, Concat):
            name = fresh("C")
            productions.append(Production(name, tuple(emit(p) for p in current.parts)))
            return name
        if isinstance(current, Alt):
            name = fresh("A")
            for option in current.options:
                productions.append(Production(name, (emit(option),)))
            return name
        if isinstance(current, Star):
            name = fresh("T")
            inner = emit(current.inner)
            productions.append(Production(name, (inner, name)))
            productions.append(Production(name, ()))
            return name
        raise TypeError(f"неизвестный узел {current!r}")

    root = emit(node)
    productions.insert(0, Production(start, (root,)))
    nonterminals = frozenset(p.lhs for p in productions)
    return CFG(start, tuple(productions), nonterminals)


def matches(node: Node, word: str) -> bool:
    """Принимает ли **каркас** выражения данное слово.

    Для выражений без `\\N` и без опережающих проверок это ответ про сам
    язык. Иначе — про язык сверху: `True` ничего не доказывает,
    а `False` доказывает, что слово в язык не входит.
    """
    from tfl.parse import recognize

    return recognize(skeleton_cfg(node), word)


# --------------------------------------------------------------------------
# Точная принадлежность: захваты, строковые ссылки и lookahead
# --------------------------------------------------------------------------


CaptureState = tuple[str | None, ...]
MatchState = tuple[int, CaptureState]


class _ExactMatcher:
    """Неподвижная точка отношений разбора на одном конечном слове.

    Обычный рекурсивный спуск зацикливается на допустимых ссылках вроде
    ``(a(?1)b|c)``. Здесь для каждой пары ``(узел, конфигурация)`` накапливается
    конечное множество результатов. Все уравнения монотонны, а позиции и
    значения захватов являются подстроками фиксированного входа, поэтому
    итерация достигает наименьшей неподвижной точки.
    """

    def __init__(self, root: Node, word: str, group_count: int) -> None:
        self.root = root
        self.word = word
        self.group_count = group_count
        self.group_bodies = groups(root)
        self.cache: dict[tuple[Node, MatchState], set[MatchState]] = {}

    def results(self, node: Node, state: MatchState) -> set[MatchState]:
        return self.cache.setdefault((node, state), set())

    def derive(self, node: Node, state: MatchState) -> set[MatchState]:
        position, captures = state
        if isinstance(node, Eps):
            return {state}
        if isinstance(node, Sym):
            if position < len(self.word) and self.word[position] == node.char:
                return {(position + 1, captures)}
            return set()
        if isinstance(node, Anchor):
            if node.char == "^" and position == 0:
                return {state}
            if node.char == "$" and position == len(self.word):
                return {state}
            return set()
        if isinstance(node, Concat):
            current = {state}
            for part in node.parts:
                following: set[MatchState] = set()
                for item in current:
                    following.update(self.results(part, item))
                current = following
                if not current:
                    break
            return current
        if isinstance(node, Alt):
            out: set[MatchState] = set()
            for option in node.options:
                out.update(self.results(option, state))
            return out
        if isinstance(node, Star):
            out = {state}
            for item in self.results(node.inner, state):
                if item != state:
                    out.update(self.results(node, item))
            return out
        if isinstance(node, Group):
            out = set()
            for end, after in self.results(node.inner, state):
                values = list(after)
                values[node.number - 1] = self.word[position:end]
                out.add((end, tuple(values)))
            return out
        if isinstance(node, NonCapturing):
            return set(self.results(node.inner, state))
        if isinstance(node, ExprRef):
            body = self.group_bodies[node.number]
            return set(self.results(body, state))
        if isinstance(node, StrRef):
            captured = captures[node.number - 1]
            if captured is not None and self.word.startswith(captured, position):
                return {(position + len(captured), captures)}
            return set()
        if isinstance(node, Lookahead):
            return {state} if self.results(node.inner, state) else set()
        raise TypeError(f"неизвестный узел {node!r}")

    def solve(self) -> set[MatchState]:
        initial: MatchState = (0, (None,) * self.group_count)
        root_results = self.results(self.root, initial)
        while True:
            changed = False
            known_keys = len(self.cache)
            for (node, state), current in list(self.cache.items()):
                before = len(current)
                current.update(self.derive(node, state))
                changed = changed or len(current) != before
            if not changed and len(self.cache) == known_keys:
                return root_results


def exact_matches(node: Node, word: str, stateful: bool = False) -> bool:
    """Точно проверить принадлежность расширенному выражению.

    В отличие от :func:`matches`, этот распознаватель сохраняет фактические
    строки групп, требует буквального совпадения ``\\N`` и исполняет
    положительный lookahead без потребления входа. Ссылки на выражение могут
    быть рекурсивными: вычисляется наименьшая неподвижная точка, а не вводится
    искусственный предел глубины.

    Некорректное по правилам задания выражение отвергается исключением.
    ``stateful=True`` включает семантику поздних вариантов: захват предыдущей
    итерации сохраняется, поэтому ссылка может стоять текстуально раньше
    инициализирующей альтернативы. Неинициализированная ссылка просто не даёт
    перехода; первый успешный проход обязан выбрать ветвь без неё.
    Алгоритм конечен на фиксированном слове, но в худшем случае экспоненциален
    по числу групп и длине: значения захватов являются частью конфигурации.
    """
    verdict = validate(node, stateful=stateful)
    if verdict.value is not True:
        raise ValueError(verdict.reason)
    table = groups(node)
    results = _ExactMatcher(node, word, len(table)).solve()
    return any(position == len(word) for position, _captures in results)
