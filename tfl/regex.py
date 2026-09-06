"""Академические регулярные выражения: разбор, AST, производные Брзозовского.

«Академическое» регулярное выражение (терминология курса) — это выражение
только из трёх операций: альтернатива `|`, конкатенация (неявная) и итерация
Клини `*`, плюс скобки, символы алфавита, `ε` (пустое слово) и `∅` (пустой язык).
Никаких `+`, `?`, классов символов и обратных ссылок — это уже расширенные
регулярные выражения, они живут в `tfl/extregex.py`.

Приоритеты (от низшего к высшему): `|` < конкатенация < `*`.

Производные Брзозовского дают второй, независимый от Томпсона, способ
построить ДКА. Два способа обязаны совпасть — на этом строится самопроверка
ядра (см. tests/test_regex.py::test_two_constructions_agree).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

__all__ = [
    "Node",
    "Empty",
    "Eps",
    "Sym",
    "Alt",
    "Cat",
    "Star",
    "parse",
    "ParseError",
    "EMPTY",
    "EPS",
]

# Символы, которые в PDF курса встречаются вместо ASCII-аналогов.
_NORMALIZE = {
    "∗": "*",  # U+2217 ASTERISK OPERATOR
    "⋅": "",   # явная точка конкатенации
    "·": "",
    "∣": "|",
    "（": "(",
    "）": ")",
    " ": " ",
}

EPS_CHARS = "ε"
EMPTY_CHARS = "∅"


class ParseError(ValueError):
    """Синтаксическая ошибка в регулярном выражении."""


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


class Node:
    """Узел AST регулярного выражения.

    Узлы неизменяемы и сравниваются структурно — это нужно, чтобы производные
    Брзозовского можно было использовать как состояния ДКА (их приходится
    класть в множество и сравнивать).
    """

    __slots__ = ()

    @cached_property
    def nullable(self) -> bool:
        """Содержит ли язык узла пустое слово."""
        raise NotImplementedError

    def alphabet(self) -> frozenset[str]:
        """Множество символов, встречающихся в выражении."""
        return frozenset(self._symbols())

    def _symbols(self) -> Iterator[str]:
        raise NotImplementedError

    def derivative(self, char: str) -> Node:
        """Производная Брзозовского: язык хвостов после символа `char`."""
        raise NotImplementedError

    def size(self) -> int:
        """Число узлов — грубая мера сложности выражения."""
        return 1


@dataclass(frozen=True)
class Empty(Node):
    """Пустой язык ∅ (не содержит ни одного слова, в т.ч. пустого)."""

    @cached_property
    def nullable(self) -> bool:
        return False

    def _symbols(self) -> Iterator[str]:
        return iter(())

    def derivative(self, char: str) -> Node:
        return EMPTY

    def __str__(self) -> str:
        return "∅"


@dataclass(frozen=True)
class Eps(Node):
    """Язык из одного пустого слова {ε}."""

    @cached_property
    def nullable(self) -> bool:
        return True

    def _symbols(self) -> Iterator[str]:
        return iter(())

    def derivative(self, char: str) -> Node:
        return EMPTY

    def __str__(self) -> str:
        return "ε"


EMPTY = Empty()
EPS = Eps()


@dataclass(frozen=True)
class Sym(Node):
    """Один символ алфавита."""

    char: str

    @cached_property
    def nullable(self) -> bool:
        return False

    def _symbols(self) -> Iterator[str]:
        yield self.char

    def derivative(self, char: str) -> Node:
        return EPS if char == self.char else EMPTY

    def __str__(self) -> str:
        return self.char


@dataclass(frozen=True)
class Alt(Node):
    """Альтернатива. Аргументы хранятся отсортированным кортежем без дублей:
    так `a|b` и `b|a` — один и тот же объект, что резко сокращает число
    состояний при построении ДКА по производным."""

    args: tuple[Node, ...]

    @cached_property
    def nullable(self) -> bool:
        return any(a.nullable for a in self.args)

    def _symbols(self) -> Iterator[str]:
        for a in self.args:
            yield from a._symbols()

    def derivative(self, char: str) -> Node:
        return alt(*(a.derivative(char) for a in self.args))

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def __str__(self) -> str:
        return "(" + "|".join(str(a) for a in self.args) + ")"


@dataclass(frozen=True)
class Cat(Node):
    """Конкатенация. Порядок аргументов значим, поэтому не сортируется."""

    args: tuple[Node, ...]

    @cached_property
    def nullable(self) -> bool:
        return all(a.nullable for a in self.args)

    def _symbols(self) -> Iterator[str]:
        for a in self.args:
            yield from a._symbols()

    def derivative(self, char: str) -> Node:
        # d(r1 r2 ... rn) = d(r1)·r2...rn | [r1 nullable] d(r2...rn)
        options: list[Node] = []
        for i, arg in enumerate(self.args):
            options.append(cat(arg.derivative(char), *self.args[i + 1 :]))
            if not arg.nullable:
                break
        return alt(*options)

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def __str__(self) -> str:
        return "".join(str(a) for a in self.args)


@dataclass(frozen=True)
class Star(Node):
    """Итерация Клини."""

    arg: Node

    @cached_property
    def nullable(self) -> bool:
        return True

    def _symbols(self) -> Iterator[str]:
        yield from self.arg._symbols()

    def derivative(self, char: str) -> Node:
        return cat(self.arg.derivative(char), self)

    def size(self) -> int:
        return 1 + self.arg.size()

    def __str__(self) -> str:
        inner = str(self.arg)
        if len(inner) > 1 and not (inner.startswith("(") and inner.endswith(")")):
            inner = f"({inner})"
        return f"{inner}*"


# --------------------------------------------------------------------------
# Умные конструкторы: нормализуют AST на лету
# --------------------------------------------------------------------------
#
# Без нормализации производные Брзозовского порождают бесконечно много
# синтаксически разных, но эквивалентных выражений, и построение ДКА
# не завершается. Тождества ниже — минимальный набор, при котором
# число различных производных конечно (теорема Брзозовского).


def alt(*args: Node) -> Node:
    """Альтернатива с нормализацией: ∅|r = r, r|r = r, порядок не важен."""
    flat: set[Node] = set()
    for a in args:
        if isinstance(a, Empty):
            continue
        if isinstance(a, Alt):
            flat.update(a.args)
        else:
            flat.add(a)
    if not flat:
        return EMPTY
    if len(flat) == 1:
        return next(iter(flat))
    return Alt(tuple(sorted(flat, key=_sort_key)))


def cat(*args: Node) -> Node:
    """Конкатенация с нормализацией: ∅r = ∅, εr = r, ассоциативность."""
    flat: list[Node] = []
    for a in args:
        if isinstance(a, Empty):
            return EMPTY
        if isinstance(a, Eps):
            continue
        if isinstance(a, Cat):
            flat.extend(a.args)
        else:
            flat.append(a)
    if not flat:
        return EPS
    if len(flat) == 1:
        return flat[0]
    return Cat(tuple(flat))


def star(arg: Node) -> Node:
    """Итерация с нормализацией: ∅* = ε, ε* = ε, (r*)* = r*."""
    if isinstance(arg, (Empty, Eps)):
        return EPS
    if isinstance(arg, Star):
        return arg
    return Star(arg)


def _sort_key(node: Node) -> tuple[int, str]:
    """Устойчивый порядок для аргументов Alt (нужен детерминизм построения)."""
    order = {Empty: 0, Eps: 1, Sym: 2, Star: 3, Cat: 4, Alt: 5}
    return (order[type(node)], str(node))


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


class _Parser:
    """Рекурсивный спуск.

        alt  := cat ('|' cat)*
        cat  := rep*                 (пусто -> ε)
        rep  := atom '*'*
        atom := '(' alt ')' | symbol | 'ε' | '∅'
    """

    def __init__(self, text: str, extra_symbols: str = "") -> None:
        for src, dst in _NORMALIZE.items():
            text = text.replace(src, dst)
        self.text = text
        self.pos = 0
        self.extra = set(extra_symbols)

    # -- вспомогательное ----------------------------------------------------

    def _skip_spaces(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str | None:
        self._skip_spaces()
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _is_symbol(self, ch: str) -> bool:
        return ch.isalnum() or ch in self.extra

    # -- правила грамматики -------------------------------------------------

    def parse(self) -> Node:
        node = self.p_alt()
        if self._peek() is not None:
            raise ParseError(
                f"лишний символ {self.text[self.pos]!r} в позиции {self.pos}: "
                f"{self.text!r}"
            )
        return node

    def p_alt(self) -> Node:
        branches = [self.p_cat()]
        while self._peek() == "|":
            self.pos += 1
            branches.append(self.p_cat())
        return alt(*branches)

    def p_cat(self) -> Node:
        parts: list[Node] = []
        while True:
            ch = self._peek()
            # Пустая альтернатива `(b | )` в условиях курса встречается и
            # означает ε — принимаем её молча.
            if ch is None or ch in "|)":
                break
            parts.append(self.p_rep())
        return cat(*parts)

    def p_rep(self) -> Node:
        node = self.p_atom()
        while self._peek() == "*":
            self.pos += 1
            node = star(node)
        return node

    def p_atom(self) -> Node:
        ch = self._peek()
        if ch is None:
            raise ParseError(f"неожиданный конец выражения: {self.text!r}")
        if ch == "(":
            self.pos += 1
            node = self.p_alt()
            if self._peek() != ")":
                raise ParseError(f"не закрыта скобка в позиции {self.pos}: {self.text!r}")
            self.pos += 1
            return node
        if ch == ")":
            raise ParseError(f"лишняя ')' в позиции {self.pos}: {self.text!r}")
        if ch == "*":
            raise ParseError(f"'*' без операнда в позиции {self.pos}: {self.text!r}")
        self.pos += 1
        if ch in EPS_CHARS:
            return EPS
        if ch in EMPTY_CHARS:
            return EMPTY
        if self._is_symbol(ch):
            return Sym(ch)
        raise ParseError(f"недопустимый символ {ch!r} в позиции {self.pos - 1}: {self.text!r}")


def parse(text: str, extra_symbols: str = "") -> Node:
    """Разобрать академическое регулярное выражение.

    `extra_symbols` — дополнительные символы алфавита сверх букв и цифр
    (например, "()" для языков скобочных последовательностей из РК1).

    >>> str(parse("(a|b)*abb"))
    '(a|b)*abb'
    >>> parse("a|a") == parse("a")
    True
    """
    return _Parser(text, extra_symbols).parse()
