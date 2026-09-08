"""Расширенные регулярные выражения ЛР2: предпросмотр и ретроспектива.

Условие ЛР2 2025 (`lab_tfl_2025_2.pdf`), третий пункт задания. Разрешено
использовать:

* wildcard `.` — произвольный символ алфавита;
* положительную итерацию `τ⁺` и опцию `τ?`;
* классы букв `[c₁…cₖ]` и их дополнения `[^c₁…cₖ]`;
* **обязательные** маркеры начала и конца `^` и `$`;
* предпросмотр и ретроспективную проверку, заданные равенствами:

      τ₀(?= τ₁)τ₂ ≡ τ₀((τ₁.*) ∩ τ₂)          τ₀(?! τ₁)τ₂ ≡ τ₀(‾(τ₁.*) ∩ τ₂)
      τ₀(?<= τ₁)τ₂ ≡ (τ₀ ∩ (τ₁.*))τ₂          τ₀(?<! τ₁)τ₂ ≡ (τ₀ ∩ ‾(τ₁.*))τ₂

Распознаватель для таких выражений — **дополнительный балл** по условию,
а автомат, совмещающий ИЛИ- и И-недетерминизм, — ещё 1–2 балла
(issue #40: «Просто вызвать библиотеку регулярок баллов, увы, не даст»).

## Позиционное прочтение, и почему оно нужнее печатного

Печатные равенства заданы для **концевого** контекста: `τ₀` — всё, что
слева, `τ₂` — всё, что справа. Буквально применить их можно только
к верхнему уровню выражения. А в собственном примере преподавателя
проверка стоит **внутри звёздочки**:

    ^((?=(b*ab*ab*)*$)a*b)*$

и смотрит она на остаток **всего слова**, а не на остаток итерации.

Поэтому здесь принято позиционное прочтение, из которого печатные
равенства следуют как частный случай верхнего уровня:

* `(?= τ₁)` в позиции `i` выполняется, если `w[i:] ∈ L(τ₁)·Σ*`,
  то есть `τ₁` сопоставляется с каким-то началом остатка;
* `(?! τ₁)` — если не выполняется;
* `(?<= τ₁)` — если `w[:i] ∈ L(τ₁)·Σ*`;
* `(?<! τ₁)` — если не выполняется.

Совпадение обоих прочтений на верхнем уровне не предполагается,
а проверяется: `check_equations` строит язык по печатным равенствам
операциями над автоматами и сверяет со `matches` на всех словах
до заданной длины.

## Ретроспектива: расхождение с общепринятой семантикой

В печатном равенстве **обе** проверки используют `(τ₁.*)`:

    τ₀(?= τ₁)τ₂ ≡ τ₀((τ₁.*) ∩ τ₂)
    τ₀(?<= τ₁)τ₂ ≡ (τ₀ ∩ (τ₁.*))τ₂

То есть ретроспектива требует, чтобы прочитанный префикс **начинался**
с `τ₁`. В PCRE и всюду за пределами курса `(?<=τ₁)` означает, что
префикс `τ₁` **заканчивает**, то есть `τ₀ ∩ (.*τ₁)`. Разница
не теоретическая: на `^a(?<=b).*$` курсовое прочтение отвергает всё,
общепринятое тоже, а на `^ab(?<=a).*$` они расходятся — курсовое
принимает (префикс `ab` начинается с `a`), общепринятое отвергает
(префикс `ab` кончается на `b`). Свидетель предъявляется тестом.

Реализованы оба (`BEHIND_COURSE` по умолчанию, `BEHIND_STANDARD`),
потому что угадывать за преподавателя тут нельзя, а вопрос стоит задать.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tfl import regex as rx
from tfl.automata import DFA, NFA, State, dfa_of, intersection, union
from tfl.verdict import Verdict, refuted, unknown

__all__ = [
    "Ext",
    "Sym",
    "Any",
    "Klass",
    "Eps",
    "Anchor",
    "Alt",
    "Cat",
    "Star",
    "Look",
    "AHEAD",
    "NEG_AHEAD",
    "BEHIND",
    "NEG_BEHIND",
    "START",
    "END",
    "BEHIND_COURSE",
    "BEHIND_STANDARD",
    "parse_extended",
    "matches",
    "accepts",
    "alphabet_of",
    "has_lookaround",
    "to_regex",
    "check_equations",
    "to_afa",
]

#: Виды проверок.
AHEAD = "предпросмотр"
NEG_AHEAD = "отрицательный предпросмотр"
BEHIND = "ретроспектива"
NEG_BEHIND = "отрицательная ретроспектива"

#: Маркеры.
START = "начало"
END = "конец"

#: Два прочтения ретроспективы, см. шапку модуля.
BEHIND_COURSE = "префикс начинается с τ₁"
BEHIND_STANDARD = "префикс кончается на τ₁"


# --------------------------------------------------------------------------
# Синтаксис
# --------------------------------------------------------------------------


class Ext:
    """Узел расширенного выражения."""


@dataclass(frozen=True)
class Eps(Ext):
    def __str__(self) -> str:
        return "ε"


@dataclass(frozen=True)
class Sym(Ext):
    char: str

    def __str__(self) -> str:
        return self.char


@dataclass(frozen=True)
class Any(Ext):
    """`.` — произвольный символ алфавита."""

    def __str__(self) -> str:
        return "."


@dataclass(frozen=True)
class Klass(Ext):
    """`[c₁…cₖ]` и `[^c₁…cₖ]`."""

    letters: frozenset[str]
    negated: bool = False

    def __str__(self) -> str:
        body = "".join(sorted(self.letters))
        return f"[{'^' if self.negated else ''}{body}]"


@dataclass(frozen=True)
class Anchor(Ext):
    kind: str

    def __str__(self) -> str:
        return "^" if self.kind == START else "$"


@dataclass(frozen=True)
class Alt(Ext):
    parts: tuple[Ext, ...]

    def __str__(self) -> str:
        return "(" + "|".join(str(p) for p in self.parts) + ")"


@dataclass(frozen=True)
class Cat(Ext):
    parts: tuple[Ext, ...]

    def __str__(self) -> str:
        return "".join(str(p) for p in self.parts)


@dataclass(frozen=True)
class Star(Ext):
    inner: Ext

    def __str__(self) -> str:
        return f"({self.inner})*"


@dataclass(frozen=True)
class Look(Ext):
    kind: str
    inner: Ext

    def __str__(self) -> str:
        mark = {
            AHEAD: "?=",
            NEG_AHEAD: "?!",
            BEHIND: "?<=",
            NEG_BEHIND: "?<!",
        }[self.kind]
        return f"({mark}{self.inner})"


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


class _Parser:
    """Рекурсивный спуск. Пробелы незначимы: в условии их ставят для читаемости."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, message: str):
        return ValueError(f"{message} (позиция {self.pos} в «{self.text}»)")

    def peek(self) -> str | None:
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1
        return self.text[self.pos] if self.pos < len(self.text) else None

    def take(self) -> str:
        char = self.peek()
        if char is None:
            raise self.error("выражение оборвалось")
        self.pos += 1
        return char

    def parse(self) -> Ext:
        node = self.alternation()
        if self.peek() is not None:
            raise self.error(f"лишний символ «{self.peek()}»")
        return node

    def alternation(self) -> Ext:
        parts = [self.concatenation()]
        while self.peek() == "|":
            self.take()
            parts.append(self.concatenation())
        return parts[0] if len(parts) == 1 else Alt(tuple(parts))

    def concatenation(self) -> Ext:
        parts: list[Ext] = []
        while (char := self.peek()) is not None and char not in "|)":
            parts.append(self.repetition())
        if not parts:
            return Eps()
        return parts[0] if len(parts) == 1 else Cat(tuple(parts))

    def repetition(self) -> Ext:
        node = self.atom()
        while (char := self.peek()) in ("*", "+", "?"):
            self.take()
            if char == "*":
                node = Star(node)
            elif char == "+":
                node = Cat((node, Star(node)))
            else:
                node = Alt((node, Eps()))
        return node

    def atom(self) -> Ext:
        char = self.take()
        if char == "(":
            node = self.group()
            if self.peek() != ")":
                raise self.error("не закрыта скобка")
            self.take()
            return node
        if char == "[":
            return self.klass()
        if char == ".":
            return Any()
        if char == "^":
            return Anchor(START)
        if char == "$":
            return Anchor(END)
        if char in "ε":
            return Eps()
        if char in "*+?|)":
            raise self.error(f"оператор «{char}» без операнда")
        return Sym(char)

    def group(self) -> Ext:
        if self.peek() != "?":
            return self.alternation()
        self.take()
        head = self.take()
        if head == "=":
            return Look(AHEAD, self.alternation())
        if head == "!":
            return Look(NEG_AHEAD, self.alternation())
        if head != "<":
            raise self.error(f"неизвестная группа «(?{head}»")
        sign = self.take()
        if sign == "=":
            return Look(BEHIND, self.alternation())
        if sign == "!":
            return Look(NEG_BEHIND, self.alternation())
        raise self.error(f"неизвестная группа «(?<{sign}»")

    def klass(self) -> Ext:
        negated = False
        if self.peek() == "^":
            self.take()
            negated = True
        letters: set[str] = set()
        while (char := self.peek()) is not None and char != "]":
            letters.add(self.take())
        if self.peek() != "]":
            raise self.error("не закрыт класс букв")
        self.take()
        if not letters:
            raise self.error("пустой класс букв")
        return Klass(frozenset(letters), negated)


def parse_extended(text: str, anchored: bool = True) -> Ext:
    """Разобрать расширенную регулярку.

    `anchored` требует маркеров `^` и `$` по краям — условие называет их
    обязательными, и без них «принадлежность слова языку» определена
    неоднозначно (совпадение с началом или в любом месте).
    """
    stripped = text.strip()
    if anchored and not (stripped.startswith("^") and stripped.endswith("$")):
        raise ValueError(
            "условие ЛР2 требует обязательных маркеров ^ и $ по краям выражения"
        )
    return _Parser(stripped).parse()


# --------------------------------------------------------------------------
# Позиционная семантика — она же распознаватель
# --------------------------------------------------------------------------


def alphabet_of(node: Ext, extra: Iterable[str] = ()) -> frozenset[str]:
    """Буквы, встречающиеся в выражении, плюс явно добавленные."""
    found = set(extra)

    def walk(current: Ext) -> None:
        if isinstance(current, Sym):
            found.add(current.char)
        elif isinstance(current, Klass):
            found.update(current.letters)
        elif isinstance(current, (Alt, Cat)):
            for part in current.parts:
                walk(part)
        elif isinstance(current, Star):
            walk(current.inner)
        elif isinstance(current, Look):
            walk(current.inner)

    walk(node)
    return frozenset(found)


def has_lookaround(node: Ext) -> bool:
    if isinstance(node, Look):
        return True
    if isinstance(node, (Alt, Cat)):
        return any(has_lookaround(part) for part in node.parts)
    if isinstance(node, Star):
        return has_lookaround(node.inner)
    return False


def matches(
    node: Ext,
    word: str,
    start: int = 0,
    alphabet: Iterable[str] | None = None,
    behind: str = BEHIND_COURSE,
) -> frozenset[int]:
    """Все позиции, на которых `node` может закончить разбор с `start`.

    Это и есть распознаватель: множество непусто тогда и только тогда,
    когда узел сопоставляется хотя бы одним способом. Возвраты не нужны —
    множество концов считается сразу, поэтому зацикливания на `(ε)*`
    не происходит.
    """
    letters = frozenset(alphabet) if alphabet is not None else alphabet_of(node)
    cache: dict[tuple[int, int], frozenset[int]] = {}
    length = len(word)

    def go(current: Ext, index: int) -> frozenset[int]:
        key = (id(current), index)
        if key in cache:
            return cache[key]
        cache[key] = frozenset()  # предохранитель от левых циклов
        result = _step(current, index)
        cache[key] = result
        return result

    def _step(current: Ext, index: int) -> frozenset[int]:
        if isinstance(current, Eps):
            return frozenset({index})
        if isinstance(current, Anchor):
            fits = index == 0 if current.kind == START else index == length
            return frozenset({index}) if fits else frozenset()
        if isinstance(current, Sym):
            return (
                frozenset({index + 1})
                if index < length and word[index] == current.char
                else frozenset()
            )
        if isinstance(current, Any):
            return (
                frozenset({index + 1})
                if index < length and word[index] in letters
                else frozenset()
            )
        if isinstance(current, Klass):
            if index >= length:
                return frozenset()
            inside = word[index] in current.letters
            hit = (not inside) if current.negated else inside
            return frozenset({index + 1}) if hit and word[index] in letters else frozenset()
        if isinstance(current, Alt):
            found: set[int] = set()
            for part in current.parts:
                found |= go(part, index)
            return frozenset(found)
        if isinstance(current, Cat):
            positions = {index}
            for part in current.parts:
                nxt: set[int] = set()
                for position in positions:
                    nxt |= go(part, position)
                positions = nxt
                if not positions:
                    break
            return frozenset(positions)
        if isinstance(current, Star):
            reached = {index}
            frontier = {index}
            while frontier:
                fresh: set[int] = set()
                for position in frontier:
                    for end in go(current.inner, position):
                        if end not in reached:
                            reached.add(end)
                            fresh.add(end)
                frontier = fresh
            return frozenset(reached)
        if isinstance(current, Look):
            return frozenset({index}) if _look(current, index) else frozenset()
        raise TypeError(f"неизвестный узел {current!r}")

    def _look(current: Look, index: int) -> bool:
        if current.kind in (AHEAD, NEG_AHEAD):
            # `w[i:] ∈ L(τ₁)·Σ*` — τ₁ ложится на какое-то начало остатка.
            holds = bool(go(current.inner, index))
            return holds if current.kind == AHEAD else not holds
        # Ретроспектива смотрит на прочитанный префикс `w[:i]`.
        if behind == BEHIND_COURSE:
            # `w[:i] ∈ L(τ₁)·Σ*` — префикс **начинается** с τ₁.
            ends = _prefix_ends(current.inner)
            holds = any(end <= index for end in ends)
        elif behind == BEHIND_STANDARD:
            # `w[:i] ∈ Σ*·L(τ₁)` — префикс **кончается** на τ₁.
            holds = any(
                index in matches(current.inner, word[:index], begin, letters, behind)
                for begin in range(index + 1)
            )
        else:
            raise ValueError(
                f"прочтение должно быть «{BEHIND_COURSE}» либо «{BEHIND_STANDARD}»"
            )
        return holds if current.kind == BEHIND else not holds

    def _prefix_ends(inner: Ext) -> frozenset[int]:
        return matches(inner, word, 0, letters, behind)

    return go(node, start)


def accepts(
    node: Ext,
    word: str,
    alphabet: Iterable[str] | None = None,
    behind: str = BEHIND_COURSE,
) -> bool:
    """Принадлежит ли слово языку выражения."""
    return len(word) in matches(node, word, 0, alphabet, behind)


# --------------------------------------------------------------------------
# Печатные равенства — независимая проверка
# --------------------------------------------------------------------------


def to_regex(node: Ext, alphabet: Iterable[str]) -> rx.Node:
    """Перевести узел **без проверок** в академическую регулярку.

    Маркеры `^` и `$` здесь стираются: язык строится для всего слова,
    и привязка к краям выражена самим построением.
    """
    letters = sorted(alphabet)
    any_of = rx.alt(*[rx.Sym(char) for char in letters]) if letters else rx.Empty()

    def walk(current: Ext) -> rx.Node:
        if isinstance(current, (Eps, Anchor)):
            return rx.Eps()
        if isinstance(current, Sym):
            return rx.Sym(current.char)
        if isinstance(current, Any):
            return any_of
        if isinstance(current, Klass):
            chosen = (
                [c for c in letters if c not in current.letters]
                if current.negated
                else [c for c in letters if c in current.letters]
            )
            return rx.alt(*[rx.Sym(char) for char in chosen]) if chosen else rx.Empty()
        if isinstance(current, Alt):
            return rx.alt(*[walk(part) for part in current.parts])
        if isinstance(current, Cat):
            return rx.cat(*[walk(part) for part in current.parts])
        if isinstance(current, Star):
            return rx.star(walk(current.inner))
        raise ValueError("проверка (?=…) в академическую регулярку не переводится")

    return walk(node)


def _concat(left: DFA, right: DFA) -> DFA:
    """Конкатенация языков двух ДКА через ε-переходы и детерминизацию."""
    letters = left.alphabet | right.alphabet
    delta: dict[tuple[State, str], frozenset[State]] = {}
    for (source, char), target in left.delta.items():
        delta[(("л", source), char)] = frozenset({("л", target)})
    for (source, char), target in right.delta.items():
        delta[(("п", source), char)] = frozenset({("п", target)})
    eps = {("л", state): frozenset({("п", right.start)}) for state in left.finals}
    return NFA(
        letters, ("л", left.start), frozenset({("п", state) for state in right.finals}), delta, eps
    ).determinize()


def _dfa_by_equations(node: Ext, letters: frozenset[str], behind: str) -> DFA:
    """Язык узла по **печатным равенствам** условия."""
    if not has_lookaround(node):
        return dfa_of(to_regex(node, letters), letters).complete()

    if isinstance(node, Alt):
        result = _dfa_by_equations(node.parts[0], letters, behind)
        for part in node.parts[1:]:
            result = union(result, _dfa_by_equations(part, letters, behind))
        return result.complete()

    if isinstance(node, Cat):
        index = next(i for i, part in enumerate(node.parts) if isinstance(part, Look))
        look: Look = node.parts[index]
        before = Cat(node.parts[:index]) if index else Eps()
        after = Cat(node.parts[index + 1 :]) if index + 1 < len(node.parts) else Eps()
        tail = Cat((look.inner, Star(Any())))
        checker = _dfa_by_equations(tail, letters, behind).complete()
        if look.kind in (NEG_AHEAD, NEG_BEHIND):
            checker = checker.complement()
        if look.kind in (AHEAD, NEG_AHEAD):
            right = intersection(checker, _dfa_by_equations(after, letters, behind))
            return _concat(_dfa_by_equations(before, letters, behind), right).complete()
        left = intersection(_dfa_by_equations(before, letters, behind), checker)
        return _concat(left, _dfa_by_equations(after, letters, behind)).complete()

    raise ValueError(
        "печатные равенства заданы для проверки на верхнем уровне конкатенации; "
        f"здесь она стоит внутри {type(node).__name__}"
    )


def check_equations(
    node: Ext, alphabet: Iterable[str] | None = None, max_len: int = 8
) -> Verdict:
    """Сверить позиционное прочтение с печатными равенствами условия.

    Равенства заданы для верхнего уровня, поэтому проверка возможна
    не всегда; когда невозможна — честное «не выяснено», а не молчание.
    """
    from tfl.words import iter_words

    letters = frozenset(alphabet) if alphabet is not None else alphabet_of(node)
    if behind_inside_star(node):
        return unknown(
            "проверка стоит под звёздочкой: печатные равенства заданы "
            "для верхнего уровня и такой случай не покрывают"
        )
    try:
        automaton = _dfa_by_equations(node, letters, BEHIND_COURSE)
    except ValueError as error:
        return unknown(str(error))

    for word in iter_words("".join(sorted(letters)), max_len):
        if automaton.accepts(word) != accepts(node, word, letters):
            return refuted(
                f"на слове «{word or 'ε'}» позиционное прочтение и печатные "
                "равенства расходятся",
                word,
            )
    return Verdict(
        True,
        f"позиционное прочтение и печатные равенства совпали на всех словах "
        f"длины ⩽ {max_len}",
    )


def behind_inside_star(node: Ext) -> bool:
    """Есть ли проверка под звёздочкой — там равенства неприменимы."""
    if isinstance(node, Star):
        return has_lookaround(node.inner)
    if isinstance(node, (Alt, Cat)):
        return any(behind_inside_star(part) for part in node.parts)
    if isinstance(node, Look):
        return behind_inside_star(node.inner)
    return False


# --------------------------------------------------------------------------
# Перевод в ПКА — то, за что дают 1–2 балла
# --------------------------------------------------------------------------


def _strip_anchors(node: Ext) -> Ext:
    """Снять обязательные маркеры `^` и `$` по краям выражения."""
    parts = list(node.parts) if isinstance(node, Cat) else [node]
    if parts and isinstance(parts[0], Anchor) and parts[0].kind == START:
        parts = parts[1:]
    if parts and isinstance(parts[-1], Anchor) and parts[-1].kind == END:
        parts = parts[:-1]
    if not parts:
        return Eps()
    return parts[0] if len(parts) == 1 else Cat(tuple(parts))


def _checker_dfa(inner: Ext, letters: frozenset[str]) -> DFA:
    """Язык остатков, на которых выполняется `(?= inner)`.

    Тонкость, которую легко потерять: если `inner` **кончается на `$`**,
    то проверка требует лечь на весь остаток целиком, и приписывать `.*`
    нельзя. Именно так устроен второй пример преподавателя
    `(?=(b*ab*ab*)*$)`: там `$` и делает инвариант условием на весь хвост.
    """
    parts = list(inner.parts) if isinstance(inner, Cat) else [inner]
    if parts and isinstance(parts[-1], Anchor) and parts[-1].kind == END:
        core = parts[:-1]
        body = Eps() if not core else (core[0] if len(core) == 1 else Cat(tuple(core)))
        return dfa_of(to_regex(body, letters), letters).complete()
    tail = Cat((inner, Star(Any())))
    return dfa_of(to_regex(tail, letters), letters).complete()


def to_afa(
    node: Ext, alphabet: Iterable[str] | None = None, max_len: int = 8
) -> Verdict:
    """Построить ПКА по расширенной регулярке.

    > Здесь нужно построить автомат, совмещающий ИЛИ-недетерминизм
    > и И-недетерминизм (одновременно черты ПКА и НКА) по расширенной
    > регулярке, если хочется дополнительные 1–2 балла, и парсинг по нему.
    > Просто вызвать библиотеку регулярок баллов, увы, не даст.
    > — issue #40

    Механически берутся ровно те две формы, которые преподаватель
    и разбирает:

    * `^(?=τ₁)…(?=τₖ)τ$` — проверки в самом начале. Язык распадается
      в конъюнкцию условий, ПКА получает И-ветвление в стартовой вершине
      (`afa.conjunction`). Её первый пример: `^(?= .* a .* $) .* b .* $`.
    * `^((?=τ₁)τ₂)*$` — проверка в начале каждой итерации, то есть
      **рекурсивный инвариант**: в стартовой вершине читателя стоит
      И-ветвление, вторая ветвь проверяет весь остаток (`afa.with_lookahead`).
      Её второй пример: `^((?=(b*ab*ab*)*$)a*b)*$`.

    Общего механического перевода здесь нет и не заявляется: проверка
    в произвольном месте требует знать, в каких состояниях
    детерминизованного читателя выражение «стоит именно тут», а это
    и есть содержательный шаг, о котором issue #40 говорит «бывают
    инварианты, вот их и используем».

    Построенный автомат **сверяется с распознавателем** на всех словах
    до длины `max_len` прежде, чем вернуться. Неверный ПКА наружу
    не выходит: вместо него `refuted` со свидетелем.
    """
    from tfl.afa import conjunction, with_lookahead
    from tfl.words import iter_words

    letters = frozenset(alphabet) if alphabet is not None else alphabet_of(node)
    body = _strip_anchors(node)
    parts = list(body.parts) if isinstance(body, Cat) else [body]

    built = None
    shape = ""
    leading = [p for p in parts if isinstance(p, Look)]
    rest = [p for p in parts if not isinstance(p, Look)]

    if leading and all(
        isinstance(part, Look) for part in parts[: len(leading)]
    ) and not any(has_lookaround(part) for part in rest):
        pieces: list[DFA] = []
        for look in leading:
            if look.kind not in (AHEAD, NEG_AHEAD):
                pieces = []
                break
            checker = _checker_dfa(look.inner, letters)
            pieces.append(checker.complement() if look.kind == NEG_AHEAD else checker)
        if pieces:
            tail = Eps() if not rest else (rest[0] if len(rest) == 1 else Cat(tuple(rest)))
            pieces.append(dfa_of(to_regex(tail, letters), letters).complete())
            built = conjunction(pieces, letters)
            shape = "И-ветвление в стартовой вершине"

    if built is None and len(parts) == 1 and isinstance(parts[0], Star):
        inner = parts[0].inner
        pieces = list(inner.parts) if isinstance(inner, Cat) else [inner]
        if pieces and isinstance(pieces[0], Look) and pieces[0].kind == AHEAD:
            tail = pieces[1:]
            if not any(has_lookaround(part) for part in tail):
                reader_body = (
                    Eps() if not tail else (tail[0] if len(tail) == 1 else Cat(tuple(tail)))
                )
                reader = dfa_of(to_regex(Star(reader_body), letters), letters).complete()
                checker = _checker_dfa(pieces[0].inner, letters)
                built = with_lookahead(reader, checker)
                shape = "рекурсивный инвариант в стартовой вершине читателя"

    if built is None:
        return unknown(
            "форма выражения не из тех двух, что переводятся механически: "
            "проверка не в начале выражения и не в начале итерации. "
            "Инвариант для ПКА здесь придумывает человек (issue #40)"
        )

    for word in iter_words("".join(sorted(letters)), max_len):
        if built.accepts(word) != accepts(node, word, letters):
            return refuted(
                f"построенный ПКА расходится с распознавателем на слове «{word or 'ε'}»",
                (word, built),
            )
    return Verdict(
        True,
        f"ПКА построен ({shape}), {len(built)} состояний; сверен "
        f"с распознавателем на всех словах длины ⩽ {max_len}",
        built,
    )


# --------------------------------------------------------------------------
# Фазз-сверка распознавателей — прямое требование условия
# --------------------------------------------------------------------------


def fuzz(
    node: Ext,
    others: "dict[str, object]",
    alphabet: Iterable[str] | None = None,
    trials: int = 400,
    max_len: int = 12,
    seed: int = 0,
) -> Verdict:
    """Сверить распознаватель расширенной регулярки с остальными.

    > Требуется только фазз-тестирование эквивалентности: строится
    > случайное слово ω и проверяется, принадлежит ли он языкам
    > регулярного выражения, ДКА, НКА и ПКА согласованно.

    `others` — имя распознавателя и объект с методом `accepts(word)`
    либо просто предикат. Возвращается первое расхождение со свидетелем;
    «да» здесь означает лишь «на этих словах сошлись» — фазз
    доказательством не является, и вердикт это говорит.
    """
    import random

    letters = sorted(alphabet) if alphabet is not None else sorted(alphabet_of(node))
    if not letters:
        return unknown("алфавит пуст: сверять не на чем")
    rng = random.Random(seed)

    for _ in range(trials):
        length = rng.randrange(0, max_len + 1)
        word = "".join(rng.choice(letters) for _ in range(length))
        mine = accepts(node, word, letters)
        for name, other in others.items():
            theirs = (
                other.accepts(word) if hasattr(other, "accepts") else bool(other(word))
            )
            if theirs != mine:
                return refuted(
                    f"на слове «{word or 'ε'}» расширенная регулярка говорит "
                    f"{'да' if mine else 'нет'}, а «{name}» — "
                    f"{'да' if theirs else 'нет'}",
                    word,
                )
    return Verdict(
        True,
        f"{trials} случайных слов длины ⩽ {max_len}: все распознаватели "
        f"({', '.join(others)}) сошлись. Фазз доказательством не является",
    )
