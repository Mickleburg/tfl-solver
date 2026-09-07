"""Проверяемые утверждения из issues преподавателя.

Слой A — прямая речь, но и её стоит прогнать через оракул: там, где
преподаватель даёт регулярку и словами описывает её язык, совпадение
проверяется перебором. Тексты — `corpus/issues/`.
"""

from __future__ import annotations

import re

from tfl.automata import dfa_of
from tfl.cfg import parse_cfg
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Issue #40 — ПКА через lookahead-регулярки
# --------------------------------------------------------------------------

CONJUNCTION = re.compile(r"^(?=.*a.*$).*b.*$")
RECURSIVE = re.compile(r"^((?=(b*ab*ab*)*$)a*b)*$")


def test_lookahead_is_a_conjunction_of_two_automata():
    """`^(?=.*a.*$).*b.*$` — это И-ветвление: «есть a» И «есть b»."""
    assert all(
        bool(CONJUNCTION.fullmatch(w)) == ("a" in w and "b" in w)
        for w in iter_words("ab", 8)
    )


def blocks(word: str) -> list[int] | None:
    """Разобрать слово как `(a* b)*`; вернуть длины блоков из `a`."""
    if word and not word.endswith("b"):
        return None
    return [len(part) for part in word.split("b")[:-1]] if word else []


def test_recursive_invariant_matches_the_teachers_words():
    """Формулировка из issue #40, слово в слово.

    «Количество букв a между каждыми двумя буквами b чётно, причём
    перед последней их ненулевое количество.»
    """

    def claim(word: str) -> bool:
        counts = blocks(word)
        if counts is None:
            return False
        if not counts:
            return True  # пустое слово
        return all(k % 2 == 0 for k in counts) and counts[-1] != 0

    assert [w for w in iter_words("ab", 11) if bool(RECURSIVE.fullmatch(w)) != claim(w)] == []


def test_recursive_language_is_thin():
    """Проверка «на глаз»: коротких слов мало и они узнаваемы."""
    accepted = [w for w in iter_words("ab", 6) if RECURSIVE.fullmatch(w)]
    assert accepted == [
        "", "aab", "baab", "aaaab", "bbaab", "aabaab", "baaaab", "bbbaab",
    ]


# --------------------------------------------------------------------------
# Issue #7 — Арден с рекурсией под звёздочкой
# --------------------------------------------------------------------------

MAX_LEN = 12


def _cut(words: set[str]) -> set[str]:
    return {w for w in words if len(w) <= MAX_LEN}


def _concat(left: set[str], right: set[str]) -> set[str]:
    return _cut({a + b for a in left for b in right})


def _star(inner: set[str]) -> set[str]:
    out, frontier = {""}, {""}
    while frontier:
        nxt = _cut({f + w for f in frontier for w in inner if w}) - out
        out |= nxt
        frontier = nxt
    return out


def _least_fixpoint(step) -> set[str]:
    """Наименьшая неподвижная точка языкового уравнения, срез до MAX_LEN."""
    current: set[str] = set()
    while True:
        nxt = _cut(step(current))
        if nxt == current:
            return current
        current = nxt


def _words_of(pattern: str) -> set[str]:
    machine = dfa_of(pattern)
    return {w for w in iter_words("ab", MAX_LEN) if machine.accepts(w)}


def test_arden_under_a_star_the_teachers_answer_is_right():
    """`A = (bA)*b` ⇒ `((bb)*bb)*b`, и это в самом деле `b(bb)*`.

    Приём из issue #7: раскрыть итерацию (`A = (bA)*bbA + b`), применить
    Арден (`A = ((bA)*bb)*b`), подставить выход из рекурсии.
    """
    fixpoint = _least_fixpoint(lambda s: _concat(_star(_concat({"b"}, s)), {"b"}))
    assert fixpoint == _words_of("((bb)*bb)*b")
    assert fixpoint == _words_of("b(bb)*")


def _second_example() -> set[str]:
    """`B → (b)*B | b | a(aB)*a` — второй пример из того же ответа."""
    return _least_fixpoint(
        lambda s: (
            _concat(_star({"b"}), s)
            | {"b"}
            | _concat(_concat({"a"}, _star(_concat({"a"}, s))), {"a"})
        )
    )


def test_second_example_has_a_slip_in_the_written_answer():
    """В ответе стоит `aa(aB)*` там, где по алгебре выходит `aa(Ba)*`.

    Тождество `x(yx)* = (xy)*x` даёт `a(aB)*a = aa(Ba)*`. Опечатка сквозная
    (повторяется во всех трёх строках вывода), поэтому логика шагов цела,
    но дословно переписывать формулу нельзя.
    """
    language = _second_example()
    left = _concat(_concat({"a"}, _star(_concat({"a"}, language))), {"a"})
    assert left == _concat({"aa"}, _star(_concat(language, {"a"})))
    assert left != _concat({"aa"}, _star(_concat({"a"}, language)))


def test_second_example_answer_taken_literally_is_a_different_language():
    """Дословная запись расходится с языком в обе стороны — есть свидетели."""
    language = _second_example()
    literal = _words_of("(b*|aaa(a(aa|b))*)*(aa|b)")
    assert "aaba" in language and "aaba" not in literal
    assert "aaab" in literal and "aaab" not in language


# --------------------------------------------------------------------------
# Issue #10 — First_k и левая рекурсия
# --------------------------------------------------------------------------


def test_first_k_ignores_left_recursion():
    """«Леворекурсивный шаг ничего не даёт» — прямой ответ преподавателя.

    Там же предупреждение: «кое-какие онлайн-калькуляторы считают
    не по этому алгоритму и левую рекурсию обрабатывать не умеют».
    У нас `first_k` — неподвижная точка, поэтому проблема не возникает.
    """
    grammar = parse_cfg("""
        A -> A B | c
        B -> b
    """)
    assert grammar.first_k(1)["A"] == {("c",)}
    assert grammar.first_k(2)["A"] == {("c",), ("c", "b")}


def test_first_k_with_a_nullable_left_recursive_nonterminal():
    """Если `A` порождает ε, леворекурсивное правило заменяется хвостом.

    `A → A B | ε`, `B → b` порождает `b*`, поэтому `first₂(A)`
    состоит из ε, `b` и `bb`.
    """
    grammar = parse_cfg("""
        S -> A c
        A -> A B |
        B -> b
    """)
    assert "A" in grammar.nullable()
    assert grammar.first_k(1)["A"] == {(), ("b",)}
    assert grammar.first_k(2)["A"] == {(), ("b",), ("b", "b")}
    assert grammar.follow_k(1)["A"] == {("b",), ("c",)}
