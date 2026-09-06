"""Проверяемые утверждения из issues преподавателя.

Слой A — прямая речь, но и её стоит прогнать через оракул: там, где
преподаватель даёт регулярку и словами описывает её язык, совпадение
проверяется перебором. Тексты — `corpus/issues/`.
"""

from __future__ import annotations

import re

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
