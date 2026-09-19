"""Регрессии обезличенных задач на регулярность и копредставление.

Конечный перебор здесь проверяет формализацию и параметрические таблицы на
начальных индексах. Доказательства для произвольных индексов находятся в
``docs/recipes/EXAM-1.md`` и предметной памяти.
"""

from __future__ import annotations

from itertools import product

from tfl.cayley import normalize
from tfl.srs import parse_srs


def has_proper_even_palindromic_prefix(word: str) -> bool:
    return any(
        word[: 2 * radius] == word[: 2 * radius][::-1]
        for radius in range(1, (len(word) - 1) // 2 + 1)
    )


def has_proper_even_palindromic_suffix(word: str) -> bool:
    return any(
        word[-2 * radius :] == word[-2 * radius :][::-1]
        for radius in range(1, (len(word) - 1) // 2 + 1)
    )


def has_proper_square_prefix(word: str) -> bool:
    return any(
        word[:radius] == word[radius : 2 * radius]
        for radius in range(1, (len(word) - 1) // 2 + 1)
    )


def test_three_disjuncts_cover_every_ordered_block_word():
    for n, m, k in product(range(8), repeat=3):
        condition = m != n or k % 2 != n % 2 or k % 2 == m % 2
        assert condition


def test_copresentation_reduces_bounded_mirror_words_to_one_class():
    system = parse_srs("aa#aa -> a#a\nba#ab -> a#a\nb#b -> a#a")

    for length in range(1, 7):
        for letters in product("ab", repeat=length):
            left = "".join(letters)
            assert normalize(system, left + "#" + left[::-1]) == "a#a"


def test_copy_language_has_identity_nerode_table():
    for i, j in product(range(1, 9), repeat=2):
        left = "b" + "a" * i
        right = "b" + "a" * j
        word = left + right
        middle = len(word) // 2
        is_copy = len(word) % 2 == 0 and word[:middle] == word[middle:]
        assert is_copy == (i == j)


def test_even_palindromic_prefix_family_has_identity_table():
    for i, j in product(range(7), repeat=2):
        left = "a" + "b" * (2 * i + 1) + "a"
        right = "a" + "b" * (2 * j + 1) + "a"
        assert has_proper_even_palindromic_prefix(left + right + "a") == (
            i == j
        )


def test_square_prefix_family_has_strict_upper_triangular_table():
    for i, j in product(range(1, 9), repeat=2):
        left = "b" + "a" * i
        right = "b" + "a" * j
        assert has_proper_square_prefix(left + right) == (j > i)


def test_no_even_palindromic_edge_family_has_identity_table():
    for i, j in product(range(8), repeat=2):
        left = "ab" * i + "a"
        right = "a" + "ba" * j
        word = left + right
        in_language = not has_proper_even_palindromic_prefix(
            word
        ) and not has_proper_even_palindromic_suffix(word)
        assert in_language == (i == j)
