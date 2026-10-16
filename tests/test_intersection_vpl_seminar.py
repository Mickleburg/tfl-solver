"""Проверки семинарских гипотез; общие доказательства находятся в корпусе.

Конечные таблицы и разбиения не являются решателем принадлежности VPL.
"""

from itertools import product
from math import gcd
import re

from tfl.intake import analyse, load_seminar_index
from tfl.lang import from_predicate, from_regex, table
from tfl.pump import cf_splits
from tfl.words import iter_words


def inequality_language(reverse=False):
    def accepts(word):
        if re.fullmatch(r"a*b*", word) is None:
            return False
        n, m = word.count("a"), word.count("b")
        if reverse:
            return n >= 2 * m or m >= 2 * n or gcd(n, m) == 1
        return n <= 2 * m or m <= 2 * n or gcd(n, m) == 1

    return from_predicate(accepts, "ab")


def square_language():
    return from_predicate(
        lambda w: len(w) % 2 == 0 and w[:len(w) // 2] == w[len(w) // 2:],
        "ab",
    )


def four_block_copy(word):
    match = re.fullmatch(r"(a+)(b+)(a+)(b+)", word)
    return bool(match and match[1] == match[3] and match[2] == match[4])


def balanced(word, call, ret):
    depth = 0
    for letter in word:
        depth += (letter == call) - (letter == ret)
        if depth < 0:
            return False
    return depth == 0


def test_disjunction_covers_all_nonnegative_exponents():
    assert all(n <= 2 * m or m <= 2 * n for n, m in product(range(101), repeat=2))
    assert not inequality_language().disagreements(from_regex("a*b*", "ab"), 12)


def test_reversed_inequalities_have_off_diagonal_table():
    language = inequality_language(reverse=True)
    assert "" in language  # gcd(0, 0) != 1, но неравенство выполнено
    assert "ab" in language  # исключительная единичная диагональ
    assert "aabb" not in language
    assert "aabbb" in language  # взаимная простота в центральной области
    assert "aaaabb" in language  # равенство на границе неравенства
    assert "ba" not in language
    prefixes = ["a" * 2**k for k in range(1, 11)]
    suffixes = ["b" * 2**k for k in range(1, 11)]
    matrix = table(language, prefixes, suffixes)
    assert matrix.cells == tuple(tuple(i != j for j in range(10)) for i in range(10))


def test_square_intersection_is_exactly_four_block_copy():
    intersection = square_language().restrict("aa*bb*aa*bb*")
    expected = from_predicate(four_block_copy, "ab")
    assert not intersection.disagreements(expected, 12)
    assert "" in square_language() and "" not in intersection
    assert "aaaa" in square_language() and "aaaa" not in intersection


def test_both_call_return_assignments_have_balanced_identity_table():
    for call, ret in (("a", "b"), ("b", "a")):
        words = [call * n + ret * n for n in range(1, 21)]
        assert all(balanced(w, call, ret) for w in words)
        matrix = table(square_language(), words, words)
        assert matrix.cells == tuple(tuple(i == j for j in range(20)) for i in range(20))


def test_training_intersection_bounds_stack_and_has_identity_table():
    for ret, call in (("a", "b"), ("b", "a")):
        language = square_language().restrict(f"{ret}{ret}*{call}{ret}{ret}*{call}")

        def expected(word):
            match = re.fullmatch(f"({ret}+)({call})({ret}+)({call})", word)
            return bool(match and match[1] == match[3])

        assert not language.disagreements(from_predicate(expected, "ab"), 12)
        prefixes = [ret * n for n in range(1, 16)]
        suffixes = [call + ret * n + call for n in range(1, 16)]
        assert table(language, prefixes, suffixes).cells == tuple(
            tuple(i == j for j in range(15)) for i in range(15)
        )
        for n, m in product(range(1, 21), repeat=2):
            depth = peak = 0
            for char in ret * n + call + ret * m + call:
                depth = depth + 1 if char == call else max(0, depth - 1)
                peak = max(peak, depth)
            assert peak == depth == 1


def test_vpl_card_does_not_require_unlearned_cf_pumping():
    records = load_seminar_index().records
    card = next(record for record in records if record["id"] == "sem-2026-10-03-square-vpl")
    assert not any("накач" in prerequisite for prerequisite in card["prerequisites"])
    assert "высот" in card["method"]


def test_every_bounded_cf_split_fails_when_pumped_twice():
    for p in range(1, 8):
        word = "a" * p + "b" * p + "a" * p + "b" * p
        assert four_block_copy(word)
        for u, v, x, y, z in cf_splits(word, p):
            assert not four_block_copy(u + v * 2 + x + y * 2 + z)


def test_new_seminar_retrieval_and_route_isolation():
    index = load_seminar_index()
    cases = (
        ("Проверить регулярность {a^n b^m | n <= 2*m или m <= 2*n или n и m взаимно просты}",
         "sem-2026-10-03-union-cover"),
        ("Проверить регулярность {a^n b^m | n >= 2*m или m >= 2*n или n и m взаимно просты}",
         "sem-2026-10-03-union-separation"),
        ("Проверить, является ли VPL язык {ww | w in {a,b}*}: через М-Н и пересечение с регулярным языком",
         "sem-2026-10-03-square-vpl"),
    )
    for text, expected in cases:
        result = analyse(text, "семинар", seminar_index=index)
        assert result.seminar_similar
        assert result.seminar_similar[0][1]["id"] == expected
    unrelated = analyse("Доказать завершимость SRS ab -> ba", seminar_index=index)
    assert not any("2026-10-03" in record["id"] for _, record in unrelated.seminar_similar)


if __name__ == "__main__":
    test_disjunction_covers_all_nonnegative_exponents()
    test_reversed_inequalities_have_off_diagonal_table()
    test_square_intersection_is_exactly_four_block_copy()
    test_both_call_return_assignments_have_balanced_identity_table()
    test_training_intersection_bounds_stack_and_has_identity_table()
    test_vpl_card_does_not_require_unlearned_cf_pumping()
    test_every_bounded_cf_split_fails_when_pumped_twice()
    print("union: exponent grid 0..100; all binary words of length <= 12 agree with a*b*")
    print("reverse union: off-diagonal table for exponents 2**k, k=1..10")
    print("square intersection: all binary words of length <= 12 agree with four-block copy")
    print("balanced words: identity tables for both call/return assignments, n=1..20")
    print("training: two-marker intersection through length 12; stack height for n,m=1..20")
    print("CF pumping: all splits fail at power 2 for p=1..7")
