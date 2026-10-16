"""Конструкции и явные границы утверждений лекции 4 (2026)."""

from itertools import product

import pytest

from tfl.automata import counterexample, dfa_of, shuffle
from tfl.words import iter_words


def _interleaving_accepts(left, right, word):
    # Независимое определение через распределение позиций, не через автомат.
    for positions in product((False, True), repeat=len(word)):
        u = "".join(c for c, take in zip(word, positions) if take)
        v = "".join(c for c, take in zip(word, positions) if not take)
        if left.accepts(u) and right.accepts(v):
            return True
    return False


@pytest.mark.parametrize("first,second", [
    ("ab", "ba"), ("a*", "b*"), ("ε", "ab"), ("∅", "a*"),
    ("a|ab", "a|ba"), ("(ab)*", "a*"),
])
def test_shuffle_against_independent_distribution_of_positions(first, second):
    left, right = dfa_of(first), dfa_of(second)
    machine = shuffle(left, right)
    for word in iter_words("ab", 7):
        assert machine.accepts(word) == _interleaving_accepts(left, right, word)
    assert counterexample(machine.determinize(), shuffle(right, left).determinize()) is None


def test_shuffle_is_not_intersection_and_handles_shared_alphabet():
    a = dfa_of("a")
    assert shuffle(a, a).accepts("aa")
    assert not shuffle(a, a).accepts("a")
    result = shuffle(dfa_of("a*"), dfa_of("b*"))
    assert counterexample(result.determinize(), dfa_of("(a|b)*")) is None


@pytest.mark.parametrize("first,second", [
    ("a*", "b*"), ("a|ab", "ba*"), ("ab", "ε"),
])
def test_kleene_identities_exactly(first, second):
    a, b = f"({first})", f"({second})"
    identities = [
        (f"({a}|{b})*", f"{a}*({b}{a}*)*"),
        (f"({a}{b})*{a}", f"{a}({b}{a})*"),
        (f"{a}*", f"({a}*)*"),
        (f"{a}*", f"{a}*{a}*"),
    ]
    for lhs, rhs in identities:
        assert counterexample(dfa_of(lhs), dfa_of(rhs)) is None


def test_full_word_concatenation_would_define_a_nonautomatic_relation():
    # x=y=a^n, z=xy=a^(2n): convolution is X^n Y^n, not regular.
    for n, m in product(range(1, 20), repeat=2):
        columns = [("a", "a", "a")] * n + [("#", "#", "a")] * m
        words = tuple("".join(col[i] for col in columns if col[i] != "#") for i in range(3))
        x, y, z = words
        assert x == y
        assert (z == x + y) == (n == m)
    # This finite table illustrates the parameterised argument in the notes.
