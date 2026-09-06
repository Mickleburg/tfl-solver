"""РК2 2025, вариант 28, задача 1: инварианты SRS над базисом.

Разбор — `reports/rk2-variant-28-task1/report.md`. Здесь закреплено то,
что доказано исполнением: три необходимых условия и опровержение обоих
ответов из проверенных работ.
"""

from __future__ import annotations

import pytest

from tfl.srs import parse_srs
from tfl.words import iter_words

RULES = "baa -> ba\nab -> ba\na -> ab\n"
LIMIT = 9


@pytest.fixture(scope="module")
def system():
    return parse_srs(RULES)


@pytest.fixture(scope="module")
def closure(system):
    """Замыкание базисов `aⁿbⁿaⁿ` при n = 1, 2, 3 — нижняя оценка языка."""
    words: set[str] = set()
    for n in (1, 2, 3):
        base = "a" * n + "b" * n + "a" * n
        found = system.reachable(base, max_len=LIMIT, max_words=400_000)
        words |= {w for w in found.words if len(w) <= LIMIT}
    return words


def prefix_a(word: str) -> int:
    index = 0
    while index < len(word) and word[index] == "a":
        index += 1
    return index


def test_ba_is_a_factor_of_every_word(system, closure):
    """Ни одно правило не разрушает вхождение `ba`."""
    broken = [
        (word, nxt)
        for word in iter_words("ab", 7)
        if "ba" in word
        for nxt in system.step(word)
        if "ba" not in nxt
    ]
    assert broken == []
    assert all("ba" in word for word in closure)


def test_measure_never_decreases(system, closure):
    """d(w) = 2|w|_b − |w|_a равна нулю на базисе и только растёт."""

    def measure(word: str) -> int:
        return 2 * word.count("b") - word.count("a")

    assert all(measure("a" * n + "b" * n + "a" * n) == 0 for n in range(1, 5))
    dropped = [
        (word, nxt)
        for word in iter_words("ab", 7)
        for nxt in system.step(word)
        if measure(nxt) < measure(word)
    ]
    assert dropped == []
    assert all(measure(word) >= 0 for word in closure)


def test_leading_block_never_grows(system, closure):
    grown = [
        (word, nxt)
        for word in iter_words("ab", 7)
        for nxt in system.step(word)
        if prefix_a(nxt) > prefix_a(word)
    ]
    assert grown == []


def test_necessary_conditions_hold_on_the_closure(closure):
    for word in closure:
        needed = max(prefix_a(word), -(-word.count("a") // 2))
        assert needed <= word.count("b"), word


def test_both_graded_answers_are_refuted(closure):
    """Оба ответа из проверенных работ ошибаются на конкретных словах."""
    # photo_112: L = {aⁿbᵐ | m ≥ n ≥ 1}
    assert "aaabbba" in closure  # в языке, а гипотеза его не принимает
    assert "aabb" not in closure  # гипотеза принимает, а слова в языке нет

    # photo_166: L = (a|b)*b(a|b)*a
    assert "aaabbbab" in closure  # оканчивается на b — гипотеза отвергает
    assert "aaba" not in closure  # гипотеза принимает, а меры не хватает


def test_conditions_are_not_sufficient(closure):
    """Честная граница: необходимые условия языка не описывают."""
    assert "ba" in "aabab" and "aabab".count("a") <= 2 * "aabab".count("b")
    assert "aabab" not in closure
