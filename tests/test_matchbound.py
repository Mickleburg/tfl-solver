"""Ограничение совпадениями: кодировка, пополнение, арбитр.

Проверок здесь три рода.

* **Арбитр независим от пополнения.** Он заново считает три условия
  операциями над автоматами, и подделку — автомат, не принимающий
  стартовые слова, либо не замкнутый — обязан отвергать.
* **Метод молчит там, где обязан.** Незавершимая система не может быть
  ограничена совпадениями: доказать завершимость незавершимой системы
  страшнее, чем не доказать завершимую.
* **Граница метода закреплена.** Ограниченная совпадениями система
  имеет **линейную** длину вывода, поэтому на `ab → ba` метода нет
  в принципе — это свойство, а не слабость перебора.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.automata import DFA
from tfl.matchbound import (
    Coding,
    MatchBound,
    _base_automaton,
    find_match_bound,
    new_height,
)
from tfl.srs import parse_srs

LAB1 = pathlib.Path(__file__).parent.parent / "evals" / "lab1_2025"


def variant(number: int):
    return parse_srs((LAB1 / f"variant-{number:02d}.srs").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Кодировка и правило высот
# --------------------------------------------------------------------------


def test_coding_is_a_bijection():
    coding = Coding("abc", 2)
    pairs = [(letter, height) for letter in "abc" for height in range(3)]
    assert len({coding.encode(*pair) for pair in pairs}) == len(pairs)
    assert all(coding.decode(coding.encode(*pair)) == pair for pair in pairs)


def test_lifting_and_showing_round_trip():
    coding = Coding("ab", 3)
    assert coding.show(coding.lift("aab", 0)) == "a⁰a⁰b⁰"
    assert coding.show(coding.lift("ba", 2)) == "b²a²"
    assert coding.show("") == "ε"


def test_the_new_height_is_one_above_the_minimum():
    """Минимум, а не максимум: иначе высоты росли бы от одного вхождения."""
    assert new_height((0, 0, 0)) == 1
    assert new_height((3, 1, 2)) == 2


def test_the_base_automaton_accepts_exactly_the_zero_height_words():
    coding = Coding("ab", 2)
    base = _base_automaton(coding)
    assert base.accepts(coding.lift("abba", 0))
    assert not base.accepts(coding.lift("ab", 1))


# --------------------------------------------------------------------------
# Доказательство и его проверка
# --------------------------------------------------------------------------


def test_a_shortening_system_is_match_bounded_by_one():
    verdict = find_match_bound(parse_srs("a -> bb"), 1)
    assert verdict.value is True
    assert "ограничена совпадениями числом 1" in verdict.reason
    assert "не длиннее" in verdict.reason  # линейная граница длины вывода


def test_the_arbiter_accepts_the_found_automaton_on_its_own():
    """Свидетель проверяется заново, без всякой памяти о пополнении."""
    system = parse_srs("aa -> b")
    verdict = find_match_bound(system, 2)
    assert verdict.value is True
    witness = verdict.witness
    fresh = MatchBound(witness.coding, witness.automaton)
    assert fresh.check(system).value is True


def test_the_arbiter_refuses_an_automaton_that_misses_starting_words():
    """Условие 1: `L(A) ⊇ (Σ × {0})*`. Пустой автомат его нарушает."""
    coding = Coding("ab", 1)
    empty = DFA(coding.alphabet, "q", frozenset(), {})
    verdict = MatchBound(coding, empty).check(parse_srs("a -> b"))
    assert verdict.value is None
    assert "стартовые слова" in verdict.reason


def test_the_arbiter_refuses_an_automaton_that_is_not_closed():
    """Условие 2: сам базовый автомат замкнутым не является."""
    coding = Coding("ab", 2)
    verdict = MatchBound(coding, _base_automaton(coding)).check(parse_srs("a -> b"))
    assert verdict.value is None
    assert "не замкнут" in verdict.reason


def test_the_arbiter_notices_letters_outside_the_coding():
    coding = Coding("a", 1)
    verdict = MatchBound(coding, _base_automaton(coding)).check(parse_srs("a -> b"))
    assert verdict.value is None
    assert "не покрывает" in verdict.reason


# --------------------------------------------------------------------------
# Границы метода и запреты
# --------------------------------------------------------------------------


def test_a_quadratic_system_is_never_match_bounded():
    """`ab → ba`: вывод из `aⁿbⁿ` квадратичен, а метод даёт линейный.

    Высоты обязаны расти с длиной слова, и пополнение это видит:
    оно упирается именно в потолок высот, а не в размер автомата.
    """
    for bound in (1, 2, 3):
        verdict = find_match_bound(parse_srs("ab -> ba"), bound)
        assert verdict.value is None
        assert "потолок высот" in verdict.reason


@pytest.mark.parametrize("text", ["aa -> aaa", "ab -> ba\nba -> ab", "a -> aa"])
def test_a_non_terminating_system_is_never_proved(text):
    for bound in (1, 2, 3):
        assert find_match_bound(parse_srs(text), bound).value is not True


@pytest.mark.parametrize("number", [1, 2, 13, 20, 28])
def test_a_variant_with_a_known_loop_is_never_proved(number):
    """У этих вариантов ЛР1 петля найдена, значит доказывать нечего."""
    system = variant(number)
    assert system.terminates().value is False
    assert system.prove_by_match_bound(2, 120, 20, timeout_s=10).value is not True


def test_the_mirror_is_a_second_attempt_and_reverses_words():
    """Завершимость от разворота слов не зависит, а пополнение — зависит."""
    system = parse_srs("ab -> b")
    assert system.mirror().rules[0].lhs == "ba"
    assert system.prove_by_match_bound(2, 200, 30, timeout_s=15).value is True
