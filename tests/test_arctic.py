"""Арктические интерпретации: полукольцо, арбитр, граница метода.

Главное здесь — не «нашли интерпретацию», а два запрета:

* арбитр обязан отвергать интерпретацию с бесконечным углом: без
  конечных углов строгое убывание в контекст не переносится, и «доказательство»
  ничего не доказывает;
* метод обязан молчать на незавершимых системах.

Отдельно закреплена **граница**: арктика ограничивает длину вывода
линейно, поэтому на `ab → ba` (квадратичный вывод) её нет в принципе,
а не «не нашлась».
"""

from __future__ import annotations

import pytest

from tfl.arctic import (
    NEG,
    ArcticInterpretation,
    ArcticMatrix,
    find_arctic_interpretation,
    have_solver,
    relative_step,
)
from tfl.srs import parse_srs

needs_solver = pytest.mark.skipif(not have_solver(), reason="нет Z3")


def matrix(*rows) -> ArcticMatrix:
    return ArcticMatrix(tuple(tuple(row) for row in rows))


# --------------------------------------------------------------------------
# Полукольцо
# --------------------------------------------------------------------------


def test_unit_is_neutral_for_the_product():
    """Пустому слову отвечает единица: интерпретация согласована с конкатенацией."""
    unit = ArcticMatrix.unit(2)
    some = matrix([1, NEG], [0, 2])
    assert unit * some == some
    assert some * unit == some


def test_the_product_is_max_plus_and_not_plus_times():
    """`(A ⊗ B)[i][j] = max_k (A[i][k] + B[k][j])` — самый длинный путь."""
    left = matrix([1, 2], [0, 3])
    right = matrix([4, 0], [1, 5])
    assert (left * right).rows == ((max(1 + 4, 2 + 1), max(1 + 0, 2 + 5)),
                                   (max(0 + 4, 3 + 1), max(0 + 0, 3 + 5)))


def test_minus_infinity_absorbs():
    lonely = matrix([NEG, NEG], [NEG, NEG])
    assert (lonely * matrix([1, 1], [1, 1])).rows == ((NEG, NEG), (NEG, NEG))


def test_strict_domination_accepts_minus_infinity_on_the_right():
    """`a ⊐ b` — это `a > b` **или** `b = −∞`, и второе не описка."""
    assert matrix([NEG]).strictly_dominates(matrix([NEG]))
    assert matrix([0]).strictly_dominates(matrix([NEG]))
    assert not matrix([NEG]).strictly_dominates(matrix([0]))
    assert not matrix([1]).strictly_dominates(matrix([1]))


def test_a_word_is_the_product_of_its_letters():
    found = ArcticInterpretation({"a": matrix([1, 0], [NEG, 0]), "b": matrix([2, 1], [0, 0])})
    assert found.value("ab") == found.matrices["a"] * found.matrices["b"]
    assert found.value("") == ArcticMatrix.unit(2)


# --------------------------------------------------------------------------
# Арбитр
# --------------------------------------------------------------------------


def test_the_arbiter_refuses_an_interpretation_with_an_infinite_corner():
    """Без конечных углов мера контекста рушится в −∞, вывод неправомерен."""
    broken = ArcticInterpretation(
        {"a": matrix([1, 5], [NEG, NEG]), "b": matrix([0, 0], [NEG, 0])}
    )
    verdict = broken.check(parse_srs("aa -> ab"))
    assert verdict.value is None
    assert "не монотонна" in verdict.reason


def test_the_arbiter_refuses_a_rule_that_does_not_fall():
    found = ArcticInterpretation({"a": matrix([0]), "b": matrix([0])})
    verdict = found.check(parse_srs("ab -> ba"))
    assert verdict.value is None
    assert "не убывает" in verdict.reason


def test_the_arbiter_names_a_missing_letter():
    verdict = ArcticInterpretation({"a": matrix([1])}).check(parse_srs("ab -> a"))
    assert verdict.value is None
    assert "нет матриц" in verdict.reason


def test_a_hand_written_interpretation_is_accepted():
    """`a ↦ 1`, `b ↦ 0`: `[a] = 1 > 0 = [bb]` — вес слова это сумма."""
    found = ArcticInterpretation({"a": matrix([1]), "b": matrix([0])})
    verdict = found.check(parse_srs("a -> bb"))
    assert verdict.value is True


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------


@needs_solver
def test_the_solver_finds_an_interpretation_for_a_shortening_system():
    verdict = find_arctic_interpretation(parse_srs("aa -> a"), 1, 3, 10_000)
    assert verdict.value is True
    assert verdict.witness.check(parse_srs("aa -> a")).value is True


@needs_solver
def test_the_solver_stays_silent_on_a_looping_system():
    """Доказать завершимость незавершимой системы страшнее всего."""
    for text in ("aa -> aaa", "ab -> ba\nba -> ab"):
        for dimension in (1, 2):
            verdict = find_arctic_interpretation(parse_srs(text), dimension, 3, 10_000)
            assert verdict.value is not True, (text, dimension)


@needs_solver
def test_the_method_has_no_chance_on_a_quadratic_derivation():
    """`ab → ba` сортирует слово за `O(n²)` шагов, а арктика даёт `O(n)`.

    Поэтому «не нашлось» здесь — не слабость перебора, а свойство метода,
    и решатель отвечает именно `unsat`, а не «не уложился».
    """
    verdict = find_arctic_interpretation(parse_srs("ab -> ba"), 2, 3, 20_000)
    assert verdict.value is None
    assert "не существует" in verdict.reason


@needs_solver
def test_a_relative_step_drops_at_least_one_rule_and_raises_none():
    system = parse_srs("ab -> ba\na -> bb")
    found = relative_step(system.rules, 1, 3, 10_000)
    assert found is not None
    assert all(found.weakly_decreases(r.lhs, r.rhs) for r in system.rules)
    assert any(found.decreases(r.lhs, r.rhs) for r in system.rules)
