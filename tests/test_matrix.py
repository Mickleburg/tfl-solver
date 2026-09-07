"""Матричные интерпретации для строковых систем.

Проверка интерпретации — свой арбитр, он считает всегда. Поиск идёт
через Z3, и тесты на него пропускаются, если решателя в окружении нет:
обязательной зависимостью проекта он не сделан.
"""

from __future__ import annotations

import pytest

from tfl.matrix import (
    Matrix,
    MatrixInterpretation,
    find_matrix_interpretation,
    have_solver,
    smtlib_model,
)
from tfl.srs import parse_srs

needs_solver = pytest.mark.skipif(not have_solver(), reason="Z3 в окружении нет")


# --------------------------------------------------------------------------
# Матрицы
# --------------------------------------------------------------------------


def test_identity_is_the_value_of_the_empty_word():
    interpretation = MatrixInterpretation({"a": Matrix(((1, 1), (0, 1)))})
    assert interpretation.value("") == Matrix.identity(2)


def test_word_is_the_product_in_order():
    a = Matrix(((1, 1), (0, 1)))
    b = Matrix(((2, 0), (0, 1)))
    interpretation = MatrixInterpretation({"a": a, "b": b})
    assert interpretation.value("ab") == a * b
    assert interpretation.value("ba") == b * a
    assert a * b != b * a  # порядок букв важен, и в этом весь смысл


def test_comparison_is_entrywise_plus_a_strict_corner():
    big = Matrix(((1, 5), (0, 1)))
    small = Matrix(((1, 3), (0, 1)))
    assert big >= small and not small >= big
    assert big.top_right == 5 and small.top_right == 3


# --------------------------------------------------------------------------
# Проверка интерпретации — арбитр
# --------------------------------------------------------------------------


def test_a_supplied_interpretation_is_checked_exactly():
    """`ab → ba` убывает при `a ↦ [[2,0],[0,1]]`, `b ↦ [[1,1],[0,1]]`."""
    interpretation = MatrixInterpretation(
        {"a": Matrix(((2, 0), (0, 1))), "b": Matrix(((1, 1), (0, 1)))}
    )
    verdict = interpretation.check(parse_srs("ab -> ba"))
    assert verdict.value is True, verdict.reason


def test_a_non_monotone_interpretation_proves_nothing():
    """Углы обязаны быть ненулевыми, иначе убывание не переносится в контекст.

    Это не педантизм: без `A[0][0] ⩾ 1` и `A[d−1][d−1] ⩾ 1` в разложении
    `(U·L·V)[0][d−1]` строгое слагаемое умножается на ноль, и правило
    внутри слова может не убывать вовсе.
    """
    interpretation = MatrixInterpretation(
        {"a": Matrix(((0, 1), (0, 1))), "b": Matrix(((1, 0), (0, 1)))}
    )
    verdict = interpretation.check(parse_srs("ab -> ba"))
    assert verdict.value is None
    assert "не монотонна" in verdict.reason


def test_a_wrong_interpretation_names_the_rule_that_fails():
    interpretation = MatrixInterpretation(
        {"a": Matrix(((1, 0), (0, 1))), "b": Matrix(((1, 0), (0, 1)))}
    )
    verdict = interpretation.check(parse_srs("ab -> ba"))
    assert verdict.value is None
    assert "не убывает" in verdict.reason


def test_missing_letters_are_reported():
    interpretation = MatrixInterpretation({"a": Matrix(((1, 1), (0, 1)))})
    verdict = interpretation.check(parse_srs("ab -> ba"))
    assert verdict.value is None
    assert "нет матриц" in verdict.reason


# --------------------------------------------------------------------------
# Поиск через SMT
# --------------------------------------------------------------------------


def test_absent_solver_is_not_an_error():
    """Без Z3 поиск обязан вернуть «не выяснено», а не упасть."""
    if have_solver():
        pytest.skip("решатель есть, ветку без него так не проверить")
    verdict = find_matrix_interpretation(parse_srs("ab -> ba"))
    assert verdict.value is None
    assert "Z3" in verdict.reason


@needs_solver
def test_search_finds_an_interpretation_and_it_passes_the_checker():
    """Ответ решателя перепроверяется своим арбитром, а не берётся на слово."""
    system = parse_srs("ab -> ba")
    verdict = find_matrix_interpretation(system, dimension=2, max_entry=3)
    assert verdict.value is True, verdict.reason
    assert verdict.witness.check(system).value is True


@needs_solver
def test_the_exam_srs_that_nothing_else_took():
    """`fg → gff`, `fh → hg` — вопрос «Аптеки» 2022 на 5 баллов.

    Ни армейский порядок, ни линейные интерпретации, ни путевой порядок,
    ни поиск петли эту систему не берут — она была единственной
    из семи SRS-вопросов, оставшейся открытой. Матрица 3×3 её закрывает.
    """
    system = parse_srs("fg -> gff\nfh -> hg")
    assert find_matrix_interpretation(system, dimension=2, max_entry=2).value is None

    verdict = find_matrix_interpretation(system, dimension=3, max_entry=2)
    assert verdict.value is True, verdict.reason
    assert verdict.witness.check(system).value is True


@needs_solver
def test_unsat_is_reported_as_a_bounded_statement():
    """«Нет интерпретации» верно только для данной размерности и границы."""
    verdict = find_matrix_interpretation(
        parse_srs("fg -> gff\nfh -> hg"), dimension=2, max_entry=2
    )
    assert verdict.value is None
    assert "не следует ничего" in verdict.reason


@needs_solver
def test_a_looping_system_has_no_interpretation():
    """У незавершимой системы её быть не может — контроль в другую сторону."""
    verdict = find_matrix_interpretation(parse_srs("ab -> ba\nba -> ab"), dimension=2)
    assert verdict.value is None


# --------------------------------------------------------------------------
# Модель текстом
# --------------------------------------------------------------------------


def test_smtlib_model_is_emitted_for_an_external_solver():
    model = smtlib_model(parse_srs("ab -> ba"), dimension=2, max_entry=3)
    assert "(set-logic QF_NIA)" in model
    assert "(declare-const a_0_0 Int)" in model
    assert "(assert (>= a_0_0 1))" in model  # монотонность
    assert "(check-sat)" in model


def test_srs_exposes_the_search_as_a_method():
    verdict = parse_srs("ab -> ba").find_matrix_interpretation(dimension=2, max_entry=3)
    assert verdict.value in (True, None)
