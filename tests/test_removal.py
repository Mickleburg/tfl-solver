"""Удаление правил: цепочка шагов и арбитр, который её пересчитывает.

Метод сильнее прямой интерпретации по построению: прямое доказательство —
это цепочка из одного шага. Поэтому проверяется не «нашли», а что
**арбитр не принимает лишнего**: шаг, где объявленное удалённым правило
строго не убывает; шаг, где какое-то правило растёт; цепочка, после
которой правила остались.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.matrix import Matrix, MatrixInterpretation, have_solver
from tfl.removal import RemovalProof, RemovalStep, prove_by_removal
from tfl.srs import parse_srs

LAB1 = pathlib.Path(__file__).parent.parent / "evals" / "lab1_2025"

needs_solver = pytest.mark.skipif(not have_solver(), reason="нет Z3")


def variant(number: int):
    return parse_srs((LAB1 / f"variant-{number:02d}.srs").read_text(encoding="utf-8"))


def scalars(**values) -> MatrixInterpretation:
    """Одномерная интерпретация: буква ↦ число, слово ↦ **произведение**.

    Именно произведение, а не сумма: полукольцо здесь `(ℕ, +, ·)`,
    и матрица 1×1 перемножается как число. Отсюда и требование «угол
    не меньше единицы» — нулевой множитель обнулил бы всё.
    """
    return MatrixInterpretation(
        {letter: Matrix(((value,),)) for letter, value in values.items()}
    )


SHIFTED = MatrixInterpretation(
    {
        "a": Matrix(((1, 1), (0, 1))),
        "b": Matrix(((1, 0), (0, 2))),
    }
)


# --------------------------------------------------------------------------
# Цепочка и арбитр
# --------------------------------------------------------------------------


def test_a_two_step_chain_is_accepted():
    """Первый шаг убирает `aa → a` числами, второй — `ab → ba` матрицей.

    Одномерная интерпретация на `ab → ba` даёт равенство при любых
    числах: перестановка букв произведение не меняет. Поэтому строгим
    первый шаг на этом правиле не станет никогда, и нужен второй.
    """
    system = parse_srs("aa -> a\nab -> ba")
    first, second = system.rules
    proof = RemovalProof(
        (
            RemovalStep("матричная", scalars(a=2, b=1), (first,)),
            RemovalStep("матричная", SHIFTED, (second,)),
        )
    )
    verdict = proof.check(system)
    assert verdict.value is True
    assert "за 2 шаг" in verdict.reason


def test_the_arbiter_refuses_a_step_that_removes_a_rule_that_does_not_fall():
    """`ab → ba` под весами не убывает строго — удалять его нельзя."""
    system = parse_srs("aa -> a\nab -> ba")
    proof = RemovalProof(
        (RemovalStep("матричная", scalars(a=2, b=1), tuple(system.rules)),)
    )
    verdict = proof.check(system)
    assert verdict.value is None
    assert "строго не убывает" in verdict.reason


def test_the_arbiter_refuses_a_step_where_a_rule_grows():
    """Нестрогое убывание требуется от **всех** правил, а не только от снятых."""
    system = parse_srs("aa -> a\na -> bb")
    proof = RemovalProof(
        (RemovalStep("матричная", scalars(a=1, b=2), (system.rules[0],)),)
    )
    verdict = proof.check(system)
    assert verdict.value is None
    assert "возрастает" in verdict.reason


def test_the_arbiter_refuses_a_chain_that_leaves_rules_behind():
    system = parse_srs("aa -> a\nab -> ba")
    proof = RemovalProof(
        (RemovalStep("матричная", scalars(a=2, b=1), (system.rules[0],)),)
    )
    verdict = proof.check(system)
    assert verdict.value is None
    assert "осталось 1 правил" in verdict.reason


def test_the_arbiter_refuses_a_non_monotone_interpretation():
    """Нулевой угол — и убывание перестаёт переноситься в контекст."""
    system = parse_srs("aa -> a")
    broken = MatrixInterpretation({"a": Matrix(((0, 2), (0, 0)))})
    verdict = RemovalProof(
        (RemovalStep("матричная", broken, tuple(system.rules)),)
    ).check(system)
    assert verdict.value is None
    assert "не монотонна" in verdict.reason


def test_an_empty_chain_proves_nothing():
    assert RemovalProof(()).check(parse_srs("aa -> a")).value is None


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------


@needs_solver
def test_removal_proves_what_a_single_interpretation_proves():
    """Прямое доказательство — частный случай цепочки из одного шага."""
    for text in ("aa -> a", "a -> bb", "ab -> ba"):
        assert prove_by_removal(parse_srs(text), (1, 2), 3, 10_000).value is True


@needs_solver
@pytest.mark.parametrize("number", [1, 2, 13, 20, 28])
def test_a_variant_with_a_known_loop_is_never_proved(number):
    system = variant(number)
    assert system.terminates().value is False
    assert prove_by_removal(system, (1, 2), 2, 5_000).value is not True


@needs_solver
def test_the_result_of_the_search_is_verified_by_the_arbiter():
    """Наружу выдаётся только то, что арбитр пересчитал сам."""
    system = parse_srs("aa -> a\nab -> ba")
    verdict = prove_by_removal(system, (1, 2), 3, 10_000)
    assert verdict.value is True
    assert isinstance(verdict.witness, RemovalProof)
    assert verdict.witness.check(system).value is True


@needs_solver
def test_removal_takes_a_system_no_single_interpretation_of_the_same_size_takes():
    """Замер, ради которого метод и сделан, — и его честная граница.

    У системы `aa → bb`, `ab → a`, `aba → aaa` нет **ни одной** прямой
    интерпретации размерности ⩽ 2: ни матричной, ни арктической,
    и решатель отвечает `unsat`, а не «не уложился». Удаление правил
    закрывает её теми же размерностями ⩽ 2 за секунды, цепочкой
    из трёх шагов, среди которых есть арктический.

    Граница: на размерности 3 прямая матричная интерпретация у неё
    всё-таки есть, решатель находит её примерно за сорок секунд.
    Значит выигрыш здесь в размерности и цене, а не в принципиальной
    силе метода. Система найдена перебором случайных систем.
    """
    from tfl.arctic import find_arctic_interpretation
    from tfl.matrix import find_matrix_interpretation

    system = parse_srs("aa -> bb\nab -> a\naba -> aaa")
    assert system.terminates().value is None
    for dimension in (1, 2):
        direct = find_matrix_interpretation(system, dimension, 3, 10_000)
        assert direct.value is None and "не существует" in direct.reason
        arctic = find_arctic_interpretation(system, dimension, 3, 10_000)
        assert arctic.value is None and "не существует" in arctic.reason
    verdict = prove_by_removal(system, (1, 2), 3, 10_000)
    assert verdict.value is True
    assert len(verdict.witness.steps) >= 2
