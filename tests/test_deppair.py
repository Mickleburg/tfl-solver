"""Пары зависимостей: построение, граф, редукционная пара, арбитр.

Главная проверка здесь — не «доказал», а **«не доказал лишнего»**.
Метод обязан молчать на системах, у которых петля найдена: доказать
завершимость незавершимой системы страшнее, чем не доказать завершимую.
Поэтому на каждом варианте ЛР1 2025 с известной петлёй стоит запрет,
и отдельно — случайные системы, где сверка идёт с поиском петли.
"""

from __future__ import annotations

import itertools
import pathlib
import random

import pytest

from tfl.deppair import (
    Affine,
    DependencyPair,
    DependencyProof,
    ReductionPair,
    Step,
    cap,
    cycles,
    defined_symbols,
    dependency_pairs,
    estimated_graph,
    have_solver,
    prove_termination,
)
from tfl.srs import parse_srs

LAB1 = pathlib.Path(__file__).parent.parent / "evals" / "lab1_2025"

needs_solver = pytest.mark.skipif(not have_solver(), reason="нет Z3")


def variant(number: int):
    return parse_srs((LAB1 / f"variant-{number:02d}.srs").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Пары
# --------------------------------------------------------------------------


def test_defined_symbols_are_the_first_letters_of_left_hand_sides():
    system = parse_srs("ab -> ba\ncc -> a")
    assert defined_symbols(system) == frozenset("ac")


def test_a_pair_is_made_for_every_defined_suffix():
    """`aaa → bab`: суффиксы `bab`, `ab`, `b` — все начинаются с определённой."""
    system = parse_srs("aaa -> bab\nbbb -> aaa")
    assert [str(pair) for pair in dependency_pairs(system)] == [
        "aaa♯ → bab♯",
        "aaa♯ → ab♯",
        "aaa♯ → b♯",
        "bbb♯ → aaa♯",
        "bbb♯ → aa♯",
        "bbb♯ → a♯",
    ]


def test_undefined_letters_start_no_pair():
    """`b` не начинает ни одной левой части, значит редекса из него нет."""
    system = parse_srs("ab -> ba")
    assert [str(pair) for pair in dependency_pairs(system)] == ["ab♯ → a♯"]


def test_an_empty_right_hand_side_gives_no_pairs():
    system = parse_srs("aa -> ε")
    assert dependency_pairs(system) == ()


def test_cap_cuts_at_the_first_defined_letter_after_the_root():
    defined = frozenset("a")
    assert cap("abba", defined) == "abb"
    assert cap("abb", defined) == "abb"
    assert cap("a", defined) == "a"


def test_the_graph_joins_pairs_by_a_common_prefix():
    system = parse_srs("aaa -> bab\nbbb -> aaa")
    pairs = dependency_pairs(system)
    graph = estimated_graph(pairs, defined_symbols(system))
    # `aaa♯ → b♯`: обрезки нет, `b` — префикс `bbb`, значит ребро есть;
    # с `aaa` первые буквы разные, ребра нет.
    targets = {str(pairs[index]) for index in graph[2]}
    assert targets == {"bbb♯ → aaa♯", "bbb♯ → aa♯", "bbb♯ → a♯"}


def test_cycles_finds_self_loops_and_ignores_acyclic_graphs():
    assert cycles({0: [0]}) == [(0,)]
    assert cycles({0: [1], 1: []}) == []
    assert cycles({0: [1], 1: [0], 2: []}) == [(0, 1)]


# --------------------------------------------------------------------------
# Арифметика редукционной пары
# --------------------------------------------------------------------------


def test_affine_composition_matches_hand_computation():
    """`f(x) = 2x + 1`, `g(x) = 3x`, `f∘g = 6x + 1`."""
    f = Affine(((2,),), (1,))
    g = Affine(((3,),), (0,))
    assert f.then(g) == Affine(((6,),), (1,))
    assert g.then(f) == Affine(((6,),), (3,))


def test_a_word_is_the_composition_of_its_letters():
    pair = ReductionPair(
        {"a": Affine(((2,),), (1,)), "b": Affine(((1,),), (3,))},
        {"a": Affine(((1,),), (0,)), "b": Affine(((1,),), (0,))},
        1,
    )
    # [ab](x) = [a]([b](x)) = 2(x+3) + 1 = 2x + 7
    assert pair.value("ab") == Affine(((2,),), (7,))
    assert pair.value("") == Affine.identity(1)


def test_strictness_lives_in_the_free_term_only():
    bigger = Affine(((2,),), (5,))
    smaller = Affine(((2,),), (4,))
    assert bigger.strictly_above(smaller)
    assert not smaller.strictly_above(bigger)
    # наклон меньше — сравнения нет вовсе, при больших `x` знак меняется
    assert not Affine(((1,),), (9,)).above(Affine(((2,),), (0,)))


# --------------------------------------------------------------------------
# Арбитр
# --------------------------------------------------------------------------


def forged(system, component, removed, plain, marked) -> DependencyProof:
    pairs = dependency_pairs(system)
    pair = ReductionPair(plain, marked, 1)
    return DependencyProof(pairs, (Step(component, removed, pair),))


def test_the_arbiter_refuses_a_proof_whose_rules_do_not_decrease():
    system = parse_srs("ab -> ba")
    proof = forged(
        system,
        (0,),
        (0,),
        # [ab] = 2x + 1, [ba] = 2x + 2 — правило растёт, а не убывает
        {"a": Affine(((1,),), (1,)), "b": Affine(((2,),), (0,))},
        {"a": Affine(((1,),), (7,)), "b": Affine(((1,),), (0,))},
    )
    verdict = proof.check(system)
    assert verdict.value is None
    assert "не убывает нестрого" in verdict.reason


def test_the_arbiter_refuses_a_step_that_removes_nothing():
    system = parse_srs("ab -> ba")
    unit = {letter: Affine(((1,),), (0,)) for letter in "ab"}
    proof = forged(system, (0,), (), unit, unit)
    verdict = proof.check(system)
    assert verdict.value is None
    assert "не выкидывает" in verdict.reason


def test_the_arbiter_refuses_a_proof_that_skips_a_component():
    """Шаг про две пары есть, а компонента в графе целиком — доказательства нет."""
    system = parse_srs("aaa -> bab\nbbb -> aaa")
    unit = {letter: Affine(((1,),), (0,)) for letter in "ab"}
    proof = forged(system, (0, 1), (0,), unit, unit)
    verdict = proof.check(system)
    assert verdict.value is None
    assert "нет шага для компоненты" in verdict.reason


def test_the_arbiter_refuses_pairs_that_do_not_belong_to_the_system():
    system = parse_srs("ab -> ba")
    stranger = DependencyPair("zz", "z", "выдумка")
    unit = {letter: Affine(((1,),), (0,)) for letter in "ab"}
    proof = DependencyProof((stranger,), (Step((0,), (0,), ReductionPair(unit, unit, 1)),))
    verdict = proof.check(system)
    assert verdict.value is None
    assert "не совпадают" in verdict.reason


def test_a_system_without_pairs_is_terminating_outright():
    """`aa → ε` не порождает редексов: цепочек нет, и доказывать нечего."""
    verdict = prove_termination(parse_srs("aa -> ε"))
    assert verdict.value is True
    assert "пар зависимостей нет" in verdict.reason


# --------------------------------------------------------------------------
# Поиск
# --------------------------------------------------------------------------


@needs_solver
def test_the_returned_proof_passes_the_arbiter_independently():
    system = parse_srs("ab -> ba")
    verdict = prove_termination(system)
    assert verdict.value is True
    assert verdict.witness.check(system).value is True
    assert "♯" in verdict.witness.markdown()


@needs_solver
def test_dependency_pairs_take_a_variant_that_nothing_else_took():
    """Вариант 8 ЛР1 2025: ни армейского порядка, ни линейной интерпретации."""
    system = variant(8)
    assert system.terminates().value is None
    verdict = system.prove_by_dependency_pairs()
    assert verdict.value is True
    assert verdict.witness.check(system).value is True


@needs_solver
@pytest.mark.parametrize("number", [1, 2, 13, 20, 28])
def test_a_system_with_a_loop_is_never_declared_terminating(number):
    """Главный запрет: доказать завершимость незавершимой системы нельзя."""
    system = variant(number)
    assert system.terminates().value is False
    assert system.prove_by_dependency_pairs(1, 3, 5_000).value is not True


@needs_solver
def test_random_systems_agree_with_the_search_for_a_loop():
    """Сверка со вторым оракулом: нашлась петля — значит доказательства нет."""
    rng = random.Random(20260908)
    words = [
        "".join(letters)
        for length in (1, 2, 3)
        for letters in itertools.product("ab", repeat=length)
    ]
    checked = 0
    for _ in range(24):
        rules = "\n".join(
            f"{rng.choice(words)} -> {rng.choice(words)}" for _ in range(2)
        )
        system = parse_srs(rules)
        verdict = system.prove_by_dependency_pairs(1, 2, 3_000)
        if verdict.value is not True:
            continue
        checked += 1
        assert verdict.witness.check(system).value is True, rules
        assert system.find_loop(max_len=8, context=2) is None, rules
    assert checked >= 5, "случайных систем с доказательством оказалось слишком мало"


@needs_solver
def test_a_wider_dimension_takes_what_the_linear_one_does_not():
    """Вариант 4: линейной пары нет, матричная размерности 2 находится."""
    system = variant(4)
    assert system.prove_by_dependency_pairs(1, 4, 10_000).value is None
    verdict = system.prove_by_dependency_pairs(2, 3, 30_000)
    assert verdict.value is True
    assert verdict.witness.check(system).value is True
    assert verdict.witness.steps[0].pair.dimension == 2
