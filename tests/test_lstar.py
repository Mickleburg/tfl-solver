"""Активное обучение `L*` и лабиринты — ЛР2 2024.

Приёмочные точки взяты из условия: обе стратегии обработки контрпримера
и оба характера МАТа, названные в issue #31, плюс формат таблицы
для запроса об эквивалентности со слайда 7.
"""

from __future__ import annotations

import pytest

from tfl.automata import dfa_of, equivalent
from tfl.lstar import (
    KIND_PUMPED,
    KIND_SHORTEST,
    PREFIXES,
    SUFFIXES,
    DFATeacher,
    ObservationTable,
    learn,
)
from tfl.maze import PlanarMaze, random_planar_maze

TARGETS = ["(a|b)*abb", "(aa)*", "a*b*", "(a|b)*a(a|b)(a|b)", "(aaa)*b", "b*(ab*ab*)*"]


# --------------------------------------------------------------------------
# Таблица наблюдений
# --------------------------------------------------------------------------


def table_for(pattern: str, alphabet: str = "ab") -> ObservationTable:
    return ObservationTable(alphabet, DFATeacher(target=dfa_of(pattern, alphabet)))


def test_closedness_is_the_condition_of_the_slide():
    """> отсутствие в `S.Σ × E` строк, которые отличаются от строк в `S × E`"""
    table = table_for("(a|b)*abb")
    assert table.unclosed() is None or table.unclosed() in table.extended()
    table.close()
    assert table.unclosed() is None


def test_consistency_adds_a_column_of_the_form_gamma_v():
    """> Иначе дополняем `E` столбцом `γvₖ`.

    Столбец обязан быть буквой, приписанной к уже имеющемуся суффиксу.
    """
    table = table_for("(a|b)*abb")
    table.close()
    column = table.inconsistency()
    if column is not None:
        assert column[0] in table.alphabet
        assert column[1:] in table.suffixes


def test_the_table_is_printed_in_the_wire_format():
    """> Здесь `epsilon` — указание на пустую строку… `valk` — 0 или 1."""
    table = table_for("(a|b)*abb")
    table.close()
    printed = table.markdown()
    assert "epsilon" in printed
    assert "[S]" in printed and "[S·Σ]" in printed
    assert all(part in "01| \n" for part in printed.split("\n")[-1].split()[1:])


def test_extended_part_excludes_what_is_already_in_s():
    table = table_for("(a|b)*abb")
    table.close()
    assert not set(table.extended()) & set(table.prefixes)


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", TARGETS)
@pytest.mark.parametrize("strategy", [SUFFIXES, PREFIXES])
def test_both_strategies_learn_the_language_exactly(pattern, strategy):
    """Обе стратегии из issue #31 обязаны сходиться к тому же языку."""
    target = dfa_of(pattern, "ab").minimize()
    result = learn(DFATeacher(target=target), "ab", strategy)
    assert result.converged
    assert equivalent(result.dfa.minimize(), target), pattern


@pytest.mark.parametrize("kind", [KIND_SHORTEST, KIND_PUMPED])
def test_both_teachers_lead_to_the_same_automaton(kind):
    """> один благосклонный… второй — издевательский, выдающий контрпример
    > с несколькими накачками
    """
    target = dfa_of("(a|b)*abb", "ab").minimize()
    result = learn(DFATeacher(target=target, kind=kind), "ab")
    assert equivalent(result.dfa.minimize(), target)


def test_the_nasty_teacher_really_gives_a_longer_counterexample():
    target = dfa_of("(a|b)*abb", "ab").minimize()
    kind = {}
    for style in (KIND_SHORTEST, KIND_PUMPED):
        result = learn(DFATeacher(target=target, kind=style), "ab")
        kind[style] = result.counterexamples[0]
    assert len(kind[KIND_PUMPED]) > len(kind[KIND_SHORTEST])


def test_a_long_counterexample_costs_queries_but_can_save_rounds():
    """Измерено, а не предположено: длинный контрпример несёт больше сведений.

    На `(aaa)*b` благосклонный МАТ уводит в три раунда, издевательский —
    в два, но запросов о принадлежности тратится вдвое больше.
    """
    target = dfa_of("(aaa)*b", "ab").minimize()
    gentle = learn(DFATeacher(target=target, kind=KIND_SHORTEST), "ab")
    nasty = learn(DFATeacher(target=target, kind=KIND_PUMPED), "ab")
    assert nasty.rounds < gentle.rounds
    assert nasty.membership_queries > gentle.membership_queries


def test_queries_are_counted_honestly():
    """Учитель считает свои обращения, таблица — свой кеш. Числа разные."""
    target = dfa_of("(a|b)*abb", "ab").minimize()
    teacher = DFATeacher(target=target)
    result = learn(teacher, "ab")
    assert result.membership_queries == teacher.membership_queries
    assert result.equivalence_queries == teacher.equivalence_queries
    assert result.distinct_words <= result.membership_queries
    assert "запросов о принадлежности" in result.summary()


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="стратегия"):
        learn(DFATeacher(target=dfa_of("a*", "a")), "a", "какая-нибудь")


def test_the_round_limit_is_a_fuse_not_a_verdict():
    """Лимит раундов — предохранитель; выученность отмечается отдельным полем."""
    target = dfa_of("(a|b)*a(a|b)(a|b)(a|b)", "ab").minimize()
    stunted = learn(DFATeacher(target=target), "ab", max_rounds=1)
    assert not stunted.converged
    assert "НЕ выучен" in stunted.summary()


# --------------------------------------------------------------------------
# Планарный лабиринт
# --------------------------------------------------------------------------


def test_extra_instructions_after_the_exit_break_the_path():
    """> Если путь… содержит дополнительные инструкции после попадания
    > в финальное состояние, считается, что он языку не принадлежит.
    """
    maze = PlanarMaze({0: (1, 0)}, frozenset({1}))
    assert maze.accepts("L")
    assert not maze.accepts("LL")
    assert not maze.accepts("")  # старт — не выход
    assert not maze.accepts("R")


def test_a_maze_becomes_a_dfa_directly():
    maze = PlanarMaze({0: (1, 0)}, frozenset({1}))
    automaton = maze.to_dfa()
    for path in ("", "L", "R", "RL", "LL", "RRL"):
        assert automaton.accepts(path) == maze.accepts(path), path


def test_a_broken_description_is_rejected():
    with pytest.raises(ValueError, match="неописанные"):
        PlanarMaze({0: (1, 5)}, frozenset({1}))
    with pytest.raises(ValueError, match="и развилки, и выходы"):
        PlanarMaze({0: (0, 0)}, frozenset({0}))


def test_the_generator_stays_within_the_bounds_and_is_connected():
    """> MAT генерирует граф, используя данные числа как оценку сверху."""
    for seed in range(6):
        maze = random_planar_maze(8, 3, seed=seed)
        assert len(maze.forks) <= 8
        assert len(maze.exits) <= 3
        reachable = maze.to_dfa().minimize()
        assert len(reachable) >= 2


def test_lstar_learns_a_generated_maze():
    """Связка целиком: МАТ генерирует лабиринт, угадыватель его выучивает."""
    for seed in range(6):
        target = random_planar_maze(8, 3, seed=seed).to_dfa().minimize()
        result = learn(DFATeacher(target=target), "LR")
        assert result.converged, seed
        assert equivalent(result.dfa.minimize(), target), seed


def test_neither_strategy_dominates_and_this_is_measured():
    """На лабиринтах выгоднее суффиксы, на регулярках выше — префиксы.

    Заранее это не выводится, поэтому в рецепте стоит «мерить», а не
    «использовать такую-то».
    """
    maze_wins = 0
    for seed in range(12):
        target = random_planar_maze(8, 3, seed=seed).to_dfa().minimize()
        cost = {}
        for strategy in (SUFFIXES, PREFIXES):
            cost[strategy] = learn(DFATeacher(target=target), "LR", strategy).membership_queries
        maze_wins += cost[SUFFIXES] < cost[PREFIXES]
    assert maze_wins >= 8  # суффиксы выигрывают в большинстве лабиринтов

    target = dfa_of("(a|b)*abb", "ab").minimize()
    by_suffix = learn(DFATeacher(target=target), "ab", SUFFIXES).membership_queries
    by_prefix = learn(DFATeacher(target=target), "ab", PREFIXES).membership_queries
    assert by_prefix < by_suffix  # а здесь наоборот
