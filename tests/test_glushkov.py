"""Тесты позиционного автомата и редукции НКА."""

from __future__ import annotations

import pytest

from tfl import regex as rx
from tfl.automata import dfa_of, disagreements, thompson
from tfl.glushkov import glushkov, linearize, reduce_nfa, small_nfa
from tfl.glushkov import is_one_unambiguous, nondeterminism
from tfl.automata import dfa_of, equivalent
from tfl.myhill import extended_fooling_set

from tests.test_core import VARIANTS


def test_linearization_numbers_every_occurrence():
    lin = linearize(rx.parse("ab*a"))
    assert lin.size == 3
    assert sorted(lin.symbol.values()) == ["a", "a", "b"]


def test_glushkov_has_n_plus_one_states():
    node = rx.parse("ab*a")
    assert len(glushkov(node).states) == linearize(node).size + 1


def test_glushkov_has_no_epsilon_transitions():
    assert not glushkov(rx.parse("(a|b)*abb")).eps


def test_glushkov_nullable_start_is_final():
    assert glushkov(rx.parse("a*")).accepts("")
    assert not glushkov(rx.parse("aa*")).accepts("")


def test_follow_through_nullable_factors():
    """Транзитивность follow сквозь аннулируемые сомножители.

    В `a b* c` позиция `a` должна вести и в `b`, и — минуя его — в `c`.
    Это тот случай, на котором ломается наивная реализация конкатенации.
    """
    node = rx.parse("ab*c")
    nfa = glushkov(node)
    assert nfa.accepts("ac")
    assert nfa.accepts("abc")
    assert nfa.accepts("abbbc")
    assert not nfa.accepts("ab")


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_glushkov_matches_dfa(num, pattern):
    node = rx.parse(pattern)
    dfa = dfa_of(node)
    bad = disagreements(glushkov(node).accepts, dfa.accepts, node.alphabet(), max_len=5)
    assert not bad, f"вариант {num}: позиционный автомат расходится с ДКА на {bad}"


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_reduction_preserves_language(num, pattern):
    """Склейка состояний с равным правым языком не меняет язык автомата."""
    node = rx.parse(pattern)
    dfa = dfa_of(node)
    bad = disagreements(small_nfa(node).accepts, dfa.accepts, node.alphabet(), max_len=5)
    assert not bad, f"вариант {num}: редукция изменила язык на {bad}"


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_reduction_never_grows(num, pattern):
    node = rx.parse(pattern)
    assert len(small_nfa(node).states) <= len(glushkov(node).states)


def test_glushkov_much_smaller_than_thompson():
    node = rx.parse("b*((ab*a)*(aabb|(babb)*))*")
    assert len(thompson(node).states) == 46
    assert len(glushkov(node).states) == 13
    assert len(small_nfa(node).states) == 6


def test_reduced_nfa_is_minimal_for_variant_15():
    """Вариант 15: редукция даёт 6 состояний, нижняя оценка тоже 6.

    Совпадение оценки снизу с числом состояний доказывает минимальность НКА.
    У `dm800-TFLlabs/lab2` НКА тоже на 6 состояний.
    """
    node = rx.parse("b*((ab*a)*(aabb|(babb)*))*")
    nfa = small_nfa(node)
    bound = extended_fooling_set(dfa_of(node)).bound
    assert len(nfa.states) == 6
    assert bound == 6, "оценка снизу совпала с размером — НКА минимален"


def test_dm800_extended_regex_simplification_is_correct():
    """Проверка нетривиального шага из отчёта dm800.

    Они убирают внутреннюю итерацию под внешней:
    `((ab*a)*(aabb|(babb)*))*` → `(ab*a|aabb|babb)*`.
    Шаг верный, хотя выглядит рискованным: тело внешней итерации после
    упрощения обязано оканчиваться на `aabb`/`babb`, но потерянные слова
    восстанавливаются другими итерациями.
    """
    from tfl.automata import counterexample

    original = dfa_of("b*((ab*a)*(aabb|(babb)*))*")
    simplified = dfa_of("b*(ab*a|aabb|babb)*")
    assert counterexample(original, simplified) is None


# --------------------------------------------------------------------------
# 1-однозначность (РК2 2023, 2 балла; вопрос «Фармы»)
# --------------------------------------------------------------------------


def test_deterministic_glushkov_proves_one_unambiguity():
    """`a*b`: разные буквы в развилках, автомат Глушкова детерминирован."""
    verdict = is_one_unambiguous("a*b")
    assert verdict.value is True
    assert nondeterminism(rx.parse("a*b")) == []


def test_nondeterminism_points_at_positions():
    """Задача 4 РК1 просит указать позиции недетерминированного разбора."""
    conflicts = nondeterminism(rx.parse("(a|b)*a"))
    assert conflicts
    start = [c for c in conflicts if c.position == 0]
    assert start and start[0].char == "a"
    assert len(start[0].targets) == 2
    assert "выбор между" in str(start[0])


def test_verdict_is_about_the_regex_not_the_language():
    """Недетерминированность записи не переносится на язык — и наоборот.

    `(a|b)*a` недетерминирован по Глушкову, но тот же язык задаётся
    1-однозначной записью `(b*a)+`. Поэтому здесь «не выяснено», а не «нет».
    """
    loose = is_one_unambiguous("(a|b)*a")
    tight = is_one_unambiguous("(b*a)(b*a)*")
    assert loose.value is None
    assert tight.value is True
    assert equivalent(dfa_of("(a|b)*a"), dfa_of("(b*a)(b*a)*"))


def test_two_stars_over_one_letter_are_ambiguous():
    """`a*a*` — учебный пример неоднозначной записи."""
    assert is_one_unambiguous("a*a*").value is None
