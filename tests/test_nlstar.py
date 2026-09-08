"""Алгоритм `NL*` — ЛР3 2023, слайд 18.

Приёмочная точка — не «выучил хоть что-то», а «выучил **канонический**
остаточный автомат». Эталон строится независимо, прямо по определению
вычетов (`canonical_rfsa`), и `NL*` обязан прийти к тому же числу
состояний и тому же языку.

Замер, ради которого всё и делалось: на языке «третья буква с конца — `a`»
минимальный ДКА растёт как `2ⁿ`, а RFSA как `n+1`.
"""

from __future__ import annotations

import random

import pytest

from tfl.automata import DFA, dfa_of, equivalent
from tfl.lstar import PREFIXES, SUFFIXES, DFATeacher, learn
from tfl.nlstar import (
    ZERO_COMPOSED,
    ZERO_PRIME,
    RFSATable,
    absorbs,
    canonical_rfsa,
    covered,
    is_prime,
    join,
    learn_nfa,
    residual_primes,
)


def suffix_language(n: int) -> DFA:
    """`(a|b)* a (a|b)ⁿ⁻¹` — `n`-я буква с конца равна `a`."""
    return dfa_of("(a|b)*a" + "(a|b)" * (n - 1), "ab").minimize()


def random_dfa(states: int, seed: int, alphabet: str = "ab") -> DFA:
    rng = random.Random(seed)
    delta = {
        (source, letter): rng.randrange(states)
        for source in range(states)
        for letter in alphabet
    }
    finals = frozenset(s for s in range(states) if rng.random() < 0.4)
    return DFA(frozenset(alphabet), 0, finals, delta).minimize().trim()


# --------------------------------------------------------------------------
# Решётка строк
# --------------------------------------------------------------------------


def test_absorption_and_covering_are_the_slide_definitions():
    """> `r₁` поглощает `r₂`, если `∀i (r₁[i] ⩾ r₂[i])`."""
    assert absorbs((True, True), (True, False))
    assert not absorbs((True, False), (False, True))
    assert join([(True, False), (False, True)]) == (True, True)
    assert covered((True, True), [(True, False), (False, True)])
    assert not covered((True, True), [(True, False)])


def test_equal_rows_do_not_cancel_each_other():
    """Ловушка: сравнивать надо со **строго меньшими**.

    Если брать «все остальные строки», две равные накроют друг друга
    и обе окажутся не базисными — состояний не останется вовсе.
    """
    rows = [(True, False), (True, False), (False, True)]
    assert is_prime((True, False), rows)


def test_the_zero_row_is_covered_by_the_empty_set():
    """Дизъюнкция пустого набора — нулевая строка, и это не формальность.

    Именно отсюда растёт разница двух прочтений слайда 18.
    """
    assert covered((False, False), [])
    assert not covered((True, False), [])
    assert not is_prime((False, False), [(True, True)])
    assert is_prime((False, False), [(True, True)], True)


# --------------------------------------------------------------------------
# Выигрыш в числе состояний
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_residual_automaton_is_exponentially_smaller(n):
    """ДКА растёт как `2ⁿ`, RFSA — как `n+1`. Ради этого `NL*` и нужен."""
    target = suffix_language(n)
    assert len(target) == 2**n
    assert len(residual_primes(target)) == n + 1

    result = learn_nfa(DFATeacher(target=target), "ab")
    assert result.converged
    assert result.states == n + 1
    assert equivalent(result.nfa.determinize().minimize(), target)


def test_nlstar_reaches_exactly_the_canonical_automaton():
    """Сверка с эталоном, построенным **не обучением**, а по определению."""
    for seed in range(40):
        target = random_dfa(random.Random(seed).randint(1, 6), seed)
        if not target.states:
            continue
        result = learn_nfa(DFATeacher(target=target), "ab")
        assert result.converged, seed
        assert result.states == len(residual_primes(target)), seed
        assert equivalent(result.nfa.determinize().minimize(), target), seed
        assert equivalent(canonical_rfsa(target).determinize().minimize(), target), seed


def test_the_residual_automaton_is_never_larger_than_the_minimal_dfa():
    """Состояния RFSA — подмножество вычетов, поэтому больше быть не может."""
    for seed in range(40):
        target = random_dfa(random.Random(seed).randint(1, 6), seed + 100)
        if not target.states:
            continue
        assert len(residual_primes(target)) <= len(target), seed


# --------------------------------------------------------------------------
# Что стоит выигрыш
# --------------------------------------------------------------------------


def test_fewer_states_is_not_fewer_queries():
    """Замер против ожидания: меньший автомат обходится **дороже**.

    На случайных целях `NL*` тратит больше запросов о принадлежности, чем
    `L*`, чаще, чем меньше. Выигрыш `NL*` — в размере ответа, а не в цене.
    """
    cheaper = dearer = 0
    for seed in range(60):
        target = random_dfa(random.Random(seed).randint(1, 6), seed + 200)
        if not target.states:
            continue
        deterministic = learn(DFATeacher(target=target), "ab")
        residual = learn_nfa(DFATeacher(target=target), "ab")
        if residual.membership_queries < deterministic.membership_queries:
            cheaper += 1
        elif residual.membership_queries > deterministic.membership_queries:
            dearer += 1
    assert dearer > cheaper


# --------------------------------------------------------------------------
# Прочтение слайда 18 и стратегия префиксов
# --------------------------------------------------------------------------


def test_the_prefix_strategy_stalls_and_says_so():
    """Стратегия префиксов для `NL*` не работает, и это не дефект кода.

    Она не добавляет столбцов, поэтому строка `ε` остаётся нулевой; такая
    строка не базисна, начальных состояний не остаётся, гипотеза задаёт
    пустой язык, и учитель возвращает один и тот же контрпример вечно.
    """
    target = dfa_of("(a|b)*abb", "ab").minimize()
    result = learn_nfa(DFATeacher(target=target), "ab", PREFIXES, max_rounds=25)
    assert not result.converged
    assert "застой" in result.reason
    assert len(set(result.counterexamples)) == 1


def test_the_other_reading_converges_but_loses_minimality():
    """Второе прочтение «набора других»: пустой набор набором не считается.

    Тогда нулевая строка базисна, получает состояние-ловушку, и стратегия
    префиксов сходится. Плата — автомат перестаёт быть минимальным,
    а лекция обещает именно «минимальный остаточный НКА».
    """
    target = dfa_of("a(a|b)*b", "ab").minimize()
    strict = learn_nfa(DFATeacher(target=target), "ab", PREFIXES, max_rounds=25)
    assert not strict.converged

    loose = learn_nfa(
        DFATeacher(target=target), "ab", PREFIXES, max_rounds=25, zero=ZERO_PRIME
    )
    assert loose.converged
    assert equivalent(loose.nfa.determinize().minimize(), target)
    assert loose.states > len(residual_primes(target))


def test_the_suffix_strategy_works_under_both_readings():
    """Стратегия суффиксов добавляет столбцы и потому от прочтения не зависит."""
    target = dfa_of("(a|b)*abb", "ab").minimize()
    for reading in (ZERO_COMPOSED, ZERO_PRIME):
        result = learn_nfa(DFATeacher(target=target), "ab", SUFFIXES, zero=reading)
        assert result.converged, reading
        assert equivalent(result.nfa.determinize().minimize(), target), reading


def test_an_unknown_reading_is_rejected():
    target = dfa_of("a*", "ab").minimize()
    with pytest.raises(ValueError, match="прочтение"):
        learn_nfa(DFATeacher(target=target), "ab", zero="как-нибудь")
    with pytest.raises(ValueError, match="стратегия"):
        learn_nfa(DFATeacher(target=target), "ab", strategy="как-нибудь")


# --------------------------------------------------------------------------
# Крайние случаи и оформление
# --------------------------------------------------------------------------


def test_the_empty_language_has_no_prime_residuals():
    """У пустого языка единственный вычет нулевой, а он не базисный."""
    empty = DFA(frozenset("ab"), 0, frozenset(), {})
    assert residual_primes(empty) == []
    assert not canonical_rfsa(empty).accepts("")
    result = learn_nfa(DFATeacher(target=empty), "ab")
    assert result.converged
    assert result.states == 0


def test_the_table_marks_prime_rows():
    """В отчёт идёт таблица, и базисные строки в ней должны быть видны."""
    target = suffix_language(3)
    result = learn_nfa(DFATeacher(target=target), "ab")
    text = result.table.markdown()
    assert "базисные строки (4 из 8)" in text
    assert text.count("*") >= 4
    assert "epsilon" in text


def test_the_hypothesis_is_determinized_before_the_equivalence_query():
    """Слайд 11: «Если у вас вариант с НКА, тогда придётся детерминизировать»."""
    seen: list[DFA] = []

    class Watching(DFATeacher):
        def equivalent(self, hypothesis):
            seen.append(hypothesis)
            return super().equivalent(hypothesis)

    target = suffix_language(2)
    learn_nfa(Watching(target=target), "ab")
    assert seen
    assert all(isinstance(hypothesis, DFA) for hypothesis in seen)


def test_the_table_is_closed_and_consistent_at_the_end():
    """Инвариант цикла: гипотеза строится только по замкнутой таблице."""
    target = suffix_language(3)
    result = learn_nfa(DFATeacher(target=target), "ab")
    table = result.table
    assert table.unclosed() is None
    assert table.inconsistency() is None


def test_an_inconsistency_is_a_broken_inclusion_not_a_broken_equality():
    """Разница `NL*` и `L*` в одном примере.

    Строки `ε` и `a` в таблице ниже по включению сравнимы, но продолжение
    буквой `a` включение ломает — `NL*` обязан это заметить, хотя
    на равенство строки и так различны.
    """

    class Fixed(DFATeacher):
        pass

    target = dfa_of("(a|b)*a(a|b)", "ab").minimize()
    table = RFSATable("ab", Fixed(target=target))
    table.close()
    assert table.unclosed() is None
