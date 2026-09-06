"""Переключающиеся автоматы: обе конструкции преподавателя и теоремы лекции.

Эталоны:

* два примера из issue #40 (`corpus/issues/lab2-2025-issue40.md`) —
  сверяются с самими lookahead-регулярками через `re`;
* автомат из лекции 6 (`corpus/txt/chat_AFA.txt`) — сверяется
  со словесной формулировкой оттуда же.
"""

from __future__ import annotations

import re

from tfl.afa import AFA, conjunction, disagreements, with_lookahead
from tfl.automata import counterexample, dfa_of, equivalent
from tfl.words import iter_words

AB = "ab"


def lecture_automaton() -> AFA:
    """`({q0,q1,q2}, {s}, {a,b}, s, δ, {s,q0})` из лекции 6.

    Состояние `q2` в лекции нарисовано как тупик для иллюстрации
    незавершённого прогона; на язык оно не влияет и здесь опущено.
    """
    return AFA(
        alphabet=frozenset(AB),
        start="s",
        finals=frozenset({"s", "q0"}),
        universal=frozenset({"s"}),
        delta={
            ("s", "a"): frozenset({"s"}),
            ("s", "b"): frozenset({"s", "q0"}),
            ("q0", "a"): frozenset({"q1"}),
            ("q0", "b"): frozenset({"q0"}),
            ("q1", "a"): frozenset({"q0"}),
            ("q1", "b"): frozenset({"q1"}),
        },
    )


# --------------------------------------------------------------------------
# Примеры преподавателя
# --------------------------------------------------------------------------


def test_conjunction_matches_the_first_lookahead_example():
    """`^(?=.*a.*$).*b.*$` — конъюнкция двух ДКА в стартовой вершине."""
    automaton = conjunction(
        [dfa_of("(a|b)*a(a|b)*", AB), dfa_of("(a|b)*b(a|b)*", AB)], AB
    )
    pattern = re.compile(r"^(?=.*a.*$).*b.*$")
    assert disagreements(automaton, lambda w: bool(pattern.fullmatch(w)), 9) == []
    assert len(automaton.universal) == 1


def test_conjunction_is_the_intersection():
    parts = [dfa_of("(a|b)*a(a|b)*", AB), dfa_of("(a|b)*b(a|b)*", AB)]
    automaton = conjunction(parts, AB)
    assert disagreements(
        automaton, lambda w: all(part.accepts(w) for part in parts), 9
    ) == []


def test_lookahead_matches_the_second_example():
    """`^((?=(b*ab*ab*)*$)a*b)*$` — рекурсивный инвариант.

    По `b` читатель возвращается в стартовую вершину, и там снова
    запускается проверка остатка — ровно как описано в issue #40.
    """
    automaton = with_lookahead(dfa_of("(a*b)*", AB), dfa_of("(b*ab*ab*)*", AB))
    pattern = re.compile(r"^((?=(b*ab*ab*)*$)a*b)*$")
    assert disagreements(automaton, lambda w: bool(pattern.fullmatch(w)), 11) == []


def test_lookahead_automaton_is_not_smaller_than_the_dfa():
    """Преподаватель это разрешает явно: «даже если формально
    он будет не меньше ДКА»."""
    automaton = with_lookahead(dfa_of("(a*b)*", AB), dfa_of("(b*ab*ab*)*", AB))
    minimal = automaton.to_dfa().minimize()
    assert len(automaton) == 6
    assert len(minimal) == 4


# --------------------------------------------------------------------------
# Автомат из лекции
# --------------------------------------------------------------------------


def test_lecture_automaton_language():
    """«Все слова, в которых число a справа от каждой b чётно»."""
    automaton = lecture_automaton()

    def claim(word: str) -> bool:
        return all(
            word[index + 1 :].count("a") % 2 == 0
            for index, char in enumerate(word)
            if char == "b"
        )

    assert disagreements(automaton, claim, 10) == []


def test_run_is_a_witness():
    """Прогон — дерево; все листья принимающие, ветвление только в `Q∀`."""
    automaton = lecture_automaton()
    run = automaton.run("abbaabaa")
    assert run is not None
    assert set(run.leaves()) <= automaton.finals
    assert automaton.run("abba") is None  # справа от первой b три a
    assert automaton.accepts("abba") is False


def test_branching_happens_only_in_universal_states():
    automaton = lecture_automaton()
    run = automaton.run("abbaabaa")
    stack = [run]
    while stack:
        node = stack.pop()
        if len(node.children) > 1:
            assert automaton.is_universal(node.state)
        stack.extend(node.children)


# --------------------------------------------------------------------------
# Теоремы 5 и 6
# --------------------------------------------------------------------------


def test_theorem_5_dual_accepts_the_complement():
    """Дополнение — это обмен `Q∃` и `Q∀` плюс дополнение `F`."""
    for automaton in (
        lecture_automaton(),
        conjunction([dfa_of("(a|b)*a(a|b)*", AB), dfa_of("(a|b)*b(a|b)*", AB)], AB),
        with_lookahead(dfa_of("(a*b)*", AB), dfa_of("(b*ab*ab*)*", AB)),
    ):
        dual = automaton.dual()
        assert all(
            dual.accepts(word) is not automaton.accepts(word)
            for word in iter_words(AB, 8)
        )


def test_theorem_6_nfa_has_the_same_language():
    for automaton in (
        lecture_automaton(),
        with_lookahead(dfa_of("(a*b)*", AB), dfa_of("(b*ab*ab*)*", AB)),
    ):
        nfa = automaton.to_nfa()
        assert all(
            nfa.accepts(word) == automaton.accepts(word) for word in iter_words(AB, 9)
        )


def test_alternating_automaton_defines_a_regular_language():
    """Итог лекции: ПКА не выходит за пределы регулярных языков.

    Сверка независимая: слева ПКА через теорему 6 и детерминизацию,
    справа академическая регулярка через Томпсона. Совпадение
    минимальных ДКА — это результат, а не тавтология.
    """
    automaton = with_lookahead(dfa_of("(a*b)*", AB), dfa_of("(b*ab*ab*)*", AB))
    plain = dfa_of("((aa)*b)*(aa)(aa)*b|", AB)
    assert equivalent(automaton.to_dfa().minimize(), plain.minimize())
    assert counterexample(automaton.to_dfa().minimize(), plain.minimize()) is None


# --------------------------------------------------------------------------
# Краевой случай, который в лекции противоречив
# --------------------------------------------------------------------------


def dead_universal() -> AFA:
    """Одно состояние: `Q∀`, принимающее, без единого перехода."""
    return AFA(
        alphabet=frozenset(AB),
        start="p",
        finals=frozenset({"p"}),
        universal=frozenset({"p"}),
    )


def test_empty_conjunction_is_true():
    """Заглохшая ветвь из `Q∀` исчезает, а не губит прогон."""
    automaton = dead_universal()
    assert all(automaton.accepts(word) for word in iter_words(AB, 5))


def test_empty_disjunction_is_false():
    automaton = dead_universal()
    automaton.universal = frozenset()
    assert automaton.accepts("") is True  # p принимающее
    assert not any(automaton.accepts(word) for word in iter_words(AB, 5) if word)


def test_this_reading_is_what_makes_theorem_5_work():
    """Другое прочтение ломает дуальность — поэтому выбрано это.

    Если считать пустую конъюнкцию ложью, то `L(A) = {ε}`, а дуальный
    автомат (пустая дизъюнкция тоже ложь, `F` пусто) не принимает ничего.
    Дополнением к `{ε}` пустой язык не является.
    """
    automaton = dead_universal()
    dual = automaton.dual()
    assert all(
        dual.accepts(word) is not automaton.accepts(word)
        for word in iter_words(AB, 5)
    )
    assert not any(dual.accepts(word) for word in iter_words(AB, 5))


def test_theorem_6_needs_the_empty_set_to_be_accepting():
    """В лекции напечатано `2^F \\ {∅}`; с этим НКА разошёлся бы с ПКА.

    Из `{p}` по любой букве единственный преемник — `∅`: обязательств
    не осталось. Исключить `∅` из принимающих значит объявить, что
    выполненный набор обязательств не выполнен.
    """
    automaton = dead_universal()
    nfa = automaton.to_nfa()
    assert frozenset() in nfa.finals
    assert all(nfa.accepts(word) == automaton.accepts(word) for word in iter_words(AB, 5))
