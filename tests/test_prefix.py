"""Алфавитные префиксные грамматики — лекция 4 курса 2022.

Проверяется главное утверждение слайда 31: язык `L⟨S, w₀⟩` регулярен,
и конструкция, которая это доказывает, действительно даёт тот же язык,
что и прямое переписывание первой буквы.
"""

from __future__ import annotations

import pytest

from tfl.automata import DFA
from tfl.prefix import (
    GeneralPrefixGrammar,
    PrefixGrammar,
    check_labelled_rule,
    dfa_to_prefix_grammar,
    parse_prefix_grammar,
)


def test_left_side_must_be_a_single_letter():
    """В APG переписывается ровно один символ, и это не украшение."""
    with pytest.raises(ValueError, match="не буква"):
        PrefixGrammar((("ab", "c"),), start="ab")


def test_collapsing_is_a_fixpoint_not_a_single_step():
    """`a ↠ ε` через цепочку: `a → bc`, а `b` и `c` коллапсируют сами."""
    apg = parse_prefix_grammar("a -> bc\nb -> ε\nc -> ε", start="a")
    assert apg.collapsing() == frozenset("abc")

    isolated = parse_prefix_grammar("a -> bc\nb -> ε", start="a")
    assert isolated.collapsing() == frozenset("b")


def test_only_the_first_letter_is_rewritten():
    """`aa` с правилом `a → b` даёт `ba`, но не `ab` и не `bb` за шаг."""
    apg = parse_prefix_grammar("a -> b", start="aa")
    assert apg.step("aa") == {"ba"}
    assert apg.step("ba") == set()


@pytest.mark.parametrize(
    "rules,start",
    [
        ("a -> bc\nb -> ε\nc -> ε", "ab"),
        ("a -> ε\nb -> ε", "ab"),
        ("a -> ab", "a"),
        ("a -> ba\nb -> ε", "aa"),
        ("a -> bb\nb -> a\nb -> ε", "ab"),
        ("a -> b\nb -> c\nc -> ε", "abc"),
    ],
)
def test_construction_matches_direct_rewriting(rules, start):
    """Конструкция лекции и перебор достижимых слов написаны независимо."""
    verdict = parse_prefix_grammar(rules, start).agrees_with_rewriting(max_len=7)
    assert verdict.value is True, verdict.reason


def test_produced_grammar_is_linear_and_hence_regular():
    """Нетерминал в правой части один и стоит первым — язык регулярен.

    Это и есть механическая часть утверждения «язык `L⟨S, w₀⟩` регулярен»:
    построенная грамматика линейна по форме, а не по обещанию.
    """
    apg = parse_prefix_grammar("a -> bb\nb -> a\nb -> ε", start="ab")
    grammar = apg.to_cfg()
    for production in grammar.productions:
        inside = [s for s in production.rhs if s in grammar.nonterminals]
        assert len(inside) <= 1
        if inside:
            assert production.rhs[0] in grammar.nonterminals


def test_front_stops_at_a_non_collapsing_letter():
    """Если `b` не коллапсирует, фронт переписывания за него не уйдёт."""
    apg = parse_prefix_grammar("a -> ε\nc -> a", start="abc")
    assert apg.reachable(max_len=6) == {"abc", "bc"}
    assert apg.agrees_with_rewriting(max_len=6).value is True


def test_empty_word_needs_the_whole_start_to_collapse():
    """`S → ε` добавляется только когда коллапсируют все буквы `w₀`."""
    collapsing = parse_prefix_grammar("a -> ε\nb -> ε", start="ab")
    assert "" in collapsing.reachable(max_len=4)

    stuck = parse_prefix_grammar("a -> ε", start="ab")
    assert "" not in stuck.reachable(max_len=4)
    assert stuck.agrees_with_rewriting(max_len=6).value is True


# --------------------------------------------------------------------------
# Общая префиксная грамматика G = (W, R)
# --------------------------------------------------------------------------


def words_ending_in_a() -> DFA:
    return DFA(
        alphabet=frozenset("ab"),
        start=0,
        finals=frozenset({1}),
        delta={
            (0, "a"): 1,
            (0, "b"): 0,
            (1, "a"): 1,
            (1, "b"): 0,
        },
    )


def test_general_prefix_rewriting_replaces_an_arbitrary_prefix():
    grammar = GeneralPrefixGrammar(
        frozenset({"ab"}),
        (("a", "ba"), ("ab", "")),
    )
    assert grammar.step("ab") == {"", "bab"}
    assert grammar.step("cab") == set()


def test_right_cayley_graph_gives_a_prefix_grammar_for_the_dfa_language():
    machine = words_ending_in_a()
    grammar = dfa_to_prefix_grammar(machine)

    assert grammar.bases == frozenset({"a"})
    assert grammar.agrees_with_dfa(machine, max_len=7).value is True
    for left, right in grammar.rules:
        assert machine.run(left) == machine.run(right)


def test_labelled_lexical_rule_is_decided_by_residual_inclusion():
    parity = DFA(
        alphabet=frozenset({"a"}),
        start=0,
        finals=frozenset({0}),
        delta={(0, "a"): 1, (1, "a"): 0},
    )

    valid = check_labelled_rule(parity, "", "a", {0}, {1})
    assert valid.value is True

    invalid = check_labelled_rule(parity, "", "a", {0}, {0})
    assert invalid.value is False
    assert invalid.witness == ""
