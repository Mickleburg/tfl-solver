"""Древесные языки: термы, контексты, восходящие автоматы.

Контрольные точки — варианты РК1 2025 (`rk1_tfl_2025.pdf`), где древесные
языки стоят минимум в шести вариантах из четырнадцати.
"""

from __future__ import annotations

import pytest

from tfl.tree import (
    HOLE,
    Context,
    Tree,
    TreeAutomaton,
    at_least_classes,
    contexts,
    disagreements,
    parse_tree,
    trees,
)

SIGNATURE = {"P": 0, "Q": 0, "not": 1, "and": 2, "or": 2}


# --------------------------------------------------------------------------
# Термы и разбор
# --------------------------------------------------------------------------


def test_parse_and_print_round_trip():
    for text in ("P", "not(P)", "and(not(P), Q)", "or(and(P, Q), not(Q))"):
        assert str(parse_tree(text)) == text


def test_parse_rejects_malformed_input():
    for text in ("and(P", "and(P, Q))", "()", "and(,)"):
        with pytest.raises(ValueError):
            parse_tree(text)


def test_size_and_height():
    tree = parse_tree("and(not(P), Q)")
    assert tree.size() == 4
    assert tree.height() == 3
    assert len(list(tree.subtrees())) == 4


def test_enumeration_is_by_increasing_size():
    found = trees({"a": 0, "f": 1, "g": 2}, 4)
    sizes = [t.size() for t in found]
    assert sizes == sorted(sizes)
    assert str(found[0]) == "a"
    assert all(t.size() <= 4 for t in found)


def test_contexts_include_the_bare_hole():
    found = contexts({"a": 0, "f": 1}, 3)
    assert str(found[0]) == HOLE
    assert [str(c) for c in found] == [HOLE, f"f({HOLE})", f"f(f({HOLE}))"]


def test_filling_a_context_puts_the_tree_in_the_hole():
    context = Context(parse_tree(f"and({HOLE}, Q)"))
    assert str(context.fill(parse_tree("not(P)"))) == "and(not(P), Q)"


# --------------------------------------------------------------------------
# В11 2025: формулы без двойных отрицаний — язык регулярен
# --------------------------------------------------------------------------


def no_double_negation(tree: Tree) -> bool:
    return not any(
        node.symbol == "not" and node.children[0].symbol == "not"
        for node in tree.subtrees()
    )


def double_negation_automaton() -> TreeAutomaton:
    """Двух состояний хватает: «корень — отрицание» и «корень — не оно».

    Двойное отрицание ловится тем, что перехода `not` из состояния `neg`
    просто нет: дерево уходит в ловушку.
    """
    delta: dict = {(leaf, ()): "ok" for leaf in ("P", "Q")}
    delta[("not", ("ok",))] = "neg"
    for op in ("and", "or"):
        for left in ("ok", "neg"):
            for right in ("ok", "neg"):
                delta[(op, (left, right))] = "ok"
    return TreeAutomaton(SIGNATURE, frozenset({"ok", "neg"}), delta)


def test_double_negation_automaton_agrees_with_the_condition():
    automaton = double_negation_automaton()
    assert len(automaton) == 2
    assert disagreements(automaton, no_double_negation, max_size=7) == []


def test_the_trap_is_what_rejects_double_negation():
    automaton = double_negation_automaton()
    assert automaton.run(parse_tree("not(not(P))")) is None
    assert automaton.accepts(parse_tree("not(and(not(P), Q))"))


def test_disagreements_catch_a_wrong_automaton():
    """Если разрешить `not` поверх `neg`, автомат начнёт принимать лишнее."""
    automaton = double_negation_automaton()
    automaton.delta[("not", ("neg",))] = "neg"
    bad = disagreements(automaton, no_double_negation, max_size=6)
    assert bad and any(str(t) == "not(not(P))" for t in bad)


# --------------------------------------------------------------------------
# В4 2025: нет подформул вида Φ & Φ — язык не регулярен
# --------------------------------------------------------------------------


def no_idempotent_conjunction(tree: Tree) -> bool:
    return not any(
        node.symbol == "and" and node.children[0] == node.children[1]
        for node in tree.subtrees()
    )


def test_idempotent_conjunction_needs_unboundedly_many_classes():
    """Контекст `and(□, t)` отделяет `t` от всех остальных деревьев.

    Приём равномерный: для любого `n` берём `n` разных деревьев и `n`
    таких контекстов, и матрица принадлежности выходит с единственным
    минусом на диагонали. Значит классов бесконечно много, и древесный
    язык не регулярен.
    """
    samples = trees(SIGNATURE, 4)[:6]
    separators = [
        Context(parse_tree(f"and({HOLE}, {tree})")) for tree in samples
    ]
    verdict = at_least_classes(no_idempotent_conjunction, samples, separators)
    assert verdict.value is True
    assert "не меньше 6" in verdict.reason


def test_weak_separators_are_reported_honestly():
    """Контекстов не хватило — вердикт «не выяснено», а не «мало классов»."""
    samples = trees(SIGNATURE, 3)[:5]
    verdict = at_least_classes(no_idempotent_conjunction, samples, contexts(SIGNATURE, 1))
    assert verdict.value is None
    assert "не хватило" in verdict.reason


def test_empty_sample_is_refused():
    assert at_least_classes(no_idempotent_conjunction, [], contexts(SIGNATURE, 1)).value is False
