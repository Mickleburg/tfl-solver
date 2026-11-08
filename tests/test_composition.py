import itertools
import json

import pytest

from tfl.automata import NFA, DFA, counterexample, dfa_of
from tfl.composition import (BudgetExceeded, Machine, combine, compare, compile_document,
                             determinize, load_json, operation_count, parse_machine)
from tfl.nlstar import canonical_rfsa, compact_residual_nfa
from tfl.words import iter_words


def portable(machine):
    if isinstance(machine, DFA):
        machine = NFA(machine.alphabet, machine.start, machine.finals,
                      {k: frozenset({v}) for k, v in machine.delta.items()})
    states = sorted(machine.states, key=repr)
    index = {q: i for i, q in enumerate(states)}
    edges = {(index[q], c): frozenset(index[r] for r in targets)
             for (q, c), targets in machine.delta.items()}
    edges.update({(index[q], None): frozenset(index[r] for r in targets)
                  for q, targets in machine.eps.items()})
    return Machine(tuple(sorted(machine.alphabet)), len(states), index[machine.start],
                   frozenset(index[q] for q in machine.finals), edges)


def test_composition_input_recovery_is_explicit_and_string_safe():
    raw = '{"s": "literal ,} and \\\"", "a": [1,2,],}'
    with pytest.raises(json.JSONDecodeError):
        load_json(raw)
    data, offsets = load_json(raw, allow_trailing_commas=True)
    assert len(offsets) == 2 and data["a"] == [1, 2] and ",}" in data["s"]
    assert load_json(json.dumps(data)) == (data, [])
    with pytest.raises(json.JSONDecodeError):
        load_json('{"a": broken,}', allow_trailing_commas=True)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_json('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        parse_machine([])
    no_trap = {"dfa": {"alphabet": ["a"], "start": 0, "finals": [0], "trap_state": None,
                       "transitions": [{"from": 0, "symbol": "a", "to": 0}]}}
    assert parse_machine(no_trap).accepts("aaa")


@pytest.mark.parametrize("bad", [
    {"kind": "dfa", "alphabet": ["a", "a"], "start": 0, "finals": [], "transitions": []},
    {"kind": "dfa", "alphabet": ["a"], "start": 0, "finals": [], "transitions": [
        {"from": 0, "symbol": None, "to": 0}]},
    {"kind": "nfa", "alphabet": ["a"], "states": [0], "start": 1, "finals": [], "transitions": []},
    {"kind": "dfa", "alphabet": ["a"], "start": 0, "finals": [], "transitions": [
        {"from": 0, "symbol": "a", "to": 0}, {"from": 0, "symbol": "a", "to": 1}]},
])
def test_invalid_automata_rejected(bad):
    with pytest.raises(ValueError):
        parse_machine(bad)


@pytest.mark.parametrize("op,first,second", [
    ("union", "a*", "b*"), ("concat", "a*", "b*"),
    ("concat", "ε|ab", "ε|ba"), ("union", "∅", "ε"),
    ("shuffle", "ab", "ba"), ("shuffle", "a*", "a*"),
])
def test_operations_against_independent_word_definitions(op, first, second):
    a, b = dfa_of(first, "ab"), dfa_of(second, "ab")
    machine = combine(op, [portable(a), portable(b)])
    for word in iter_words("ab", 7):
        if op == "union":
            expected = a.accepts(word) or b.accepts(word)
        elif op == "concat":
            expected = any(a.accepts(word[:i]) and b.accepts(word[i:]) for i in range(len(word)+1))
        else:
            expected = any(a.accepts("".join(c for c, bit in zip(word, mask) if bit)) and
                           b.accepts("".join(c for c, bit in zip(word, mask) if not bit))
                           for mask in itertools.product((0, 1), repeat=len(word)))
        assert machine.accepts(word) == expected


def test_star_and_shuffle_with_epsilon_arguments():
    a = combine("union", [portable(dfa_of("ε", "ab")), portable(dfa_of("ab", "ab"))])
    star = combine("star", [a])
    assert compare(star, portable(dfa_of("(ab)*", "ab")))["equivalent"]
    shuffled = combine("shuffle", [a, portable(dfa_of("ε|b", "ab"))])
    assert compare(shuffled, portable(dfa_of("ε|b|ab|abb|bab", "ab")))["equivalent"]


@pytest.mark.parametrize("pattern", ["a", "aa", "(ab)*", "a*b", "ε|aa|bbbb", "(a|b)*abb"])
def test_square_root_tracks_the_whole_word_transformation(pattern):
    base = portable(dfa_of(pattern, "ab"))
    root = combine("sqrt", [base])
    for word in iter_words("ab", 8):
        assert root.accepts(word) == base.accepts(word+word)
    if pattern == "a":
        assert not root.accepts("a")  # Not merely accepting a path of half a length.
    if pattern == "aa":
        assert root.accepts("a") and not root.accepts("aa")


def test_formula_counts_each_occurrence_and_round_trip_nfa():
    a = portable(dfa_of("a*", "ab"))
    assert compare(parse_machine(a.as_json()), a)["equivalent"]
    ref = {"ref": "A"}
    star = {"op": "star", "args": [ref]}
    tree = {"op": "concat", "args": [star, star]}
    assert operation_count(tree, {"A": a}) == 3
    document = {"schema_version": 1, "alphabet": ["a", "b"], "arguments": {"A": a.as_json()}, "formula": tree}
    with pytest.raises(ValueError):
        compile_document(document, max_operations=2)
    result, count = compile_document(document)
    assert count == 3 and compare(result, a)["equivalent"]
    for node in ({"ref": "B"}, {"op": "union", "args": [ref]},
                 {"op": "cascade", "args": [ref, ref]}):
        with pytest.raises(ValueError):
            operation_count(node, {"A": a})


def test_exact_comparison_counterexample_and_unknown_budgets():
    a, b = portable(dfa_of("a*", "ab")), portable(dfa_of("aa*", "ab"))
    assert compare(a, b)["counterexample"] == ""
    result = compare(portable(dfa_of("ab", "ab")), portable(dfa_of("aa", "ab")))
    assert not result["equivalent"] and result["counterexample"] == "aa"
    with pytest.raises(BudgetExceeded):
        compare(a, a, max_pairs=1)
    with pytest.raises(BudgetExceeded):
        determinize(a, max_states=1)
    with pytest.raises(BudgetExceeded):
        combine("sqrt", [a], max_states=1)
    with pytest.raises(BudgetExceeded):
        combine("union", [a, a], max_states=1)


@pytest.mark.parametrize("pattern", ["∅", "ε", "a*", "(a|b)*a(a|b)(a|b)", "(ab|ba)*", "a*b*a*"])
def test_fast_residual_basis_agrees_with_oracle(pattern):
    source = dfa_of(pattern, "ab")
    fast, canonical = compact_residual_nfa(source), canonical_rfsa(source)
    assert counterexample(source, fast.determinize()) is None
    assert len(fast.states) == len(canonical.states)
    with pytest.raises(ValueError):
        compact_residual_nfa(source, max_columns=0)


def test_residual_partial_table_is_not_a_certificate():
    with pytest.raises(RuntimeError, match="no residual certificate"):
        compact_residual_nfa(dfa_of("ab", "ab"), max_columns=1)


@pytest.mark.parametrize("pattern", ["a*|b*", "(a|b)*a(a|b)(a|b)", "a*b*a*", "∅", "ε"])
def test_pruning_covered_targets_preserves_all_residual_languages(pattern):
    source = dfa_of(pattern, "ab")
    sparse = compact_residual_nfa(source, prune_covered_transitions=True)
    dense = compact_residual_nfa(source)
    assert counterexample(source, sparse.determinize()) is None
    assert sum(map(len, sparse.delta.values())) <= sum(map(len, dense.delta.values()))
