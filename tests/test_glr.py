"""ЛР5 2023: недетерминированный КС-разбор с гиперстеком.

Приёмочная точка — не «разобрал», а «разобрал **столько же** способов».
Число деревьев считается независимо, прямым перебором разбиений отрезка
по правилам, и должно совпасть с числом, вынутым из упакованного леса.

Контрольная точка на грамматике `S → SS | a` выбрана не случайно: число
разборов слова `aⁿ` — это число Каталана, оно известно заранее и растёт
экспоненциально, тогда как графовидный стек обязан остаться линейным.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.cfg import CFG, parse_cfg
from tfl.glr import (
    ACCEPT,
    GRAPH,
    LR0,
    REDUCE,
    SHIFT,
    SLR1,
    TREE,
    actions_table,
    parse,
    parse_ll,
)
from tfl.parse import recognize

CATALAN = [1, 1, 2, 5, 14, 42, 132]


def count_trees(grammar: CFG, word: str) -> int:
    """Независимый счёт деревьев вывода: перебор разбиений отрезка.

    Реализация нарочно прямая и медленная — она арбитр, а не парсер.
    Работает только для грамматик без ε-правил и без циклов `A ⇒⁺ A`,
    и обе оговорки в наших грамматиках выполнены.
    """
    memo: dict[tuple[str, int, int], int] = {}

    def splits(symbols, start, end):
        if not symbols:
            return 1 if start == end else 0
        if len(symbols) == 1:
            return count(symbols[0], start, end)
        total = 0
        for middle in range(start + 1, end):
            left = count(symbols[0], start, middle)
            if left:
                total += left * splits(symbols[1:], middle, end)
        return total

    def count(symbol, start, end):
        key = (symbol, start, end)
        if key in memo:
            return memo[key]
        if symbol not in grammar.nonterminals:
            memo[key] = 1 if end == start + 1 and word[start] == symbol else 0
            return memo[key]
        memo[key] = 0  # защита от циклов: развёрнутый вход даёт ноль
        memo[key] = sum(
            splits(production.rhs, start, end)
            for production in grammar.rules_for(symbol)
        )
        return memo[key]

    return count(grammar.start, 0, len(word))


# --------------------------------------------------------------------------
# Таблица действий
# --------------------------------------------------------------------------


def test_conflicts_are_kept_not_resolved():
    """Обычный парсер на конфликте встаёт, Generic — ветвится."""
    grammar = parse_cfg("S -> S S | a")
    assert not grammar.is_slr1()
    _, _, table = actions_table(grammar, SLR1)
    conflicting = [key for key, group in table.items() if len(group) > 1]
    assert conflicting
    kinds = {action.kind for key in conflicting for action in table[key]}
    assert kinds == {SHIFT, REDUCE}


def test_lr0_reduces_on_every_lookahead():
    """Разница таблиц: LR(0) сворачивает по любому символу, SLR(1) — по Follow."""
    grammar = parse_cfg("S -> a S b | a b")
    _, _, weak = actions_table(grammar, LR0)
    _, _, strong = actions_table(grammar, SLR1)
    assert sum(len(group) for group in weak.values()) > sum(
        len(group) for group in strong.values()
    )


def test_epsilon_rules_are_refused_loudly():
    grammar = parse_cfg("S -> a S | ε")
    with pytest.raises(ValueError, match="без ε-правил"):
        actions_table(grammar, SLR1)


def test_an_unknown_mode_is_refused():
    grammar = parse_cfg("S -> a")
    with pytest.raises(ValueError, match="режим"):
        actions_table(grammar, "LALR")
    with pytest.raises(ValueError, match="стек"):
        parse(grammar, "a", SLR1, "какой-нибудь")


# --------------------------------------------------------------------------
# Разбор против независимых арбитров
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rules,words",
    [
        ("S -> a S b | a b", ["ab", "aabb", "aaabbb", "abb", "ba", "a", "b", "aab"]),
        ("S -> S S | a", ["a", "aa", "aaa", "aaaa", "b", "ab"]),
        ("E -> E + E | E * E | a", ["a", "a+a", "a+a*a", "a+a*a+a", "+a", "a+"]),
        ("S -> a S a | b S b | c", ["c", "aca", "abcba", "ab", "acb"]),
    ],
)
def test_acceptance_agrees_with_earley(rules, words):
    grammar = parse_cfg(rules, nonterminals="E" if rules.startswith("E") else "")
    for word in words:
        result = parse(grammar, word, SLR1)
        assert result.accepted == recognize(grammar, word), (rules, word)


@pytest.mark.parametrize(
    "rules,words",
    [
        ("S -> S S | a", ["a", "aa", "aaa", "aaaa", "aaaaa"]),
        ("E -> E + E | E * E | a", ["a", "a+a", "a+a*a", "a+a*a+a", "a*a+a*a"]),
        ("S -> a S b | a b", ["ab", "aabb", "aaabbb"]),
        ("S -> A B | a S b\nA -> a A | a\nB -> b B | b", ["ab", "aabb", "aab", "abb"]),
    ],
)
def test_the_forest_counts_exactly_as_many_trees(rules, words):
    """Главная проверка: лес не теряет и не выдумывает разборы."""
    grammar = parse_cfg(rules, nonterminals="E" if rules.startswith("E") else "")
    for word in words:
        result = parse(grammar, word, SLR1)
        assert result.parses == count_trees(grammar, word), (rules, word)


def test_catalan_numbers_come_out():
    """`S → SS | a`: разборов слова `aⁿ` ровно `Cₙ₋₁`."""
    grammar = parse_cfg("S -> S S | a")
    for length in range(1, 7):
        result = parse(grammar, "a" * length, SLR1)
        assert result.parses == CATALAN[length - 1], length


def test_the_trees_themselves_are_distinct():
    grammar = parse_cfg("S -> S S | a")
    result = parse(grammar, "aaaa", SLR1)
    trees = result.forest.trees(result.root)
    assert len(trees) == 5
    assert len(set(trees)) == 5
    for tree in trees:
        assert tree[0] == "S"


def test_ambiguity_is_packed_not_duplicated():
    """Неоднозначность — это несколько семейств у **одного** узла леса."""
    grammar = parse_cfg("E -> E + E | E * E | a", nonterminals="E")
    result = parse(grammar, "a+a*a", SLR1)
    assert result.parses == 2
    packed = result.forest.ambiguous()
    assert packed == [("E", 0, 5)]
    assert len(result.forest.families[("E", 0, 5)]) == 2


# --------------------------------------------------------------------------
# Ради чего гиперстек
# --------------------------------------------------------------------------


def test_the_graph_stack_stays_linear_while_parses_explode():
    """Разборов экспоненциально много, вершин — линейно.

    В этом и весь смысл: слить вершины можно, а деревья — нет,
    и потому лес пакуется отдельно от стека.
    """
    grammar = parse_cfg("S -> S S | a")
    sizes = []
    for length in range(1, 7):
        result = parse(grammar, "a" * length, SLR1, GRAPH)
        sizes.append(result.nodes)
    growth = [second - first for first, second in zip(sizes, sizes[1:])]
    assert len(set(growth)) == 1, sizes  # прирост постоянный
    assert parse(grammar, "aaaaaa", SLR1, GRAPH).parses == 42


def test_the_tree_stack_pays_for_not_merging():
    """Соло-версия задания: та же грамматика, вершин вдвое больше на букву."""
    grammar = parse_cfg("S -> S S | a")
    shared = parse(grammar, "aaaaa", SLR1, GRAPH)
    solo = parse(grammar, "aaaaa", SLR1, TREE)
    assert shared.parses == solo.parses
    assert solo.nodes > 3 * shared.nodes


def test_in_the_tree_stack_every_node_has_one_edge_down():
    """Древовидный стек — это дерево: общая нижняя часть, несливаемые вершины."""
    grammar = parse_cfg("S -> S S | a")
    result = parse(grammar, "aaaa", SLR1, TREE)
    last = result.snapshots[-1]
    outgoing: dict[object, int] = {}
    for source, _, _ in last.edges:
        outgoing[source] = outgoing.get(source, 0) + 1
    assert max(outgoing.values()) == 1

    shared = parse(grammar, "aaaa", SLR1, GRAPH)
    last = shared.snapshots[-1]
    outgoing = {}
    for source, _, _ in last.edges:
        outgoing[source] = outgoing.get(source, 0) + 1
    assert max(outgoing.values()) > 1  # ровно то, чего в дереве нет


# --------------------------------------------------------------------------
# Что требует выдать условие
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,position",
    [("abb", 2), ("ba", 0), ("a", 1), ("aabbb", 4)],
)
def test_the_first_impossible_position_is_reported(word, position):
    """> …с указанием первой найденной ошибочной позиции (то есть такой,
    > в которой невозможен ни один из путей разбора).
    """
    grammar = parse_cfg("S -> a S b | a b")
    result = parse(grammar, word, SLR1)
    assert not result.accepted
    assert result.error_position == position, result.summary()


def test_a_correct_word_has_no_error_position():
    grammar = parse_cfg("S -> a S b | a b")
    result = parse(grammar, "aabb", SLR1)
    assert result.accepted
    assert result.error_position is None


def test_both_step_conventions_are_available():
    """Варианты считают шаги по-разному, и условие это оговаривает.

    Для SLR(1) шаг — действие (перенос или свёртка), для LR(0) —
    прочитанная буква. Снимок берётся по обоим счётчикам.
    """
    grammar = parse_cfg("S -> a S b | a b")
    result = parse(grammar, "aabb", SLR1)

    by_action = result.snapshot_by_action(3)
    assert by_action is not None
    assert by_action.actions == 3
    assert by_action.kind in (SHIFT, REDUCE)

    by_letter = result.snapshot_by_letter(2)
    assert by_letter is not None
    assert by_letter.letters == 2
    assert by_action.actions != by_letter.actions or by_action is by_letter


def test_snapshots_grow_monotonically():
    grammar = parse_cfg("S -> S S | a")
    result = parse(grammar, "aaa", SLR1)
    numbers = [snapshot.actions for snapshot in result.snapshots]
    assert numbers == list(range(1, len(numbers) + 1))
    letters = [snapshot.letters for snapshot in result.snapshots]
    assert letters == sorted(letters)


def test_drawings_are_produced():
    """«с предъявлением графа разбора на n-ом шаге» — рисунок нужен по условию."""
    grammar = parse_cfg("S -> S S | a")
    result = parse(grammar, "aaa", SLR1)
    picture = result.snapshots[-1].to_dot()
    assert picture.startswith("digraph")
    assert "->" in picture

    forest = result.forest.to_dot()
    assert forest.startswith("digraph")
    assert "пакет" in forest  # упаковка видна на рисунке


def test_lr0_and_slr1_reach_the_same_answer():
    """Таблицы разные, ответ обязан быть один."""
    grammar = parse_cfg("S -> a S b | a b")
    for word in ("ab", "aabb", "abb", "aab"):
        weak = parse(grammar, word, LR0)
        strong = parse(grammar, word, SLR1)
        assert weak.accepted == strong.accepted, word
        assert weak.parses == strong.parses, word


def test_accept_action_exists_for_the_augmented_rule():
    grammar = parse_cfg("S -> a")
    _, _, table = actions_table(grammar, SLR1)
    assert any(
        action.kind == ACCEPT for group in table.values() for action in group
    )


# --------------------------------------------------------------------------
# Generic LL(1) — варианты 2, 4, 5
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rules",
    [
        "S -> a S b | a b",
        "S -> a S a | b S b | c",
        "S -> a B\nB -> b B | b",
        "S -> a S | a S b | c",
        "S -> A B\nA -> a A | a\nB -> b B | b",
        "S -> a S b S | c",
    ],
)
def test_top_down_parsing_agrees_with_earley(rules):
    """Сверка с независимым разборщиком на всех словах длины ⩽ 6."""
    grammar = parse_cfg(rules)
    alphabet = "".join(sorted(grammar.terminals))
    for length in range(0, 7):
        for letters in itertools.product(alphabet, repeat=length):
            word = "".join(letters)
            expected = recognize(grammar, word)
            assert parse_ll(grammar, word, GRAPH).accepted == expected, (rules, word)
            assert parse_ll(grammar, word, TREE).accepted == expected, (rules, word)


def test_the_stack_node_is_a_return_point_not_a_symbol():
    """Ловушка, стоившая бы всей работы.

    Если помечать вершину гиперстека просто символом и позицией, то один
    и тот же нетерминал на разной глубине стека склеивается. На `S → aSb | ab`
    со словом `abb` из этого немедленно вырастает петля, стек порождает
    любое число `b`, и слово принимается зря. Вершина помечена пунктом
    `A → α B • β`, и такого не происходит.
    """
    grammar = parse_cfg("S -> a S b | a b")
    assert not parse_ll(grammar, "abb").accepted
    assert not recognize(grammar, "abb")
    assert parse_ll(grammar, "aabb").accepted


def test_left_recursion_is_refused_by_the_statement():
    grammar = parse_cfg("S -> S a | b")
    with pytest.raises(ValueError, match="без левой рекурсии"):
        parse_ll(grammar, "ba")


def test_top_down_conflicts_branch_instead_of_stopping():
    """Ячейка `(A, a)` с двумя правилами — это ветвление, а не отказ."""
    grammar = parse_cfg("S -> a S b | a b")
    table, conflicts = grammar.ll1_table()
    assert conflicts
    assert len(table[("S", "a")]) == 2
    assert parse_ll(grammar, "aaabbb").accepted


@pytest.mark.parametrize("word,position", [("ba", 0), ("a", 1), ("abb", 2)])
def test_top_down_reports_the_first_impossible_position(word, position):
    grammar = parse_cfg("S -> a S b | a b")
    result = parse_ll(grammar, word)
    assert not result.accepted
    assert result.error_position == position


def test_top_down_counts_rule_applications_as_steps():
    """> действие — применение правила из таблицы."""
    grammar = parse_cfg("S -> a S b | a b")
    result = parse_ll(grammar, "aabb")
    assert result.snapshots
    assert all(snapshot.kind == REDUCE for snapshot in result.snapshots)
    assert [s.actions for s in result.snapshots] == list(
        range(1, len(result.snapshots) + 1)
    )
    assert result.snapshot_by_action(1) is not None


def test_top_down_says_it_builds_no_forest():
    """Честная граница: лес для нисходящего разбора не строится."""
    grammar = parse_cfg("S -> a S b | a b")
    result = parse_ll(grammar, "aabb")
    assert result.accepted
    assert result.root is None
    assert result.parses == 0
    assert result.forest.families == {}
    assert "LL(1)" in result.summary()
