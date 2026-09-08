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
from tfl.conj import parse_conjunctive
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
    parse_conj,
    parse_ll,
    relaxed_cfg,
)
from tfl.parse import recognize
from tfl.words import iter_words

CATALAN = [1, 1, 2, 5, 14, 42, 132]


def count_trees(grammar: CFG, word: str) -> int:
    """Независимый счёт деревьев вывода: перебор разбиений отрезка.

    Реализация нарочно прямая и медленная — она арбитр, а не парсер.
    ε-правила допускаются (нисходящий разбор их разрешает), а вот циклы
    `A ⇒⁺ A` нет: развёрнутый вход даёт ноль. Для грамматик без левой
    рекурсии и без ε-правил цикла и не бывает.
    """
    memo: dict[tuple[str, int, int], int] = {}

    def splits(symbols, start, end):
        if not symbols:
            return 1 if start == end else 0
        total = 0
        for middle in range(start, end + 1):
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


# --------------------------------------------------------------------------
# Бонус +6: лес для нисходящего разбора
# --------------------------------------------------------------------------

#: Грамматики без левой рекурсии, на которых сверяется нисходящий лес.
TOP_DOWN = [
    ("S -> a S b | a b", "ab", 7),
    ("S -> a S S | a", "a", 9),
    ("S -> a B | a C\nB -> b\nC -> b", "ab", 4),
    ("S -> a S b S | ε", "ab", 6),
    ("S -> A B C\nA -> a | ε\nB -> b | ε\nC -> c | ε", "abc", 4),
]


@pytest.mark.parametrize("rules,alphabet,cap", TOP_DOWN)
def test_the_top_down_forest_counts_exactly_as_many_trees(rules, alphabet, cap):
    """Та же приёмочная точка, что у восходящего: лес и перебор сходятся.

    Перебор ничего не знает ни про гиперстек, ни про бинаризацию —
    он просто режет отрезок по правилам, — поэтому совпадение чисел
    и есть проверка конструкции.
    """
    grammar = parse_cfg(rules)
    for length in range(cap + 1):
        for letters in itertools.product(alphabet, repeat=length):
            word = "".join(letters)
            result = parse_ll(grammar, word, GRAPH)
            expected = count_trees(grammar, word)
            assert result.accepted == (expected > 0), (rules, word)
            assert result.parses == expected, (rules, word)


def test_the_top_down_forest_is_binarised():
    """Узлы на пункты `A → α • β` — то, чего при восходящем разборе нет."""
    grammar = parse_cfg("S -> a S S | a")
    result = parse_ll(grammar, "aaaaa")
    assert result.forest.intermediate
    assert all("•" in label[0] for label in result.forest.intermediate)
    bottom = parse(parse_cfg("S -> S S | a"), "aaa", SLR1)
    assert not bottom.forest.intermediate


def test_binarisation_does_not_leak_into_the_trees():
    """Узлы бинаризации вклеиваются в родителя: у `S → aSS` три потомка."""
    grammar = parse_cfg("S -> a S S | a")
    result = parse_ll(grammar, "aaaaa")
    trees = result.forest.trees(result.root)
    assert len(trees) == result.parses == 2
    assert all(tree[0] == "S" for tree in trees)
    assert {len(tree) - 1 for tree in trees} == {3}

    def symbols(tree):
        return {tree[0]} | {s for child in tree[1:] for s in symbols(child)}

    assert symbols(trees[0]) <= grammar.nonterminals | grammar.terminals


def test_binarised_forest_stays_small_while_parses_explode():
    """Ради чего лес и пакуется: разборов Каталаново много, узлов — `O(n²)`."""
    grammar = parse_cfg("S -> a S S | a")
    counted, sizes = [], []
    for length in range(1, 12, 2):
        result = parse_ll(grammar, "a" * length)
        counted.append(result.parses)
        sizes.append(len(result.forest.families))
    assert counted == CATALAN[:6]
    assert sizes == [1, 6, 15, 28, 45, 66]


def test_a_single_symbol_prefix_gets_no_node_of_its_own():
    """`A → x • β` при непустом `β` отдаёт наверх узел самого `x`."""
    grammar = parse_cfg("S -> a S b | a b")
    result = parse_ll(grammar, "aabb")
    starts = {label[0] for label in result.forest.intermediate}
    assert not any(text.startswith("S → a •") for text in starts)


def test_the_empty_right_hand_side_becomes_a_leaf():
    """ε-правила при нисходящем разборе разрешены, и в лесу у них лист."""
    grammar = parse_cfg("S -> a S b S | ε")
    result = parse_ll(grammar, "ab")
    assert ("ε", 2, 2) in {
        child
        for group in result.forest.families.values()
        for _, children in group
        for child in children
    }
    assert result.forest.yield_of(result.root) == "ab"


@pytest.mark.parametrize("word", ["ab", "aabb", "aaabbb"])
def test_children_are_stored_left_to_right(word):
    """Порядок потомков в лесу — порядок символов правой части.

    Проверка нужна именно такая: число разборов от перестановки
    потомков не меняется, поэтому счёт её не ловит. Ловит только
    слово, собранное обратно по листьям.
    """
    grammar = parse_cfg("S -> a S b | a b")
    for result in (parse(grammar, word, SLR1), parse_ll(grammar, word)):
        assert result.forest.yield_of(result.root) == word


def test_the_yield_of_an_asymmetric_rule_is_not_mirrored():
    grammar = parse_cfg("S -> a B c\nB -> b")
    result = parse(grammar, "abc", SLR1)
    assert result.forest.trees(result.root) == [("S", ("a",), ("B", ("b",)), ("c",))]


# --------------------------------------------------------------------------
# Бонус +4: конъюнктивные грамматики гиперстеком
# --------------------------------------------------------------------------

#: Грамматика лекции 11 для `{(aⁿb)ᵏ | n, k ⩾ 1}`: блоки обязаны быть равными.
BLOCKS = """
    S -> S A & C b | A
    A -> a A | a b
    C -> a C a | B
    B -> B A | b
"""


def test_relaxed_grammar_replaces_conjunction_with_alternative():
    grammar = parse_conjunctive("A -> B & C | B\nB -> b\nC -> b")
    relaxed, origin = relaxed_cfg(grammar)
    assert [str(p) for p in relaxed.productions] == ["A → B", "A → C", "B → b", "C → b"]
    # `A → B` — это и конъюнкт первого правила, и всё второе правило целиком.
    assert origin[relaxed.productions[0]] == ((0, 0), (1, 0))


@pytest.mark.parametrize("mode", [SLR1, LR0])
def test_the_conjunctive_parser_agrees_with_the_recogniser(mode):
    """Арбитр — обобщённый CYK из `tfl/conj.py`: другой алгоритм, тот же ответ."""
    grammar = parse_conjunctive(BLOCKS)
    for word in iter_words("ab", 8):
        assert parse_conj(grammar, word, mode).accepted == grammar.recognize(word), word


def test_a_single_conjunct_is_not_enough_to_push():
    """Проверка того, ради чего всё и делается: `&` строже, чем `|`.

    Слово `abaab` разбирается по послаблению (блок `ab`, потом `aab`),
    но конъюнкцию не проходит: блоки разной длины.
    """
    grammar = parse_conjunctive(BLOCKS)
    relaxed, _ = relaxed_cfg(grammar)
    assert recognize(relaxed, "abaab")
    assert not grammar.recognize("abaab")
    result = parse_conj(grammar, "abaab")
    assert not result.accepted
    assert result.error_position is not None


def test_the_tree_has_one_subtree_per_conjunct():
    """У узла `A` столько поддеревьев, сколько конъюнктов, и все — про одно слово."""
    grammar = parse_conjunctive(BLOCKS)
    result = parse_conj(grammar, "aabaab")
    assert result.accepted
    assert result.parses == 1
    families = result.forest.families[result.root]
    assert len(families) == 1
    _, children = next(iter(families))
    assert len(children) == 2
    assert all(child[1:] == (0, 6) for child in children)
    assert {result.forest.yield_of(child) for child in children} == {"aabaab"}
    assert result.forest.yield_of(result.root) == "aabaab"


def test_a_grammar_without_conjunction_parses_exactly_as_a_context_free_one():
    text = "S -> S S | a"
    plain = parse_cfg(text)
    grammar = parse_conjunctive(text)
    for length in range(1, 6):
        word = "a" * length
        assert parse_conj(grammar, word).parses == parse(plain, word, SLR1).parses


def test_conjunctive_parsing_refuses_epsilon_rules():
    """Ограничение условия то же, что у восходящего разбора."""
    grammar = parse_conjunctive("S -> a S & S a | ε")
    with pytest.raises(ValueError, match="без ε-правил"):
        parse_conj(grammar, "aa")


def test_the_conjunctive_stack_is_graph_shaped_by_the_statement():
    """> Актуальны… только если реализуется графовидный стек."""
    grammar = parse_conjunctive(BLOCKS)
    result = parse_conj(grammar, "aabaab")
    assert result.sharing == GRAPH
    assert "конъюнктивная" in result.summary()
