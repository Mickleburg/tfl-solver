"""Переписывание образцов — обезличенные регрессионные примеры.

Приёмочная точка — задача 1: `P = {aXb → bXa, Xb → aaX}`. Разбор семинара
опирался на меру `(|w|_b, Σ позиций b)` и инвариант `μ(w) = |w|_a + 2|w|_b`;
здесь оба проверяются исполнением, а не пересказом. Задача 2 — то же
моделирование маркером-долгом, сверенное в обе стороны.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.pattern import PatternSystem, matches, parse_patterns
from tfl.srs import parse_srs


def seminar_system() -> PatternSystem:
    return parse_patterns("aXb -> bXa\nXb -> aaX", variables="X")


def words_upto(limit: int, alphabet: str = "ab") -> list[str]:
    return [
        "".join(letters)
        for length in range(1, limit + 1)
        for letters in itertools.product(alphabet, repeat=length)
    ]


# --------------------------------------------------------------------------
# Сопоставление
# --------------------------------------------------------------------------


def test_a_variable_covers_an_arbitrary_piece():
    """Именно этим образец отличается от строкового правила."""
    found = sorted(
        (end, binding["X"])
        for end, binding in matches("aXb", "aXXb".replace("X", "c"), 0, frozenset("X"), {})
    )
    assert found == [(4, "cc")]


def test_a_repeated_variable_must_agree_with_itself():
    assert list(matches("XaX", "bab", 0, frozenset("X"), {})) == [(3, {"X": "b"})]
    assert list(matches("XaX", "bac", 0, frozenset("X"), {})) == []


def test_the_empty_binding_is_allowed():
    """`Xb` с `X = ε` — законное сопоставление, и на нём держится разбор."""
    system = seminar_system()
    assert "aab" in system.step("bb")


# --------------------------------------------------------------------------
# Задача 1 семинара
# --------------------------------------------------------------------------


def test_the_measure_of_the_seminar_is_an_invariant():
    """> Ключевое наблюдение: мера `μ(w) = |w|_a + 2|w|_b` — инвариант.

    Правило 1 меняет `a` и `b` местами, правило 2 убирает `b`
    и добавляет два `a`: `−2 + 2 = 0`.
    """
    verdict = seminar_system().check_invariant(
        lambda w: w.count("a") + 2 * w.count("b"), words_upto(4), max_len=9
    )
    assert verdict.value is True, verdict.reason
    assert "сохраняется" in verdict.reason


def test_the_well_founded_measure_decreases():
    """> Фундированная мера: пара `(|w|_b, Σ позиций b)` в лексикографическом
    > порядке. Правило 2 уменьшает первую компоненту, правило 1 её сохраняет,
    > а вторую уменьшает, потому что `b` сдвигается влево.
    """
    verdict = seminar_system().check_measure(
        lambda w: (w.count("b"), sum(i for i, c in enumerate(w) if c == "b")),
        words_upto(4),
        max_len=9,
    )
    assert verdict.value is True, verdict.reason
    assert "фундирована" in verdict.reason  # оговорка на месте


def test_a_wrong_measure_is_refuted_with_the_step():
    """Оракул обязан ловить меру, которая не убывает, и показывать где."""
    verdict = seminar_system().check_measure(lambda w: len(w), ["ab"], max_len=6)
    assert verdict.value is False
    assert "не убывает" in verdict.reason


def test_normal_form_is_a_to_the_power_of_the_measure():
    """> Из инвариантности меры нормальная форма единственна и равна `a^μ(w)`."""
    system = seminar_system()
    for word in words_upto(3):
        measure = word.count("a") + 2 * word.count("b")
        assert system.normal_forms(word, max_len=measure) == {"a" * measure}, word


def test_single_normal_form_is_not_confluence():
    """Единственность на срезе — наблюдение, и вердикт этого не скрывает."""
    verdict = seminar_system().confluent_on(words_upto(3), max_len=8)
    assert verdict.value is None
    assert "конфлюэнтностью это не является" in verdict.reason


# --------------------------------------------------------------------------
# Задача 2 семинара: моделирование маркером-долгом
# --------------------------------------------------------------------------


def marker_system():
    """Приём семинара: долг `M` для правила 1, долг `N` для правила 2."""
    return parse_srs(
        "b -> Ma\naM -> Ma\nbM -> Mb\naM -> b\n"
        "b -> N\naN -> Na\nbN -> Nb\nN -> aa"
    )


def test_the_marker_system_reaches_exactly_the_same_words():
    """> Проверено в обе стороны… множество достижимых из `^w$` строк
    > без маркеров **совпадает** с множеством достижимых из `w` в `P`.
    """
    verdict = seminar_system().agrees_with_srs(
        marker_system(), words_upto(4), wrap=lambda w: f"^{w}$", max_len=7
    )
    assert verdict.value is True, verdict.reason


def test_only_marker_free_words_count():
    """Строка с недогашенным маркером — промежуточная, а не достижимая.

    Без этого фильтра моделирование «добавляет» `^Ma$` и сверка рушится
    на первом же слове.
    """
    system = seminar_system()
    assert "M" not in system.alphabet
    assert system.alphabet == frozenset("ab")


def test_the_length_cap_must_be_the_same_on_both_sides():
    """Разные потолки дают ложное расхождение, а не настоящее.

    Строковой системе нужен запас на маркеры, поэтому её обход идёт
    дальше; но в сравнение берутся слова одной и той же длины.
    """
    verdict = seminar_system().agrees_with_srs(
        marker_system(), ["bbbb"], wrap=lambda w: f"^{w}$", max_len=8
    )
    assert verdict.value is True, verdict.reason


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_variables_are_declared_not_guessed():
    system = parse_patterns("variables = [X]\naXb -> bXa")
    assert system.variables == frozenset("X")
    assert str(system) == "aXb → bXa"


def test_a_free_variable_on_the_right_is_rejected():
    with pytest.raises(ValueError, match="справа, но не слева"):
        parse_patterns("ab -> aYb", variables="Y")


def test_a_rule_without_an_arrow_is_rejected():
    with pytest.raises(ValueError, match="нет стрелки"):
        parse_patterns("aXb bXa", variables="X")


def test_a_loop_is_found_when_the_left_side_is_nested_in_the_right():
    """Цикла нет, петля есть — та же ловушка, что в `tfl/srs.py`."""
    system = parse_patterns("aXb -> aXbb", variables="X")
    loop = system.find_loop(["ab"], max_len=8)
    assert loop is not None
    assert loop[0] == "ab" and "ab" in loop[-1] and loop[-1] != "ab"


# --------------------------------------------------------------------------
# Критические пары
# --------------------------------------------------------------------------


VARIABLE_FREE = [
    "ab -> c\nbc -> a",
    "aa -> b\nab -> a",
    "ad -> (d\nda -> )a",
    "abc -> d\nbc -> e\nc -> f",
    "aa -> a",
    "ab -> ba",
    "ab -> c\nba -> c",
]


@pytest.mark.parametrize("text", VARIABLE_FREE)
def test_without_variables_the_overlaps_are_the_srs_ones(text):
    """Сверка с независимой реализацией: без переменных ответ обязан совпасть."""
    patterns = parse_patterns(text, variables="")
    system = parse_srs(text)
    mine = {
        (overlap.word, frozenset({overlap.left, overlap.right}))
        for overlap in patterns.overlaps()
    }
    theirs = {
        (word, frozenset({left, right}))
        for word, left, right, _, _ in system.critical_pairs()
    }
    assert mine == theirs


@pytest.mark.parametrize("text", VARIABLE_FREE)
def test_without_variables_the_verdict_is_the_srs_one(text):
    patterns = parse_patterns(text, variables="")
    assert patterns.locally_confluent(0, 14).value == parse_srs(text).locally_confluent(14).value


def test_without_variables_the_enumeration_is_complete_and_says_so():
    """Только здесь вывод «локально конфлюэнтна» доказателен."""
    verdict = parse_patterns("ab -> ba", variables="").locally_confluent(0, 14)
    assert verdict.value is True
    assert "перебор наложений полон" in verdict.reason


def test_with_variables_a_positive_answer_stays_unproved():
    """Наложений бесконечно много: «все проверенные сошлись» — не довод."""
    verdict = seminar_system().locally_confluent(1, 12)
    assert verdict.value is None
    assert "словесным уравнениям" in verdict.reason


def test_the_seminar_system_has_converging_critical_pairs():
    """Разбор семинара: система конфлюэнтна — значит пары обязаны сходиться."""
    system = seminar_system()
    pairs = system.overlaps(1)
    assert len(pairs) >= 10
    for overlap in pairs:
        assert system.joinable(overlap.left, overlap.right, 12).value is not False


def test_a_diverging_critical_pair_is_a_proof():
    """`aX → X` и `Xa → b` спорят за слово `a`: `ε` и `b` — разные Н.Ф."""
    system = parse_patterns("aX -> X\nXa -> b", variables="X")
    verdict = system.locally_confluent(1, 10)
    assert verdict.value is False
    assert "вычислены полностью" in verdict.reason
    assert verdict.witness.word == "a"
    assert {verdict.witness.left, verdict.witness.right} == {"", "b"}


def test_two_matchings_at_one_position_are_a_critical_pair():
    """Того, чего у строк не бывает: один редекс, два разбора.

    `aXb` ложится на `abb` двумя способами — `X = ε` и `X = b`, —
    и результаты разные.
    """
    system = parse_patterns("aXb -> bXa", variables="X")
    found = [
        overlap
        for overlap in system.overlaps(1)
        if overlap.word == "abb" and overlap.shift == 0
    ]
    assert {overlap.left for overlap in found} == {"bab", "bba"}


def test_an_overlap_inside_a_variable_is_not_critical():
    """Классическая оговорка: переписывание внутри переменной сходится само."""
    system = parse_patterns("aXb -> c" + chr(10) + "aa -> d", variables="X")
    pairs = system.overlaps(2)
    # В слове `aaab` образец `aXb` ложится с `X = aa`: разметка a[0,1) X[1,3) b[3,4).
    # Редекс `aa` на позиции 1 сидит целиком внутри `X` — не критический.
    assert not [
        overlap
        for overlap in pairs
        if overlap.word == "aaab" and overlap.shift == 1 and overlap.second.lhs == "aa"
    ]
    # А тот же `aa` на позиции 0 задевает букву самого образца — критический.
    assert [
        overlap
        for overlap in pairs
        if overlap.word == "aaab" and overlap.shift == 0 and overlap.second.lhs == "aa"
    ]


def test_a_non_linear_left_side_keeps_variable_overlaps():
    """Если переменная повторяется, оговорка снимается — и пар становится больше."""
    linear = parse_patterns("aXbY -> c\naa -> d", variables="XY")
    repeated = parse_patterns("aXbX -> c\naa -> d", variables="X")
    assert linear.rules[0].is_linear(linear.variables)
    assert not repeated.rules[0].is_linear(repeated.variables)
    assert len(repeated.overlaps(1)) > len(linear.overlaps(1))


def test_joinable_has_three_outcomes():
    system = seminar_system()
    assert system.joinable("ab", "ab").value is True
    assert system.joinable("ab", "aaa", 8).value is True
    growing = parse_patterns("aXb -> aXbb", variables="X")
    assert growing.joinable("ab", "ba", 6).value is None
    dead = parse_patterns("a -> b\nc -> d", variables="")
    assert dead.joinable("a", "c", 6).value is False
