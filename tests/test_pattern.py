"""Переписывание образцов — семинар 05.09.2026, группа 52-Б.

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
