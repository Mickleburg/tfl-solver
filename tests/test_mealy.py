"""Автоматы Мили и Мура: преобразователи и алгоритмы над ними.

Главное, что здесь закреплено, — **сдвиг на один символ** при переводе.
Автомат Мура выдаёт `n+1` символ на слове длины `n`, автомат Мили ровно
`n`, и путать их нельзя: `Moore.to_mealy` теряет начальный выход,
`Mealy.to_moore` его добавляет.

Второе — что «не эквивалентны» здесь всегда подкреплено **словом**,
на котором автоматы расходятся, а не остаётся голым вердиктом.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.automata import equivalent
from tfl.mealy import Mealy, Moore

PARITY = Mealy(
    "01",
    "чёт",
    {
        ("чёт", "0"): "чёт",
        ("чёт", "1"): "нечёт",
        ("нечёт", "0"): "нечёт",
        ("нечёт", "1"): "чёт",
    },
    {
        ("чёт", "0"): "1",
        ("чёт", "1"): "0",
        ("нечёт", "0"): "0",
        ("нечёт", "1"): "1",
    },
)


def words(alphabet: str, limit: int):
    for length in range(limit + 1):
        for letters in itertools.product(alphabet, repeat=length):
            yield "".join(letters)


# --------------------------------------------------------------------------
# Мили
# --------------------------------------------------------------------------


def test_a_transition_without_an_output_is_refused():
    with pytest.raises(ValueError, match="нет выхода"):
        Mealy("a", "q", {("q", "a"): "q"}, {})


def test_the_output_word_is_as_long_as_the_input():
    for word in words("01", 6):
        assert len(PARITY.run(word)) == len(word)


def test_parity_is_computed_as_expected():
    """Единица на выходе, если прочитанных единиц чётное число."""
    for word in words("01", 7):
        expected = "".join(
            "1" if word[: i + 1].count("1") % 2 == 0 else "0" for i in range(len(word))
        )
        assert PARITY.run(word) == expected, word


def test_a_missing_transition_is_named():
    lonely = Mealy("ab", "q", {("q", "a"): "q"}, {("q", "a"): "x"})
    assert not lonely.is_complete()
    with pytest.raises(KeyError, match="перехода"):
        lonely.run("b")


def test_unreachable_states_are_dropped():
    machine = Mealy(
        "a",
        "q",
        {("q", "a"): "q", ("мусор", "a"): "мусор"},
        {("q", "a"): "0", ("мусор", "a"): "1"},
    )
    assert len(machine) == 2
    assert len(machine.trim()) == 1


def test_minimisation_keeps_the_behaviour_and_merges_twins():
    """Два состояния с одинаковой реакцией обязаны склеиться."""
    twin = Mealy(
        "01",
        "a",
        {
            ("a", "0"): "b",
            ("a", "1"): "c",
            ("b", "0"): "b",
            ("b", "1"): "c",
            ("c", "0"): "b",
            ("c", "1"): "c",
        },
        {
            ("a", "0"): "x",
            ("a", "1"): "y",
            ("b", "0"): "x",
            ("b", "1"): "y",
            ("c", "0"): "x",
            ("c", "1"): "y",
        },
    )
    small = twin.minimize()
    assert len(small) == 1
    assert twin.equivalent(small)


def test_minimisation_of_parity_changes_nothing():
    assert len(PARITY.minimize()) == len(PARITY)
    assert PARITY.equivalent(PARITY.minimize())


def test_inequivalence_comes_with_the_shortest_word():
    other = Mealy(
        "01",
        "чёт",
        dict(PARITY.delta),
        {**PARITY.output, ("нечёт", "1"): "0"},
    )
    word = PARITY.distinguishing(other)
    assert word == "11"
    assert PARITY.run(word) != other.run(word)


def test_machines_over_different_alphabets_are_refused():
    with pytest.raises(ValueError, match="разными алфавитами"):
        PARITY.distinguishing(Mealy("ab", "q", {}, {}))


# --------------------------------------------------------------------------
# Мура и перевод
# --------------------------------------------------------------------------


def test_a_moore_machine_outputs_one_symbol_more():
    """Тот самый сдвиг: до первого чтения автомат Мура уже что-то выдал."""
    labelled = Moore("01", "чёт", dict(PARITY.delta), {"чёт": "1", "нечёт": "0"})
    for word in words("01", 5):
        assert len(labelled.run(word)) == len(word) + 1, word


def test_the_conversion_marks_the_start_as_silent_and_keeps_lengths():
    """`to_moore` метит начальное состояние «выхода нет» — и сдвига не будет.

    Иначе перевод менял бы длину выходного слова, а это уже другой
    преобразователь, а не тот же самый в другой записи.
    """
    moore = PARITY.to_moore()
    assert moore.label[moore.start] is None
    for word in words("01", 5):
        assert moore.run(word) == PARITY.run(word), word


def test_the_round_trip_through_moore_keeps_behaviour():
    assert PARITY.equivalent(PARITY.to_moore().to_mealy())


def test_moore_to_mealy_loses_the_first_symbol():
    labelled = Moore(
        "01",
        "чёт",
        dict(PARITY.delta),
        {"чёт": "1", "нечёт": "0"},
    )
    for word in words("01", 5):
        if not word:
            continue
        assert labelled.to_mealy().run(word) == labelled.run(word)[1:], word


def test_moore_minimisation_merges_states_with_the_same_reaction():
    machine = Moore(
        "a",
        0,
        {(0, "a"): 1, (1, "a"): 2, (2, "a"): 1},
        {0: "x", 1: "x", 2: "x"},
    )
    assert len(machine.minimize()) == 1


# --------------------------------------------------------------------------
# Связь с распознавателями
# --------------------------------------------------------------------------


def test_the_transducer_language_is_regular_and_matches_the_run():
    """`to_dfa` обязан принимать ровно те слова, где последний выход нужный."""
    machine = PARITY.to_dfa("1")
    for word in words("01", 8):
        expected = bool(word) and PARITY.run(word)[-1] == "1"
        assert machine.accepts(word) is expected, word


def test_the_empty_word_is_never_in_the_transducer_language():
    """Выхода до чтения не было, значит и в язык пустое слово не входит."""
    assert not PARITY.to_dfa("1").accepts("")
    assert not PARITY.to_dfa("01").accepts("")


def test_moore_language_and_mealy_language_differ_exactly_on_epsilon():
    labelled = Moore("01", "чёт", dict(PARITY.delta), {"чёт": "1", "нечёт": "0"})
    from tfl.automata import union, difference

    only_epsilon = difference(labelled.to_dfa("1"), PARITY.to_dfa("1"))
    assert only_epsilon.shortest_word() == ""
    assert equivalent(
        union(PARITY.to_dfa("1"), labelled.to_dfa("1")), labelled.to_dfa("1")
    )


def test_both_can_be_drawn():
    assert "digraph" in PARITY.to_dot()
    assert "digraph" in PARITY.to_moore().to_dot()
