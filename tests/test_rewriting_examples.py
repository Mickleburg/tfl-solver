"""Регрессии новых обезличенных примеров переписывания и копредставлений."""

from __future__ import annotations

from itertools import product

from tfl.cayley import normalize
from tfl.presentation import SEMIGROUP, parse_presentation
from tfl.srs import parse_srs


def words(max_len: int):
    for length in range(max_len + 1):
        for letters in product("ab", repeat=length):
            yield "".join(letters)


def occurrences(word: str, factor: str) -> int:
    return sum(
        word.startswith(factor, index)
        for index in range(len(word) - len(factor) + 1)
    )


def closure(start: str, step) -> set[str]:
    seen = {start}
    queue = [start]
    while queue:
        current = queue.pop()
        for following in step(current):
            if following not in seen:
                seen.add(following)
                queue.append(following)
    return seen


def test_all_four_critical_pairs_are_preserved():
    system = parse_srs("ababa -> bb\nbabb -> aa")
    assert [
        (word, left, right)
        for word, left, right, _first, _second in system.critical_pairs()
    ] == [
        ("ababababa", "bbbaba", "ababbb"),
        ("abababa", "bbba", "abbb"),
        ("abababb", "bbbb", "abaaa"),
        ("babbabb", "aaabb", "babaa"),
    ]


def test_subword_measure_decreases_on_every_short_step():
    system = parse_srs("aabb -> abab\naba -> bab")

    def measure(word: str) -> tuple[int, int]:
        return word.count("a"), occurrences(word, "aa")

    for word in words(8):
        assert all(measure(following) < measure(word) for following in system.step(word))


def test_marker_srs_models_the_anchored_pattern_on_clean_words():
    marker_system = parse_srs(
        "^ -> P\n"
        "Pa -> AP\n"
        "Pb -> BP\n"
        "P -> D\n"
        "Da -> D\n"
        "Db -> D\n"
        "Db -> R\n"
        "AR -> Ra\n"
        "BR -> Rb\n"
        "R -> ^"
    )

    def pattern_step(word: str) -> set[str]:
        return {
            word[:start] + word[end + 1 :]
            for end, letter in enumerate(word)
            if letter == "b"
            for start in range(end + 1)
        }

    def clean_reachable(word: str) -> set[str]:
        return {
            candidate[1:-1]
            for candidate in closure(f"^{word}$", marker_system.step)
            if candidate.startswith("^")
            and candidate.endswith("$")
            and candidate.count("^") == 1
            and candidate.count("$") == 1
            and set(candidate[1:-1]) <= set("ab")
        }

    for word in words(5):
        assert closure(word, pattern_step) == clean_reachable(word)


def test_finite_semigroup_excludes_the_adjoined_empty_word():
    presentation = parse_presentation("ab = baa\naaa = bb\naba = b", SEMIGROUP)
    graph = presentation.cayley().witness

    assert graph.order == 9
    assert graph.presented_order == 8
    assert graph.presented_elements == (
        "a",
        "b",
        "aa",
        "ab",
        "ba",
        "bb",
        "abb",
        "bab",
    )
    assert presentation.is_finite().witness == 8

    system, completed = presentation.complete()
    assert completed.value is True
    for _word, left, right, _first, _second in system.critical_pairs():
        assert normalize(system, left) == normalize(system, right)


def test_the_group_presentation_has_order_six():
    presentation = parse_presentation("aaa = 1\nbb = 1\nab = baa")
    graph = presentation.cayley().witness
    assert graph.presented_order == 6
