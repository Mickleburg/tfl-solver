"""Расширенные регулярки ЛР2: предпросмотр, ретроспектива, перевод в ПКА.

Две приёмочные точки, обе — примеры самой преподавательницы из issue #40:

* `^(?= .* a .* $) .* b .* $` — конъюнкция двух условий, ПКА получает
  И-ветвление в стартовой вершине;
* `^((?=(b*ab*ab*)*$)a*b)*$` — рекурсивный инвариант. Про него она сама
  пишет, какой язык выходит: «количество букв `a` между каждыми двумя
  буквами `b` чётно, причём перед последней их ненулевое количество».
  Это описание здесь выписано предикатом и сверено с распознавателем
  на всех словах длины ⩽ 9 — сходится независимая проверка её разбора
  и нашего кода.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.lookaround import (
    AHEAD,
    BEHIND,
    BEHIND_COURSE,
    BEHIND_STANDARD,
    NEG_AHEAD,
    Look,
    accepts,
    alphabet_of,
    check_equations,
    has_lookaround,
    matches,
    parse_extended,
    to_afa,
)

FIRST = "^(?= .* a .* $) .* b .* $"
SECOND = "^((?=(b*ab*ab*)*$)a*b)*$"


def words_upto(limit: int, alphabet: str = "ab") -> list[str]:
    return [
        "".join(letters)
        for length in range(0, limit + 1)
        for letters in itertools.product(alphabet, repeat=length)
    ]


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_anchors_are_mandatory_by_the_statement():
    """> (обязательно) маркеры начала и конца выражения ^ и $."""
    with pytest.raises(ValueError, match="обязательных маркеров"):
        parse_extended("(a|b)*")
    assert parse_extended("^(a|b)*$") is not None


def test_spaces_do_not_matter():
    """В условии пробелы стоят для читаемости: `^(?= .* a .* $) .* b .* $`."""
    assert str(parse_extended(FIRST)) == str(
        parse_extended("^(?=.*a.*$).*b.*$")
    )


def test_sugar_is_expanded():
    """`τ⁺ = ττ*`, `τ? = (τ|ε)` — прямо по условию."""
    plus = parse_extended("^a+$")
    option = parse_extended("^a?$")
    assert accepts(plus, "a", "ab") and accepts(plus, "aaa", "ab")
    assert not accepts(plus, "", "ab")
    assert accepts(option, "", "ab") and accepts(option, "a", "ab")
    assert not accepts(option, "aa", "ab")


def test_letter_classes_and_their_complements():
    inside = parse_extended("^[ab]*$")
    outside = parse_extended("^[^a]*$")
    assert accepts(inside, "abab", "abc")
    assert not accepts(inside, "abc", "abc")
    assert accepts(outside, "bcbc", "abc")
    assert not accepts(outside, "bca", "abc")


def test_wildcard_uses_the_given_alphabet():
    node = parse_extended("^.$")
    assert accepts(node, "c", "abc")
    assert not accepts(node, "c", "ab")


@pytest.mark.parametrize(
    "text,message",
    [
        ("^(a$", "не закрыта скобка"),
        ("^[]$", "пустой класс"),
        ("^(?<x)$", "неизвестная группа"),
        ("^(*a)$", "без операнда"),
    ],
)
def test_broken_expressions_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_extended(text)


def test_lookaround_is_detected_and_alphabet_collected():
    node = parse_extended(FIRST)
    assert has_lookaround(node)
    assert alphabet_of(node) == frozenset("ab")
    assert not has_lookaround(parse_extended("^a*b$"))


# --------------------------------------------------------------------------
# Первый пример преподавателя
# --------------------------------------------------------------------------


def test_the_first_example_is_the_conjunction_of_two_conditions():
    """> конъюнкция двух ДКА, один проверяет наличие `a`, другой наличие `b`."""
    node = parse_extended(FIRST)
    for word in words_upto(7):
        expected = "a" in word and "b" in word
        assert accepts(node, word, "ab") == expected, word


def test_the_first_example_translates_to_an_and_branching_afa():
    verdict = to_afa(parse_extended(FIRST), "ab")
    assert verdict.value is True, verdict.reason
    assert "И-ветвление" in verdict.reason
    automaton = verdict.witness
    assert automaton.universal
    for word in words_upto(7):
        assert automaton.accepts(word) == ("a" in word and "b" in word), word


def test_a_negative_lookahead_also_becomes_an_and_branch():
    node = parse_extended("^(?! .* c .* $) .* b .* $")
    verdict = to_afa(node, "abc")
    assert verdict.value is True, verdict.reason
    for word in words_upto(5, "abc"):
        assert verdict.witness.accepts(word) == ("c" not in word and "b" in word), word


# --------------------------------------------------------------------------
# Второй пример преподавателя
# --------------------------------------------------------------------------


def teacher_language(word: str) -> bool:
    """> количество букв `a` между каждыми двумя буквами `b` чётно,
    > причём перед последней их ненулевое количество.
    """
    if word == "":
        return True
    if not word.endswith("b"):
        return False
    blocks = word.split("b")[:-1]
    return all(len(block) % 2 == 0 for block in blocks) and len(blocks[-1]) > 0


def test_the_recogniser_reproduces_the_teachers_own_analysis():
    """Независимая сверка: её описание языка против нашего распознавателя."""
    node = parse_extended(SECOND)
    for word in words_upto(9):
        assert accepts(node, word, "ab") == teacher_language(word), word


def test_the_second_example_translates_to_a_recursive_invariant():
    verdict = to_afa(parse_extended(SECOND), "ab")
    assert verdict.value is True, verdict.reason
    assert "рекурсивный инвариант" in verdict.reason
    for word in words_upto(8):
        assert verdict.witness.accepts(word) == teacher_language(word), word


def test_the_dollar_inside_a_lookahead_is_not_decoration():
    """`(?=τ$)` требует лечь на **весь** остаток, приписывать `.*` нельзя.

    Без этой оговорки инвариант превращается в «где-то дальше найдётся
    чётное число `a`», и язык раздувается.
    """
    with_end = parse_extended(SECOND)
    without_end = parse_extended("^((?=(b*ab*ab*)*)a*b)*$")
    differing = [
        word
        for word in words_upto(6)
        if accepts(with_end, word, "ab") != accepts(without_end, word, "ab")
    ]
    assert differing, "оговорка про $ должна быть видна на конкретном слове"
    assert accepts(without_end, "ab", "ab")
    assert not accepts(with_end, "ab", "ab")


# --------------------------------------------------------------------------
# Печатные равенства против позиционного прочтения
# --------------------------------------------------------------------------


def test_the_printed_equations_agree_at_the_top_level():
    """`τ₀(?= τ₁)τ₂ ≡ τ₀((τ₁.*) ∩ τ₂)` — считаем язык операциями и сверяем."""
    for text in (FIRST, "^(?! .* a .* $) .* b .* $", "^a(?= b .* $) .* $"):
        verdict = check_equations(parse_extended(text), "ab", 7)
        assert verdict.value is True, (text, verdict.reason)


def test_under_a_star_the_equations_do_not_apply_and_that_is_said():
    """Честная граница: печатные равенства заданы для верхнего уровня."""
    verdict = check_equations(parse_extended(SECOND), "ab", 6)
    assert verdict.value is None
    assert "под звёздочкой" in verdict.reason


# --------------------------------------------------------------------------
# Ретроспектива: два прочтения
# --------------------------------------------------------------------------


def test_the_two_readings_of_lookbehind_differ():
    """Печатное равенство даёт «префикс **начинается** с τ₁», PCRE — «кончается».

    Свидетель: `^ab(?<=a).*$`. Префикс `ab` начинается с `a`, но кончается
    на `b`, поэтому прочтения расходятся.
    """
    node = parse_extended("^ab(?<=a).*$")
    assert accepts(node, "ab", "ab", BEHIND_COURSE)
    assert not accepts(node, "ab", "ab", BEHIND_STANDARD)


def test_the_course_reading_matches_the_printed_equation():
    """`τ₀(?<= τ₁)τ₂ ≡ (τ₀ ∩ (τ₁.*))τ₂` — то же самое, посчитанное автоматами."""
    verdict = check_equations(parse_extended("^ab(?<=a).*$"), "ab", 7)
    assert verdict.value is True, verdict.reason


def test_a_negative_lookbehind_is_the_complement():
    node = parse_extended("^ab(?<!a).*$")
    assert not accepts(node, "ab", "ab", BEHIND_COURSE)
    assert accepts(node, "ab", "ab", BEHIND_STANDARD)


def test_an_unknown_reading_is_refused():
    node = parse_extended("^a(?<=a)$")
    with pytest.raises(ValueError, match="прочтение"):
        accepts(node, "a", "ab", "как-нибудь")


# --------------------------------------------------------------------------
# Границы перевода в ПКА
# --------------------------------------------------------------------------


def test_a_lookahead_in_the_middle_is_not_translated_mechanically():
    """Общего механического перевода нет, и это сказано, а не замолчано."""
    verdict = to_afa(parse_extended("^a(?= b .* $) .* $"), "ab")
    assert verdict.value is None
    assert "нвариант" in verdict.reason


def test_a_plain_expression_needs_no_and_branching():
    verdict = to_afa(parse_extended("^a*b$"), "ab")
    assert verdict.value is None


def test_the_built_automaton_is_verified_before_it_is_returned():
    """Сверка стоит внутри перевода: неверный ПКА наружу не выходит."""
    verdict = to_afa(parse_extended(FIRST), "ab", max_len=6)
    assert verdict.value is True
    assert "сверен с распознавателем" in verdict.reason


# --------------------------------------------------------------------------
# Распознаватель как таковой
# --------------------------------------------------------------------------


def test_matches_returns_all_end_positions():
    node = parse_extended("^a*$")
    assert matches(node.parts[1], "aaa", 0, "a") == frozenset({0, 1, 2, 3})


def test_an_empty_iteration_does_not_hang():
    """`(ε)*` — классическое место зацикливания разбора с возвратами."""
    node = parse_extended("^(a?)*b$")
    assert accepts(node, "aab", "ab")
    assert not accepts(node, "aa", "ab")


def test_lookaround_consumes_nothing():
    node = Look(AHEAD, parse_extended("a", anchored=False))
    assert matches(node, "abc", 0, "abc") == frozenset({0})
    negative = Look(NEG_AHEAD, parse_extended("a", anchored=False))
    assert matches(negative, "abc", 0, "abc") == frozenset()
    assert matches(negative, "bca", 0, "abc") == frozenset({0})


# --------------------------------------------------------------------------
# Фазз-сверка, которую требует условие
# --------------------------------------------------------------------------


def test_fuzz_agrees_with_an_equivalent_automaton():
    """> строится случайное слово ω и проверяется… согласованно."""
    from tfl.automata import dfa_of
    from tfl.lookaround import fuzz

    node = parse_extended(FIRST)
    both = dfa_of(
        "((a|b)*a(a|b)*b(a|b)*)|((a|b)*b(a|b)*a(a|b)*)", "ab"
    ).minimize()
    verdict = fuzz(node, {"ДКА": both, "ПКА": to_afa(node, "ab").witness}, "ab", 300, 10)
    assert verdict.value is True, verdict.reason
    assert "доказательством не является" in verdict.reason


def test_fuzz_catches_a_wrong_recogniser_with_a_witness():
    from tfl.automata import dfa_of
    from tfl.lookaround import fuzz

    node = parse_extended(FIRST)
    only_a = dfa_of("(a|b)*a(a|b)*", "ab").minimize()
    verdict = fuzz(node, {"ДКА только про a": only_a}, "ab", 300, 8)
    assert verdict.value is False
    assert verdict.witness is not None
    assert accepts(node, verdict.witness, "ab") != only_a.accepts(verdict.witness)
