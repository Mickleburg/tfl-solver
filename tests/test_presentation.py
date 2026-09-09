"""Копредставления: ориентация соотношений, пополнение, проблема равенства.

Ключевое различие, которое здесь закрепляется: **равенство доказывается
без конфлюэнтности, а неравенство — нет**. Совпадение нормальных форм это
готовая цепочка соотношений, и она верна независимо от того, сошлось
пополнение или нет; вывод «не равны» требует полной системы, иначе слова
могли бы сойтись где-то дальше.
"""

from __future__ import annotations

import pytest

from tfl.presentation import (
    GROUP,
    SEMIGROUP,
    Presentation,
    inverse,
    parse_presentation,
)


# --------------------------------------------------------------------------
# Запись
# --------------------------------------------------------------------------


def test_inverse_reverses_and_flips_case():
    assert inverse("abA") == "aBA"
    assert inverse(inverse("abBA")) == "abBA"
    assert inverse("") == ""


def test_the_parser_understands_three_ways_to_write_an_inverse():
    first = parse_presentation("группа: a, b\na b a^-1 = b b")
    second = parse_presentation("группа: a, b\na b a' = bb")
    third = parse_presentation("группа: a, b\nabA = bb")
    assert first.relations == second.relations == third.relations == (("abA", "bb"),)


def test_the_empty_word_has_several_spellings():
    for token in ("1", "e", "ε"):
        assert parse_presentation(f"группа: a\naa = {token}").relations == (("aa", ""),)


def test_generators_are_inferred_when_the_header_is_absent():
    assert parse_presentation("ab = ba").generators == "ab"


def test_uppercase_generators_are_refused():
    """Заглавная буква занята под обратный элемент, и это не переопределяется."""
    with pytest.raises(ValueError, match="строчными"):
        Presentation("aB", (("a", "a"),))


def test_a_semigroup_has_no_inverses():
    with pytest.raises(ValueError, match="обратные элементы"):
        Presentation("ab", (("aB", "b"),), SEMIGROUP)


def test_a_semigroup_gets_no_free_reduction_rules():
    """`aa⁻¹ = ε` — свойство группы, а не полугруппы."""
    assert parse_presentation("полугруппа: a, b\nab = ba", SEMIGROUP).free_rules() == ()
    assert len(parse_presentation("группа: a, b\nab = ba").free_rules()) == 4


# --------------------------------------------------------------------------
# Ориентация и пополнение
# --------------------------------------------------------------------------


def test_orientation_makes_every_rule_decrease():
    """Отсюда завершимость: армейский порядок фундирован."""
    presentation = parse_presentation("группа: a, b\nbb = aba^-1")
    system = presentation.rewriting()
    order = presentation.precedence()
    from tfl.srs import shortlex_key

    for rule in system.rules:
        assert shortlex_key(rule.lhs, order) > shortlex_key(rule.rhs, order), rule


def test_a_completed_presentation_decides_the_word_problem():
    """`⟨a, b | a² = 1, b² = 1⟩` — бесконечная диэдральная группа."""
    presentation = parse_presentation("группа: a, b\naa = 1\nbb = 1")
    _, verdict = presentation.complete()
    assert verdict.value is True
    assert presentation.equal("aa", "").value is True
    assert presentation.equal("ab", "ba").value is False


def test_commuting_generators_make_the_whole_group_commutative():
    presentation = parse_presentation("группа: a, b\nab = ba")
    assert presentation.is_commutative().value is True


def test_a_non_commutative_group_is_refuted_with_a_pair():
    presentation = parse_presentation("группа: a, b\naa = 1\nbb = 1")
    verdict = presentation.is_commutative()
    assert verdict.value is False
    assert verdict.witness == ("a", "b")


# --------------------------------------------------------------------------
# Равенство доказывается без конфлюэнтности
# --------------------------------------------------------------------------


def test_a_group_is_proved_trivial_even_though_completion_diverges():
    """`⟨a, b | aba⁻¹ = b², bab⁻¹ = a²⟩` — классический пример: группа тривиальна.

    Пополнение здесь **расходится**, и это не мешает: совпадение нормальных
    форм — уже готовая цепочка соотношений, применённых в обе стороны,
    и конфлюэнтность для такого вывода не нужна.
    """
    presentation = parse_presentation("группа: a, b\naba^-1 = bb\nbab^-1 = aa")
    _, verdict = presentation.complete()
    assert verdict.value is None
    assert "неразрешима" in verdict.reason
    trivial = presentation.is_trivial()
    assert trivial.value is True


def test_a_difference_without_a_complete_system_is_not_a_verdict():
    """Полугруппа Цейтина: пополнение не сходится, и «не равны» сказать нельзя."""
    presentation = parse_presentation(
        "полугруппа: a, b, c, d, e\n"
        "ac = ca\nad = da\nbc = cb\nbd = db\n"
        "eca = ce\nedb = de\ncca = ccae",
        SEMIGROUP,
    )
    assert presentation.kind == SEMIGROUP
    verdict = presentation.equal("eca", "edb", max_len=10)
    assert verdict.value is not False
