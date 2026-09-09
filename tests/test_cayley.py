"""Граф Кэли и действие группы: два разных источника вывода.

Граф Кэли строится переписыванием и отвечает на всё сразу, но **только
для конечной группы** и только когда пополнение сошлось. Действие
проверяется перебором точек и умеет ровно одно, зато всегда: доказывать
**неравенство**. Оба вывода здесь сверяются друг с другом.

Отдельно закреплено следствие, ради которого граф Кэли и строится как
автомат: число классов Майхилла–Нероуда языка проблемы равенства равно
порядку группы. Это лёгкая половина теоремы Анисимова, и она считается,
а не проговаривается.
"""

from __future__ import annotations

import pytest

from tfl.cayley import (
    Action,
    cayley_graph,
    irreducible_automaton,
    is_finite,
    normalize,
)
from tfl.presentation import SEMIGROUP, parse_presentation

CYCLIC = "группа: a\naaa = 1"
KLEIN = "группа: a, b\naa = 1\nbb = 1\nab = ba"
SYM3 = "группа: a, b\naa = 1\nbb = 1\nababab = 1"
DIHEDRAL = "группа: a, b\naa = 1\nbb = 1"


def graph(text: str, limit: int = 200):
    verdict = cayley_graph(parse_presentation(text), limit)
    assert verdict.value is True, verdict.reason
    return verdict.witness


# --------------------------------------------------------------------------
# Граф Кэли
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,order", [(CYCLIC, 3), (KLEIN, 4), (SYM3, 6)]
)
def test_the_walk_closes_and_counts_the_group(text, order):
    assert graph(text).order == order


def test_the_number_of_classes_equals_the_order_of_the_group():
    """Лёгкая половина теоремы Анисимова, посчитанная.

    Вычет языка проблемы равенства по слову `u` — это множество слов,
    равных `u⁻¹`. Значит вычеты и элементы группы — одно и то же,
    и минимальный автомат обязан иметь ровно `|G|` классов.
    """
    for text in (CYCLIC, KLEIN, SYM3):
        found = graph(text)
        assert found.dfa().class_count() == found.order, text


def test_the_word_problem_automaton_accepts_exactly_the_unit():
    found = graph(SYM3)
    machine = found.dfa()
    assert machine.accepts("")
    assert machine.accepts("aa")
    assert machine.accepts("ababab")
    assert not machine.accepts("ab")
    assert not machine.accepts("a")


def test_multiplication_agrees_with_the_edges():
    found = graph(SYM3)
    for element in found.elements:
        for letter in found.presentation.alphabet:
            assert found.multiply(element, letter) == found.edges[(element, letter)]


def test_commutativity_from_the_table_agrees_with_rewriting():
    """Две независимые дороги к одному ответу: таблица и пополнение."""
    for text in (CYCLIC, KLEIN, SYM3):
        presentation = parse_presentation(text)
        assert (
            graph(text).is_commutative().value
            is presentation.is_commutative().value
        ), text


def test_an_infinite_group_is_refuted_and_not_left_unknown():
    """Обход упирается в потолок — и вопрос передаётся разрешимой проверке.

    Гадать про бесконечность не нужно: у пополненной системы элементы
    моноида это неприводимые слова, а конечность регулярного языка
    разрешима.
    """
    verdict = cayley_graph(parse_presentation(DIHEDRAL), 60)
    assert verdict.value is False
    assert "бесконечен" in verdict.reason


def test_finiteness_is_decided_both_ways():
    for text, expected in ((CYCLIC, True), (KLEIN, True), (SYM3, True), (DIHEDRAL, False)):
        assert is_finite(parse_presentation(text)).value is expected, text


def test_the_count_of_irreducible_words_is_the_order():
    for text in (CYCLIC, KLEIN, SYM3):
        assert is_finite(parse_presentation(text)).witness == graph(text).order, text


def test_the_free_group_is_infinite():
    """Без соотношений неприводимы все приведённые слова, и их бесконечно много."""
    verdict = is_finite(parse_presentation("группа: a, b\na = a"))
    assert verdict.value is False
    assert "Анисимова" in verdict.reason


def test_a_diverging_completion_leaves_finiteness_unknown():
    verdict = is_finite(
        parse_presentation("группа: a, b\naba^-1 = bb\nbab^-1 = aa")
    )
    assert verdict.value is None


def test_the_irreducible_automaton_rejects_exactly_the_reducible_words():
    from tfl.srs import parse_srs

    system = parse_srs("aba -> b\nbb -> a")
    machine = irreducible_automaton(system, "ab")
    for word in ("", "a", "ab", "abb", "baab"):
        assert machine.accepts(word) is (
            "aba" not in word and "bb" not in word
        ), word
    assert not machine.accepts("aba")
    assert not machine.accepts("abba")


def test_a_diverging_completion_blocks_the_walk():
    """Без полной системы нормальная форма не единственна — строить нельзя."""
    verdict = cayley_graph(parse_presentation("группа: a, b\naba^-1 = bb\nbab^-1 = aa"))
    assert verdict.value is None
    assert "Без полной системы" in verdict.reason


def test_normalisation_is_idempotent():
    found = graph(SYM3)
    for element in found.elements:
        assert normalize(found.system, element) == element


def test_the_table_can_be_printed():
    text = graph(KLEIN).markdown()
    assert "4** элемент" in text or "**4**" in text
    assert "| элемент |" in text


# --------------------------------------------------------------------------
# Действие
# --------------------------------------------------------------------------


def symmetric_action() -> Action:
    """`S₃` на трёх точках: `a` меняет 0 и 1, `b` меняет 1 и 2."""
    return Action(3, {"a": (1, 0, 2), "b": (0, 2, 1)})


def test_an_action_must_send_generators_to_permutations():
    with pytest.raises(ValueError, match="не перестановка"):
        Action(3, {"a": (0, 0, 2)})
    with pytest.raises(ValueError, match="не 3 точек"):
        Action(3, {"a": (1, 0)})


def test_the_inverse_letter_acts_by_the_inverse_permutation():
    action = Action(3, {"a": (1, 2, 0)})
    assert action.apply("aA", 0) == 0
    assert action.apply("Aa", 2) == 2
    assert action.of("A") == (2, 0, 1)


def test_a_real_action_satisfies_every_relation():
    assert symmetric_action().check(parse_presentation(SYM3)).value is True


def test_a_broken_relation_is_refuted_with_the_point():
    """`a` и `b` не коммутируют, значит клейновой четвёркой это не является."""
    verdict = symmetric_action().check(parse_presentation(KLEIN))
    assert verdict.value is False
    assert "нарушено" in verdict.reason


def test_a_missing_generator_is_named():
    verdict = Action(3, {"a": (1, 0, 2)}).check(parse_presentation(SYM3))
    assert verdict.value is None
    assert "не заданы образы" in verdict.reason


def test_an_action_proves_inequality_where_rewriting_needs_completeness():
    """Единственный здесь способ доказать «не равны» без полной системы."""
    action = symmetric_action()
    assert action.check(parse_presentation(SYM3)).value is True
    verdict = action.separates("ab", "ba")
    assert verdict.value is False
    assert graph(SYM3).multiply("ab", "") != graph(SYM3).multiply("ba", "")


def test_an_action_that_does_not_separate_says_nothing():
    verdict = symmetric_action().separates("aa", "")
    assert verdict.value is None
    assert "не следует ничего" in verdict.reason


def test_the_orbit_is_computed_through_inverses_too():
    action = Action(4, {"a": (1, 2, 3, 0)})
    assert action.orbit(0) == frozenset({0, 1, 2, 3})
    assert Action(3, {"a": (1, 0, 2)}).orbit(2) == frozenset({2})


def test_for_a_semigroup_the_walk_builds_the_monoid_and_says_so():
    """Пустое слово в обходе есть, значит получается `A*/≡`, а не `A⁺/≡`.

    У `⟨a | a² = a⟩` сама полугруппа одноэлементна, а моноид с приписанной
    снаружи единицей — двухэлементен. Разницу вердикт называет вслух.
    """
    presentation = parse_presentation("полугруппа: a\naa = a", SEMIGROUP)
    verdict = cayley_graph(presentation, 20)
    assert verdict.value is True
    assert verdict.witness.order == 2
    assert verdict.witness.elements == ("", "a")
    assert "моноид" in verdict.reason


def test_the_class_count_identity_is_about_groups_and_not_monoids():
    """Ловушка: у моноида без обратных вычеты и элементы — разные вещи.

    У `⟨a, b | a² = a, b² = b, ab = ba⟩` элементов четыре (`ε`, `a`, `b`,
    `ab`), а классов Майхилла–Нероуда языка «слово равно единице» всего
    два: у необратимых элементов вычет пуст, и все они склеиваются.
    Совпадение чисел — утверждение про группу, а не про моноид.
    """
    presentation = parse_presentation(
        "полугруппа: a, b\naa = a\nbb = b\nab = ba", SEMIGROUP
    )
    verdict = cayley_graph(presentation, 50)
    assert verdict.value is True
    found = verdict.witness
    assert found.order == 4
    assert found.dfa().class_count() == 2
