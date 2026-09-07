"""Переписывания регулярных выражений по тождествам Конвея и Кроба.

Термин «переписывания по Конвею–Кробу» стоит в разбалловке РК1 2023
(issue #21) и больше нигде в корпусе не встречается. Здесь реализовано
то, что за ним стоит в литературе, и тесты фиксируют главное: каждый шаг
сохраняет язык (это проверяет оракул эквивалентности), а групповые
тождества делают то, чего классические не могут.
"""

from __future__ import annotations

import pytest

from tfl import regex as rx
from tfl.automata import dfa_of, equivalent
from tfl.conway import (
    IDENTITIES,
    cyclic_collapse,
    cyclic_expansions,
    from_term,
    is_minimal,
    rewritings,
    shorten,
    to_term,
)

# --------------------------------------------------------------------------
# Перевод регулярка ↔ терм
# --------------------------------------------------------------------------


def test_round_trip_through_terms():
    for text in ("(a|b)*", "a*b", "((ab)*|c)d", "ε", "∅", "a"):
        node = rx.parse(text)
        assert from_term(to_term(node)) == node


def test_n_ary_nodes_are_folded_to_binary():
    """Тождества двуместные, а `Alt` и `Cat` — многоместные."""
    term = to_term(rx.parse("abc"))
    assert term.arity == 2 and term.args[1].arity == 2


# --------------------------------------------------------------------------
# Тождества сохраняют язык — главная проверка
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["(a|b)*", "a*b*", "(ab)*", "a*a*", "(ε|a)*", "a(b|c)", "(a|b)c"]
)
def test_every_single_step_preserves_the_language(text):
    """Тождества записаны руками, и ошибка в любом иначе прошла бы тихо."""
    node = rx.parse(text)
    letters = "abc"
    reference = dfa_of(node, letters)
    steps = rewritings(node)
    assert steps, f"из «{text}» не нашлось ни одного переписывания"
    for name, other in steps:
        assert equivalent(reference, dfa_of(other, letters)), (name, str(other))


def test_the_identity_list_is_not_accidentally_empty():
    assert len(IDENTITIES) >= 8


# --------------------------------------------------------------------------
# Сокращение
# --------------------------------------------------------------------------


def test_the_sum_star_identity_shortens_the_classic_example():
    """`(a*b)*a* = (a|b)*` — тождество Конвея о звёздочке суммы."""
    verdict = is_minimal("(a*b)*a*", max_size=12)
    assert verdict.value is False
    assert str(verdict.witness.best) == "(a|b)*"


def test_unrolling_is_undone():
    for text, expected in (("ε|aa*", "a*"), ("a*a*", "a*"), ("(ε|a)*", "a*")):
        found = shorten(text, max_size=12)
        assert str(found.best) == expected, text


def test_a_shortening_comes_with_the_derivation():
    found = shorten("(a*b)*a*", max_size=12)
    printed = found.markdown()
    assert "(a*b)*a*" in printed and "(a|b)*" in printed
    assert "звёздочка суммы" in printed


def test_no_shortening_is_never_reported_as_minimality():
    """`True` среди исходов нет намеренно: система тождеств бесконечна."""
    verdict = is_minimal("(a|b)*", max_size=12)
    assert verdict.value is None
    assert "Минимальностью это не является" in verdict.reason


def test_the_bounds_of_the_search_are_stated():
    verdict = is_minimal("(a|b)*", max_size=12, max_nodes=5)
    assert verdict.value is None
    assert "обход обрезан" in verdict.reason


# --------------------------------------------------------------------------
# Групповые тождества — то, чего классической системе не хватает
# --------------------------------------------------------------------------


def test_cyclic_collapse_recognises_the_residue_split():
    assert str(cyclic_collapse(rx.parse("(aa)*|a(aa)*"))) == "a*"
    assert str(cyclic_collapse(rx.parse("(aaa)*|a(aaa)*|aa(aaa)*"))) == "a*"


def test_cyclic_collapse_refuses_a_broken_split():
    """Лишнее слагаемое — и это уже не разбиение итерации по остаткам."""
    assert cyclic_collapse(rx.parse("(aa)*|a(aa)*|b")) is None
    assert cyclic_collapse(rx.parse("(aa)*|aa(aa)*")) is None
    assert cyclic_collapse(rx.parse("a*")) is None


def test_cyclic_expansions_preserve_the_language():
    node = rx.parse("a*")
    reference = dfa_of(node, "a")
    grown = cyclic_expansions(node)
    assert grown
    for other in grown:
        assert equivalent(reference, dfa_of(other, "a"))


def test_the_classical_identities_alone_do_not_reach_it():
    """Вот ради чего Кроб добавил групповые аксиомы.

    `(aa)*|a(aa)*` — это `a*`, но классическая система Конвея к нему
    не приводит: обход **всех** выражений размера до 22 (1781 штука)
    ничего короче не находит. С тождеством для циклической группы
    порядка 2 ответ получается сразу.
    """
    without = is_minimal("(aa)*|a(aa)*", max_size=22, max_nodes=6000, groups=False)
    assert without.value is None
    assert without.witness.exhausted

    with_groups = is_minimal("(aa)*|a(aa)*", max_size=14, groups=True)
    assert with_groups.value is False
    assert str(with_groups.witness.best) == "a*"


def test_the_group_identity_works_for_order_three_too():
    verdict = is_minimal("(aaa)*|a(aaa)*|aa(aaa)*", max_size=20, max_nodes=6000)
    assert verdict.value is False
    assert str(verdict.witness.best) == "a*"


def test_normalising_constructors_are_why_matching_is_structural():
    """`ε·r` не существует как объект, и путь через дистрибутивность обрывается.

    Именно поэтому групповое тождество распознаётся образцом целиком,
    а не выводится из `(Rⁿ)*(ε|R) = R*` вынесением общего множителя.
    """
    assert rx.cat(rx.EPS, rx.parse("a*")) == rx.parse("a*")
