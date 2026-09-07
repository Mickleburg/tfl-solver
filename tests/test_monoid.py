"""Трансформационный и синтаксический моноиды.

Приём взят из работы, которую преподаватель рекомендовала в issue #6
(`vendor/UsefulTornado-Formal-Languages/transition_monoid`), и переписан
с одной существенной поправкой: там моноид считается по полезным
состояниям, из-за чего функции становятся частичными. Синтаксическим
такой моноид не является — см. `test_trap_state_changes_the_monoid`.
"""

from __future__ import annotations

from tfl.automata import DFA, dfa_of
from tfl.monoid import syntactic_monoid, transition_monoid
from tfl.srs import parse_srs
from tfl.words import iter_words


def test_parity_monoid_is_the_two_element_group():
    """`(aa)*` — классический моноид из двух элементов: тождество и обмен."""
    monoid = transition_monoid(dfa_of("(aa)*"))
    assert len(monoid) == 2
    assert [element.word for element in monoid] == ["", "a"]
    assert monoid.elements[0].is_identity()
    assert monoid.reduce("aaaa") == ""
    assert monoid.reduce("aaa") == "a"


def test_single_b_language_monoid():
    """`a*ba*` — три класса: ни одной `b`, ровно одна, две и больше."""
    monoid = syntactic_monoid(dfa_of("a*ba*"))
    assert [element.word or "ε" for element in monoid] == ["ε", "b", "bb"]
    assert len(monoid.accepting()) == 1
    assert monoid.accepting()[0].word == "b"


def test_monoid_recognises_the_same_language_as_the_automaton():
    """Распознавание моноидом сверено с автоматом на всех словах до длины 8."""
    for pattern in ("(aa)*", "a*ba*", "(a|b)*ab", "a(a|b)*b|b"):
        machine = dfa_of(pattern)
        monoid = syntactic_monoid(machine)
        for word in iter_words("ab", 8):
            assert monoid.accepts(word) == machine.accepts(word), (pattern, word)


def test_rewriting_rules_are_a_presentation_of_the_monoid():
    """Каждое правило `w → r` соединяет слова с одной функцией переходов.

    Это и есть SRS, которую просят в ЛР1: правила переписывают слово
    к неприводимому представителю его класса.
    """
    monoid = syntactic_monoid(dfa_of("(a|b)*ab"))
    assert monoid.rules
    for lhs, rhs in monoid.rules:
        assert monoid.image_of(lhs) == monoid.image_of(rhs)
        assert len(rhs) <= len(lhs)
    # правила разбираются нашим же парсером SRS
    system = parse_srs(monoid.srs())
    assert len(system.rules) == len(monoid.rules)


def test_representatives_are_irreducible():
    """Представитель не содержит левой части ни одного правила."""
    monoid = syntactic_monoid(dfa_of("a*ba*"))
    lefts = [lhs for lhs, _ in monoid.rules]
    for element in monoid.elements:
        assert not any(lhs in element.word for lhs in lefts if lhs)


def test_trap_state_changes_the_monoid():
    """Без ловушки моноид другой — ровно та ошибка, что в чужой работе.

    У `a*ba*` слова с двумя `b` уходят в ловушку. Если её выбросить,
    класс `bb` исчезает и моноид перестаёт быть синтаксическим.
    """
    machine = dfa_of("a*ba*")
    with_trap = syntactic_monoid(machine)
    without = transition_monoid(machine.trim(), complete=False)
    assert len(with_trap) == 3
    assert len(without) < len(with_trap)


def test_aperiodicity_matches_star_freeness():
    """Теорема Шютценберже: апериодичность ⟺ бесзвёздочность.

    `a*ba*` бесзвёздочен, чётность длины — нет, и это самый известный
    пример языка, не выражаемого без итерации.
    """
    assert syntactic_monoid(dfa_of("a*ba*")).is_aperiodic() is True
    assert syntactic_monoid(dfa_of("(aa)*")).is_aperiodic() is False


def test_monoid_of_a_one_state_automaton():
    """Автомат, принимающий всё: моноид тривиален."""
    machine = DFA(frozenset("ab"), 0, frozenset({0}), {(0, "a"): 0, (0, "b"): 0})
    monoid = transition_monoid(machine)
    assert len(monoid) == 1
    assert monoid.elements[0].is_identity()
    assert monoid.reduce("abba") == ""


def test_markdown_table_lists_every_element():
    monoid = syntactic_monoid(dfa_of("a*ba*"))
    table = monoid.markdown()
    assert table.count("\n") == len(monoid) + 1  # заголовок и разделитель
    assert "Принимающий" in table
    assert "3 элементов" in monoid.summary()
