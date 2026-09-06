"""Тесты регулярных аппроксимаций сверху и пересечения по Бар-Хиллелу.

Контрольная точка взята из лекции 9: аппроксимацией языка `(ⁿ a )ⁿ`,
построенной по LR(0)-автомату (Перейра–Райт), является язык `(* a )*`.
Скобки здесь переименованы в `c` и `d`, потому что в академических
регулярных выражениях круглые скобки — метасимволы.

Второе, что проверяется на всех 14 грамматиках ЛР3: аппроксимация —
надмножество, а пересечение с ней язык не меняет. Первое — свойство
конструкции (нарушение означает ошибку), второе — то самое
«автоматическое тестирование предполагаемой эквивалентности», которого
требует задание.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.approx import (
    intersect,
    ll1_automaton,
    lr0_automaton,
    over_approximates,
    surplus,
    to_dfa,
)
from tfl.automata import counterexample, dfa_of
from tfl.cfg import parse_cfg
from tfl.parse import equivalent_up_to, language

LAB3 = pathlib.Path(__file__).parent.parent / "evals" / "lab3_2025"

# Язык cⁿ a dⁿ — это `(ⁿ a )ⁿ` из лекции с переименованными скобками.
NESTED = parse_cfg("S -> c S d | a")
BUILDERS = [("lr0", lr0_automaton), ("ll1", ll1_automaton)]


def lab3_grammars():
    return [
        (int(path.stem.split("-")[1]), parse_cfg(path.read_text(encoding="utf-8")))
        for path in sorted(LAB3.glob("variant-*.cfg"))
    ]


GRAMMARS = lab3_grammars()
IDS = [f"v{n}" for n, _ in GRAMMARS]


# --------------------------------------------------------------------------
# Контрольная точка из лекции
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,build", BUILDERS)
def test_nested_language_approximates_to_star(name, build):
    """`cⁿ a dⁿ` аппроксимируется языком `c* a d*` — пример из лекции 9."""
    got = to_dfa(build(NESTED)).complete()
    want = dfa_of("c*ad*", alphabet="cda").complete()
    assert counterexample(got, want) is None


@pytest.mark.parametrize("name,build", BUILDERS)
def test_approximation_is_strictly_coarser(name, build):
    """Скобки перестают считаться — это и есть потеря точности."""
    dfa = to_dfa(build(NESTED))
    assert dfa.accepts("ccad")
    assert "ccad" not in language(NESTED, 5)


@pytest.mark.parametrize("name,build", BUILDERS)
def test_intersection_restores_the_language(name, build):
    dfa = to_dfa(build(NESTED))
    assert equivalent_up_to(NESTED, intersect(NESTED, dfa), 7) == ([], [])


def test_triple_nonterminals_look_like_in_the_reports():
    grammar = intersect(NESTED, to_dfa(lr0_automaton(NESTED)))
    assert any(name.startswith("⟨") for name in grammar.nonterminals)


# --------------------------------------------------------------------------
# Свойства конструкции
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,build", BUILDERS)
@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_approximation_covers_the_language(number, grammar, name, build):
    """Аппроксимация сверху обязана принимать всё, что порождает грамматика.

    Непустой список — ошибка построения автомата, и её надо ловить до
    пересечения: пересечение с «дырявой» аппроксимацией молча отрежет
    часть языка, и отчёт получится правдоподобным, но неверным.
    """
    missed = over_approximates(grammar, to_dfa(build(grammar)), 7)
    assert not missed, f"вариант {number}, {name}: не покрыты {missed[:5]}"


@pytest.mark.parametrize("name,build", BUILDERS)
@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_intersection_preserves_the_language(number, grammar, name, build):
    lost, gained = equivalent_up_to(grammar, intersect(grammar, to_dfa(build(grammar))), 7)
    assert not lost, f"вариант {number}, {name}: пересечение потеряло {lost[:5]}"
    assert not gained, f"вариант {number}, {name}: пересечение добавило {gained[:5]}"


def test_empty_language_intersects_to_empty():
    grammar = parse_cfg("S -> S a")
    assert language(intersect(grammar, to_dfa(lr0_automaton(grammar))), 4) == set()


def test_nullable_start_survives_intersection():
    """ε легко теряется при пересечении: путь из старта в финал пустой."""
    grammar = parse_cfg("S -> a S b | ε")
    assert "" in language(intersect(grammar, to_dfa(lr0_automaton(grammar))), 4)


def test_intersection_with_a_narrower_automaton_cuts_the_language():
    """Пересечение не обязано сохранять язык — только с аппроксимацией сверху."""
    grammar = parse_cfg("S -> a S b | ε")
    narrow = dfa_of("(ab)*", alphabet="ab")
    assert language(intersect(grammar, narrow), 6) == {"", "ab"}


def test_surplus_shows_what_the_automaton_stopped_distinguishing():
    extra = surplus(to_dfa(lr0_automaton(NESTED)), NESTED, "cda", 4, 3)
    assert "ad" in extra or "ca" in extra
