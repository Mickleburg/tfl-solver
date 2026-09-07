"""Системы переписывания термов: унификация, переписывание, интерпретации.

Приёмочные примеры: разобранная преподавателем унификация из issue #2
и задачи «Аптеки» из раздела с TRS.
"""

from __future__ import annotations

import pytest

from tfl.trs import (
    Poly,
    TRS,
    Rule,
    app,
    match,
    parse_interpretation,
    parse_term,
    parse_trs,
    substitute,
    unify,
    var,
)
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Переменные объявляются, а не угадываются
# --------------------------------------------------------------------------


def test_two_opposite_conventions_live_in_the_corpus():
    """В issue #2 переменные строчные, в «Аптеке» — заглавные.

    Один и тот же текст читается по-разному, и разница не косметическая:
    от неё зависит, что вообще является правилом.
    """
    lowercase = parse_term("F(q, q, q)", {"q"})
    assert lowercase.variables() == {"q"}
    assert lowercase.symbols() == {"F": 3}

    uppercase = parse_term("w(w(s, X), Y)", {"X", "Y"})
    assert uppercase.variables() == {"X", "Y"}
    assert uppercase.symbols() == {"w": 2, "s": 0}


def test_variable_cannot_take_arguments():
    with pytest.raises(ValueError, match="не может иметь аргументов"):
        parse_term("x(a)", {"x"})


def test_numeric_constants_are_symbols():
    """`plus(0, y) → y` — обычная запись, и `0` тут константа."""
    term = parse_term("plus(0, y)", {"y"})
    assert term.symbols() == {"plus": 2, "0": 0}


# --------------------------------------------------------------------------
# Унификация: разобранный пример преподавателя
# --------------------------------------------------------------------------


def test_unification_reproduces_the_teachers_example():
    """Пара `F(q,q,q)` и `F(F(x,y,R), F(a,w,a), F(w,x,y))` из issue #2.

    Ответ преподавателя записан мультиуравнениями: `{q} := F(x,y,a)`,
    `{x,a,w,y} = R`. В обычной записи это `q ↦ F(R,R,R)`, а `x`, `y`,
    `a`, `w` — все `R`.
    """
    names = {"q", "x", "y", "a", "w"}
    left = parse_term("F(q,q,q)", names)
    right = parse_term("F(F(x,y,R), F(a,w,a), F(w,x,y))", names)

    mgu = unify(left, right)
    assert mgu is not None
    assert str(mgu["q"]) == "F(R, R, R)"
    assert {str(mgu[n]) for n in ("x", "y", "a", "w")} == {"R"}
    assert substitute(left, mgu) == substitute(right, mgu)


def test_occurs_check():
    """Без проверки вхождения `x` и `F(x)` «унифицируются» в бесконечный терм."""
    assert unify(parse_term("x", {"x"}), parse_term("F(x)", {"x"})) is None


def test_unification_fails_on_different_heads():
    assert unify(parse_term("F(a)", set()), parse_term("G(a)", set())) is None


def test_matching_is_one_way():
    """Сопоставление подставляет только в образец — терм не трогается.

    Унификация здесь была бы ошибкой: она позволила бы «доопределить»
    переписываемый терм и применить правило там, где оно не применимо.
    """
    pattern = parse_term("f(a, x)", {"x"})
    assert match(pattern, parse_term("f(a, b)", set())) == {"x": parse_term("b", set())}
    assert match(pattern, parse_term("f(y, b)", {"y"})) is None
    assert unify(pattern, parse_term("f(y, b)", {"x", "y"})) is not None


def test_nonlinear_pattern_requires_equal_subterms():
    """`f(x, x)` подходит только к терму с двумя одинаковыми аргументами."""
    pattern = parse_term("f(x, x)", {"x"})
    assert match(pattern, parse_term("f(a, a)", set())) is not None
    assert match(pattern, parse_term("f(a, b)", set())) is None


# --------------------------------------------------------------------------
# Правила и переписывание
# --------------------------------------------------------------------------


def test_free_variables_on_the_right_are_rejected():
    """`f(x) → g(y)` не задаёт переписывания: что подставлять вместо `y`?"""
    with pytest.raises(ValueError, match="свободные переменные"):
        parse_trs("variables = [x, y]\nf(x) -> g(y)")


def test_rewriting_applies_at_every_position():
    system = parse_trs("variables = [x]\nf(x) -> g(x)")
    term = parse_term("f(f(a))", set())
    got = {str(t) for t in system.step(term)}
    assert got == {"g(f(a))", "f(g(a))"}


def test_normal_form():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    term = parse_term("plus(s(s(0)), s(0))", set())
    assert str(system.normal_form(term)) == "s(s(s(0)))"


# --------------------------------------------------------------------------
# Завершимость
# --------------------------------------------------------------------------


def test_loop_is_found_when_the_term_grows():
    """`f(x) → f(f(x))` не возвращает терм к себе — растёт, но содержит его."""
    verdict = parse_trs("variables = [x]\nf(x) -> f(f(x))").terminates()
    assert verdict.value is False
    assert "найдена петля" in verdict.reason


def test_cycle_is_called_a_cycle():
    verdict = parse_trs("variables = []\na -> a").terminates()
    assert verdict.value is False
    assert "найден цикл" in verdict.reason


def test_peano_addition_terminates_by_interpretation():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    verdict = system.terminates()
    assert verdict.value is True
    assert "интерпретация" in verdict.reason


def test_a_supplied_interpretation_can_be_checked():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    good = parse_interpretation("plus(x, y) = 2*x + y + 1; s(x) = x + 1; 0 = 1")
    assert system.check_interpretation(good).value is True

    bad = parse_interpretation("plus(x, y) = x + y; s(x) = x + 1; 0 = 1")
    verdict = system.check_interpretation(bad)
    assert verdict.value is False
    assert "не убывает" in verdict.reason


def test_question_bank_interpretation_does_not_decrease():
    """Банк вопросов 2024: `f(x) => g(x,x)` при `f(x) := x+1; g(x,y) := x + y*2 + 1`.

    Интерпретация в том виде, в каком она выписана в банке, правило
    **не уменьшает**: `[f(x)] = x + 1`, а `[g(x,x)] = 3x + 1`. Сама система
    при этом завершима — правая часть не содержит `f`, — и оракул находит
    для неё другую интерпретацию.
    """
    system = parse_trs("variables = [x]\nf(x) -> g(x, x)")
    quoted = parse_interpretation("f(x) := x+1; g(x,y) := x + y*2 + 1")
    verdict = system.check_interpretation(quoted)
    assert verdict.value is False
    assert "[f(x)] = x + 1" in verdict.reason

    assert system.terminates().value is True


# --------------------------------------------------------------------------
# Полиномы
# --------------------------------------------------------------------------


def test_polynomial_arithmetic():
    x, y = Poly.of("x"), Poly.of("y")
    assert str(x * y + x + Poly.number(2)) == "x*y + x + 2"
    assert str((x + Poly.number(1)) * (x + Poly.number(1))) == "x^2 + 2*x + 1"
    assert (x * y).constant() == 0


def test_domination_needs_a_positive_constant_and_no_negative_coefficients():
    x = Poly.of("x")
    assert (x + Poly.number(2)).dominates(x + Poly.number(1))
    assert not (x + Poly.number(2)).dominates(x + Poly.number(2))
    assert not (x + Poly.number(5)).dominates(x * Poly.number(2))


# --------------------------------------------------------------------------
# Мост в строковые системы
# --------------------------------------------------------------------------


def test_unary_system_becomes_a_string_system():
    """Все символы одноместные — значит это переписывание строк.

    Так устроена задача F6 «Аптеки» (2022, 2023 и 2024 годов): семь правил
    над `E`, `Q`, `q`, `W`, все унарные, и вся строковая машинерия
    к ней применима без изменений.
    """
    system = parse_trs("variables = [X]\nE(Q(q(X))) -> Q(W(X))\nq(E(X)) -> W(X)")
    assert system.is_unary()
    strings = system.as_srs()
    assert [str(rule) for rule in strings.rules] == ["EQq → QW", "qE → W"]


def test_bridge_preserves_rewriting():
    """Шаг в термах и шаг в строках дают одно и то же — это и проверяется."""
    system = parse_trs("variables = [X]\nE(Q(X)) -> Q(E(E(X)))\nq(E(X)) -> W(X)")
    strings = system.as_srs()

    def to_term(word: str):
        term = var("X")
        for letter in reversed(word):
            term = app(letter, term)
        return term

    for word in iter_words("EQqW", 5):
        by_terms = {str(t) for t in system.step(to_term(word))}
        by_strings = {str(to_term(w)) for w in strings.step(word)}
        assert by_terms == by_strings, word


def test_binary_system_has_no_bridge():
    system = parse_trs("variables = [x, y]\nf(x, y) -> f(y, x)")
    assert not system.is_unary()
    with pytest.raises(ValueError, match="только для унарных"):
        system.as_srs()
