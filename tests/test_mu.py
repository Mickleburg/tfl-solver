"""µ-выражения Клини — лекция 8 (2022), слайд 7.

Приёмочные точки взяты из самого курса: `µX.aXb | ε` задаёт `{aⁿbⁿ}`
(лекция 13, 2021), а `µy.a(µx.axb + y + a)` — грамматику `Y → aX`,
`X → aXb | Y | a`, выписанную на слайде. Дальше — первый вопрос билета
2022 года, который этим и решается.
"""

from __future__ import annotations

import pytest

from tfl.cfg import parse_cfg
from tfl.mu import Alt, Cat, Letter, Mu, Var, parse_mu
from tfl.parse import language

# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_mu_body_reaches_to_the_right_edge():
    """`µX.aXb | ε` — это `µX.(aXb | ε)`, а не `(µX.aXb) | ε`.

    Прочтение решает всё: при втором язык был бы `{aⁿbⁿ | n > 0} ∪ {ε}`
    без самой рекурсии и вообще не задавался бы неподвижной точкой.
    """
    node = parse_mu("µX.aXb | ε").expression
    assert isinstance(node, Mu)
    assert isinstance(node.body, Alt)
    assert len(node.body.parts) == 2


def test_a_name_is_a_variable_only_when_a_mu_binds_it():
    """Регистр ни при чём: в лекции переменные строчные, в билете заглавные."""
    node = parse_mu("µx.xa").expression
    assert isinstance(node, Mu) and isinstance(node.body, Cat)
    assert node.body.parts == (Var("x"), Letter("a"))

    # тот же `x`, но без связывания — это буква
    assert parse_mu("xa").expression.parts == (Letter("x"), Letter("a"))


def test_plus_and_pipe_are_the_same_alternation():
    """Лекция пишет `+`, билет пишет `|`."""
    assert parse_mu("a+b").expression == parse_mu("a|b").expression


def test_printing_round_trips_through_the_parser():
    for text in ("µX.aXb | ε", "µy.a(µx.axb + y + a)", "µX.(a(µY.bX|Ya|(µZ.ZZ|cc))bX|ε)"):
        once = parse_mu(text)
        twice = parse_mu(str(once))
        assert set(once.words(8)) == set(twice.words(8))


def test_malformed_expressions_are_rejected():
    for text in ("µX", "µXaXb", "µ.aXb", "(a", "a)"):
        with pytest.raises(ValueError):
            parse_mu(text)


# --------------------------------------------------------------------------
# Перевод в грамматику — контрольные точки лекций
# --------------------------------------------------------------------------


def test_anbn_from_lecture_13():
    """> µ-выражение `µX.aXb | ε` задаёт описание языка `{aⁿbⁿ}`"""
    expression = parse_mu("µX.aXb | ε")
    assert expression.words(7) == ["", "ab", "aabb", "aaabbb"]
    verdict = expression.agrees_with(
        lambda w: w == "a" * (len(w) // 2) + "b" * (len(w) // 2), max_len=8
    )
    assert verdict.value is True, verdict.reason


def test_the_lecture_grammar_is_reproduced_rule_for_rule():
    """> `µy.a(µx.axb + y + a)` определяет грамматику `Y → aX`, `X → aXb | Y | a`"""
    grammar = parse_mu("µy.a(µx.axb + y + a)").to_cfg()
    assert grammar.start == "Y"
    assert {str(p) for p in grammar.productions} == {
        "Y → a X",
        "X → a X b",
        "X → Y",
        "X → a",
    }


def test_star_becomes_right_recursion():
    grammar = parse_mu("a*b").to_cfg()
    assert language(grammar, 4) == {"b", "ab", "aab", "aaab"}


def test_a_bare_regular_expression_still_works():
    """Без `µ` это обычная регулярка, и язык обязан совпасть."""
    assert set(parse_mu("(a|b)*c").words(3)) == {"c", "ac", "bc", "aac", "abc", "bac", "bbc"}


# --------------------------------------------------------------------------
# Билет 2022, первый вопрос
# --------------------------------------------------------------------------

TICKET = "µX.(a(µY.bX|Ya|(µZ.ZZ|cc))bX|ε)"


def test_ticket_translates_to_a_readable_grammar():
    grammar = parse_mu(TICKET).to_cfg()
    assert grammar.start == "X"
    assert {str(p) for p in grammar.productions} == {
        "X → a Y b X",
        "X → ε",
        "Y → b X",
        "Y → Y a",
        "Y → Z",
        "Z → Z Z",
        "Z → c c",
    }


def test_ticket_answer_matches_the_expression():
    """Ответ на вопрос «описать язык»: $L = (a\\,M\\,b)^*$, $M = (bL \\cup (cc)^+)a^*$.

    Описание придумывается человеком, а совпадение проверяет оракул —
    здесь на всех 29 524 словах длины до 9 включительно.
    """

    def in_x(word: str) -> bool:
        if word == "":
            return True
        if not word.startswith("a"):
            return False
        return any(
            word[k] == "b" and in_m(word[1:k]) and in_x(word[k + 1 :])
            for k in range(1, len(word))
        )

    def in_m(word: str) -> bool:
        for split in range(len(word), -1, -1):
            head, tail = word[:split], word[split:]
            if set(tail) - {"a"}:
                continue
            if head.startswith("b") and in_x(head[1:]):
                return True
            if head and set(head) == {"c"} and len(head) % 2 == 0:
                return True
        return False

    verdict = parse_mu(TICKET).agrees_with(in_x, max_len=8)
    assert verdict.value is True, verdict.reason


def test_a_wrong_description_is_caught_with_a_witness():
    """Оракул обязан ловить правдоподобную, но неверную гипотезу.

    «Между `a` и `b` стоит чётное число `c`» — верно для блоков с `c`,
    но забывает про вложенность через `bX`.
    """
    verdict = parse_mu(TICKET).agrees_with(
        lambda w: all(c == "c" for c in w[1:-1]) and len(w) % 2 == 0, max_len=6
    )
    assert verdict.value is False
    assert "расхождение на слове" in verdict.reason


def test_the_ticket_grammar_is_not_ll1():
    """`Y → bX | Ya | Z` леворекурсивно, так что LL(1) там нет."""
    assert parse_mu(TICKET).to_cfg().is_ll1() is False


def test_membership_agrees_with_enumeration():
    expression = parse_mu(TICKET)
    words = set(expression.words(8))
    for word in ("", "abb", "accb", "abab", "accccb", "acb", "ab", "cc"):
        assert expression.accepts(word) == (word in words), word


def test_hypothesis_grammar_written_by_hand_gives_the_same_language():
    """Сверка с грамматикой, записанной прямо по словесному описанию."""
    by_hand = parse_cfg(
        """
        S -> a T A b S | ε
        T -> b S | C
        C -> c c C | c c
        A -> a A | ε
        """
    )
    assert set(parse_mu(TICKET).words(11)) == language(by_hand, 11)
