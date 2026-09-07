"""Конъюнктивные грамматики и автоматы Треллиса — лекция 11.

Приёмочные примеры взяты из самой лекции: грамматика для `{(aⁿb)ᵏ}`
и «язык равенства» `{wcw}`. Обе сверяются с предикатом, написанным
по словесной формулировке, а не по грамматике.
"""

from __future__ import annotations

import itertools
import re

import pytest

from tfl.conj import ConjunctiveGrammar, Trellis, parse_conjunctive
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Семантика конъюнкции
# --------------------------------------------------------------------------


def test_conjunction_is_intersection_on_the_same_word():
    """`A & B` — это одно и то же слово, разобранное дважды, а не две части.

    Здесь `A` даёт слова из одних `a`, `B` — слова чётной длины,
    а их конъюнкция — `a` в чётной степени.
    """
    grammar = parse_conjunctive("""
        S -> A & B
        A -> a A | ε
        B -> X X B | ε
        X -> a | b
    """)
    assert grammar.words(5) == ["", "aa", "aaaa"]


def test_single_conjunct_behaves_like_a_context_free_rule():
    from tfl.cfg import parse_cfg
    from tfl.parse import recognize

    text = "S -> a S b | ε"
    conjunctive = parse_conjunctive(text)
    plain = parse_cfg(text)
    for word in iter_words("ab", 7):
        assert conjunctive.recognize(word) == recognize(plain, word)


def test_epsilon_rules_are_handled():
    grammar = parse_conjunctive("S -> a S | ε")
    assert grammar.recognize("")
    assert grammar.recognize("aaa")
    assert not grammar.recognize("b")


def test_rule_without_conjunction_is_reported_as_plain():
    grammar = parse_conjunctive("S -> a S & S b | ε")
    assert not grammar.rules[0].is_plain
    assert grammar.rules[1].is_plain
    assert str(grammar.rules[0]) == "S → a S & S b"


# --------------------------------------------------------------------------
# Примеры из лекции
# --------------------------------------------------------------------------


def test_lecture_grammar_for_equal_blocks():
    """`S → SA & Cb | A` и далее — грамматика лекции для `{(aⁿb)ᵏ | n,k ⩾ 1}`."""
    grammar = parse_conjunctive("""
        S -> S A & C b | A
        A -> a A | a b
        C -> a C a | B
        B -> B A | b
    """)

    def equal_blocks(word: str) -> bool:
        if not re.fullmatch(r"(a+b)+", word):
            return False
        blocks = [len(part) for part in word.split("b")[:-1]]
        return len(set(blocks)) == 1

    verdict = grammar.agrees_with(equal_blocks, max_len=9)
    assert verdict.value is True, verdict.reason


def test_lecture_grammar_for_the_equality_language():
    """«Язык равенства» `{wcw | w ∈ {a,b}*}` — слайд 11 лекции 11."""
    grammar = parse_conjunctive("""
        S -> C & D
        C -> X C X | c
        D -> a A & a D | b B & b D | c E
        A -> X A X | c E a
        B -> X B X | c E b
        E -> X E | ε
        X -> a | b
    """)

    def equality(word: str) -> bool:
        parts = word.split("c")
        return len(parts) == 2 and parts[0] == parts[1]

    verdict = grammar.agrees_with(equality, max_len=7)
    assert verdict.value is True, verdict.reason


def test_disagreement_names_the_word_and_the_side():
    grammar = parse_conjunctive("S -> a S | ε")
    verdict = grammar.agrees_with(lambda w: w == "", max_len=3)
    assert verdict.value is False
    assert "грамматика лишнее" in verdict.reason


# --------------------------------------------------------------------------
# Экзамен: билет 23 пачки 2024
# --------------------------------------------------------------------------


def test_exam_grammar_for_strictly_increasing_blocks():
    """`{a^i₁ b a^i₂ b … b a^iⁿ | k > 0 ⇒ i_j < i_{j+k}}` — блоки растут.

    Конъюнкция нужна ровно в одном месте: `S → D & T` требует от **одного
    и того же** слова двух вещей сразу — что первые два блока сравнимы
    (`D`) и что хвост после первого `b` сам лежит в языке (`T`).
    """
    grammar = parse_conjunctive("""
        S -> D & T | A
        D -> P a M
        P -> a P a | b
        M -> a M | G
        G -> ε | b Z
        Z -> a Z | b Z | ε
        T -> a T | b S
        A -> a A | ε
    """)

    def increasing(word: str) -> bool:
        blocks = [len(part) for part in word.split("b")]
        return all(x < y for x, y in zip(blocks, blocks[1:]))

    verdict = grammar.agrees_with(increasing, max_len=9)
    assert verdict.value is True, verdict.reason


# --------------------------------------------------------------------------
# Автоматы Треллиса
# --------------------------------------------------------------------------


def middle_letter_trellis() -> Trellis:
    """Автомат для «центральная буква — `a`» (первая таблица лекции).

    Состояние нечётной ячейки — её средняя буква, чётной — две средние.
    Соседние ячейки одной длины, поэтому смешанных переходов не бывает.
    """
    letters = "ab"
    init = {x: f"O{x}" for x in letters}
    delta: dict[tuple[str, str], str] = {}
    for x, y in itertools.product(letters, repeat=2):
        delta[(f"O{x}", f"O{y}")] = f"E{x}{y}"
    for p, q, r, s in itertools.product(letters, repeat=4):
        delta[(f"E{p}{q}", f"E{r}{s}")] = f"O{q}"
    return Trellis(frozenset(letters), init, delta, frozenset({"Oa"}))


def test_trellis_recognises_the_central_letter():
    machine = middle_letter_trellis()
    for word in iter_words("ab", 9):
        if not word:
            continue
        expected = len(word) % 2 == 1 and word[len(word) // 2] == "a"
        assert machine.accepts(word) is expected, word


def test_trellis_rejects_the_empty_word():
    """Real-time автомат стартует с нижнего ряда, а у `ε` его нет."""
    assert middle_letter_trellis().accepts("") is False


def test_trellis_translation_matches_the_automaton():
    verdict = middle_letter_trellis().agrees_with_grammar(max_len=7)
    assert verdict.value is True, verdict.reason


def test_translation_of_a_trellis_is_linear():
    """Лекция: в каждой базисной правой части не больше одного нетерминала."""
    grammar = middle_letter_trellis().to_conjunctive()
    assert grammar.is_linear()
    assert isinstance(grammar, ConjunctiveGrammar)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_translation_works_for_arbitrary_automata(seed):
    """Конструкция обязана работать для любого автомата, не только удачного."""
    import random

    rng = random.Random(seed)
    letters = "ab"
    states = ["q0", "q1", "q2"]
    machine = Trellis(
        frozenset(letters),
        {x: rng.choice(states) for x in letters},
        {(p, q): rng.choice(states) for p in states for q in states},
        frozenset(rng.sample(states, 2)),
    )
    verdict = machine.agrees_with_grammar(max_len=6)
    assert verdict.value is True, verdict.reason
