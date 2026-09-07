"""Скобочное представление по Хомскому–Шютценберже — лекция 8 (2021).

Конструкция взята со слайдов дословно, и проверка её же средствами
показала, что напечатанное в лекции регулярное условие `R` теоремы
не обеспечивает. Тесты фиксируют и построение, и найденную дыру,
и условие, при котором равенство держится.
"""

from __future__ import annotations

import pytest

from tfl.cfg import parse_cfg
from tfl.schutzenberger import bracketed, dyck_words, is_dyck

# --------------------------------------------------------------------------
# Язык Дика
# --------------------------------------------------------------------------

PAIRS = (("[", "]"), ("(", ")"))


def test_dyck_respects_bracket_types():
    assert is_dyck(tuple("([])"), PAIRS)
    assert not is_dyck(tuple("([)]"), PAIRS)
    assert not is_dyck(tuple("(()"), PAIRS)
    assert is_dyck((), PAIRS)


def test_dyck_words_are_generated_not_filtered():
    words = dyck_words(PAIRS, 4)
    assert ("(", ")") in words and ("(", "[", "]", ")") in words
    assert ("(", "]") not in words
    assert all(is_dyck(word, PAIRS) for word in words)


# --------------------------------------------------------------------------
# Построение по теореме
# --------------------------------------------------------------------------


def test_the_construction_follows_the_slide():
    """> 1. `A → BC` порождает `A → [ₙB]ₙ(ₙC)ₙ`
    > 2. `A → a` порождает `A → [ₙ]ₙ(ₙ)ₙ`
    """
    representation = bracketed(parse_cfg("S -> A B\nA -> a\nB -> b"))
    printed = {str(p) for p in representation.grammar.productions}
    assert "A → [₁ ]₁ (₁ )₁" in printed
    assert "B → [₂ ]₂ (₂ )₂" in printed
    assert "S → [₃ A ]₃ (₃ B )₃" in printed


def test_only_one_bracket_carries_the_letter():
    """«`h([ₙ) = a`, для остальных скобок так же» буквально читать нельзя.

    Если бы в `a` переходили все четыре скобки, `h([ₙ]ₙ(ₙ)ₙ)` дало бы
    `aaaa` вместо `a`, и теорема бы не работала. В букву переходит одна.
    """
    representation = bracketed(parse_cfg("S -> a"))
    carrying = {s for s, image in representation.homomorphism.items() if image}
    assert carrying == {"[₁"}
    assert representation.image(("[₁", "]₁", "(₁", ")₁")) == "a"


def test_the_homomorphic_image_is_the_original_language():
    """Главная проверка: `h(L(G′)) = L(G)`. Грамматика для `{aⁿbⁿ | n > 0}`."""
    representation = bracketed(parse_cfg("S -> A X | A B\nX -> S B\nA -> a\nB -> b"))
    verdict = representation.agrees_with_source(40)
    assert verdict.value is True, verdict.reason
    assert {representation.image(w) for w in representation.words(40)} == {"ab", "aabb"}


def test_epsilon_has_no_bracket_representation():
    with pytest.raises(ValueError, match="ε"):
        bracketed(parse_cfg("S -> a S b | ε"))


# --------------------------------------------------------------------------
# Проверка самой теоремы
# --------------------------------------------------------------------------


def test_the_printed_regular_condition_is_not_enough():
    """Найдено проверкой: `L′ = R ∩ PARENₙ` в лекционной записи неверно.

    На грамматике из одного правила `S → a` язык `L(G′)` состоит
    из единственного слова `[₁]₁(₁)₁`, а условию со слайда 9
    удовлетворяют и другие правильные скобочные последовательности:
    например `[₁[₁[₁]₁]₁]₁` — она начинается с `[₁` (правило стартового
    нетерминала), а «все `]₁` раньше `(₁`» выполнено впустую,
    потому что `(₁` в слове нет вовсе.
    """
    representation = bracketed(parse_cfg("S -> a"))
    verdict = representation.check_theorem(8, level="слайд-9")
    assert verdict.value is False
    assert verdict.witness not in representation.words(8)
    assert is_dyck(verdict.witness, representation.pairs)
    assert representation.in_regular(verdict.witness, level="слайд-9")


def test_the_properties_from_the_earlier_slides_do_not_close_the_hole():
    """Свойства `L(G′)` со слайдов 7-8 в `R` не вошли — и не спасли бы.

    Они ничего не говорят ни про то, что за `]ₙ` обязана идти `(ₙ`,
    ни про согласование скобок по нетерминалам.
    """
    verdict = bracketed(parse_cfg("S -> a")).check_theorem(8, level="свойства")
    assert verdict.value is False


def test_the_strengthened_condition_makes_the_theorem_hold():
    """Строгий уровень: локальные переходы с учётом нетерминалов и конец слова.

    Оба добавленных требования локальны, так что `R` остаётся регулярным
    и теорема сохраняет смысл: разбор делится на лексический анализ
    и скобочную структуру.
    """
    for text, cap in (
        ("S -> a", 8),
        ("S -> A B\nA -> a\nB -> b", 16),
        ("S -> A S | a\nA -> a", 16),
        ("S -> A B | A C\nC -> S B\nA -> a\nB -> b", 20),
    ):
        verdict = bracketed(parse_cfg(text)).check_theorem(cap, max_nodes=1_500_000)
        assert verdict.value is True, (text, verdict.reason)


def test_word_must_end_with_a_round_closing_bracket():
    """Условия на конец слова в лекции нет, а без него в `R` лезут обрубки.

    `[₁]₁` — правильная скобочная последовательность, все локальные
    запреты к ней неприменимы (следующей скобки просто нет), и в `R`
    она попадала бы, хотя в `L(G′)` её нет.
    """
    representation = bracketed(parse_cfg("S -> a"))
    assert not representation.in_regular(("[₁", "]₁"))
    assert representation.in_regular(("[₁", "]₁", "(₁", ")₁"))


def test_a_weak_level_may_run_out_of_budget_and_says_so():
    """На слабых условиях отсечение не работает, и это честно сообщается."""
    verdict = bracketed(parse_cfg("S -> A B\nA -> a\nB -> b")).check_theorem(
        12, level="слайд-9", max_nodes=50_000
    )
    assert verdict.value is None
    assert "не уложился" in verdict.reason


def test_an_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="уровень"):
        bracketed(parse_cfg("S -> a")).in_regular(("[₁",), level="какой-нибудь")
