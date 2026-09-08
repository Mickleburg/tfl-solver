"""Кодировки и морфизмы — семинар 05.09.2026 (issue #44).

Главное здесь — переход от перебора к доказательству. Разбор семинара
про морфизм `x ↦ 0`, `y ↦ 01`, `z ↦ 11` честно оговаривал: «проверено
перебором до длины 9 — коллизий нет. Это перебор, а не доказательство;
уникальную декодируемость строго проверяет алгоритм Сардинаса–Паттерсона».
Теперь проверяет.
"""

from __future__ import annotations

import pytest

from tfl.code import Code, parse_morphism

# --------------------------------------------------------------------------
# Сардинас–Паттерсон
# --------------------------------------------------------------------------


def test_the_classic_ambiguous_code():
    """`{a, ab, ba}`: слово `aba` разбирается двумя способами."""
    verdict = Code(("a", "ab", "ba")).uniquely_decodable()
    assert verdict.value is False
    assert verdict.witness.word == "aba"
    assert set(verdict.witness.left) | set(verdict.witness.right) <= {"a", "ab", "ba"}
    assert "".join(verdict.witness.left) == "".join(verdict.witness.right)
    assert verdict.witness.left != verdict.witness.right


def test_a_prefix_of_another_codeword_is_not_yet_ambiguity():
    """`{a, ab}` однозначен, хотя `a` — префикс `ab`.

    Частая ошибка: считать префиксность необходимым условием. Она
    достаточное, и только.
    """
    code = Code(("a", "ab"))
    assert not code.is_prefix_code()
    assert code.uniquely_decodable().value is True


def test_the_seminar_morphism_is_uniquely_decodable_by_proof():
    """`{0, 01, 11}` — это доказательство, а не перебор до девятой длины."""
    verdict = Code(("0", "01", "11")).uniquely_decodable()
    assert verdict.value is True
    assert "Сардинаса–Паттерсона" in verdict.reason


def test_a_prefix_code_is_decided_at_once():
    code = Code(("0", "10", "110"))
    assert code.is_prefix_code()
    assert code.uniquely_decodable().value is True


@pytest.mark.parametrize(
    "words,expected",
    [
        (("a", "b", "ab"), False),
        (("01", "1", "10"), False),
        (("a", "ab", "abb"), True),
        (("aa", "ab", "ba"), True),
        (("0", "01", "11"), True),
    ],
)
def test_the_algorithm_agrees_with_a_direct_search(words, expected):
    """Контроль: точный алгоритм и перебор коротких слов не расходятся.

    Перебор слабее — он не доказывает однозначности, — но противоречить
    алгоритму не должен.
    """
    code = Code(words)
    assert code.uniquely_decodable().value is expected
    if not expected:
        witness = code.ambiguity()
        assert len(code.decodings(witness.word)) > 1


def test_dangling_sets_terminate():
    """Хвосты — суффиксы кодовых слов, поэтому последовательность конечна."""
    sets = Code(("a", "ab", "ba")).dangling_sets()
    assert sets
    assert all(
        word in {code[i:] for code in ("a", "ab", "ba") for i in range(len(code) + 1)}
        for level in sets
        for word in level
    )


def test_an_empty_codeword_is_rejected():
    with pytest.raises(ValueError, match="пустое слово"):
        Code(("a", ""))
    with pytest.raises(ValueError, match="повторяются"):
        Code(("a", "a"))


# --------------------------------------------------------------------------
# Морфизмы
# --------------------------------------------------------------------------


def test_morphism_parsing_and_application():
    morphism = parse_morphism("x -> 0; y -> 01; z -> 11")
    assert morphism.apply("xyz") == "00111"
    assert str(morphism) == "x ↦ 0; y ↦ 01; z ↦ 11"


def test_injectivity_goes_through_the_same_algorithm():
    morphism = parse_morphism("x -> 0; y -> 01; z -> 11")
    verdict = morphism.is_injective()
    assert verdict.value is True
    assert "инъективен" in verdict.reason


def test_two_letters_with_one_image_are_caught_before_the_algorithm():
    verdict = parse_morphism("x -> 0; y -> 0").is_injective()
    assert verdict.value is False
    assert "один образ" in verdict.reason


def test_a_non_injective_morphism_is_refuted_with_a_witness():
    verdict = parse_morphism("x -> a; y -> ab; z -> ba").is_injective()
    assert verdict.value is False


# --------------------------------------------------------------------------
# Задержка раскодирования
# --------------------------------------------------------------------------


def test_the_delay_of_the_seminar_morphism_grows():
    """> Задержка растёт линейно и ничем не ограничена.

    Чтобы понять, начинается слово с `x` или с `y`, нужна чётность числа
    единиц — то есть чтение до конца.
    """
    morphism = parse_morphism("x -> 0; y -> 01; z -> 11")
    curve = morphism.delay_growth(6)
    assert curve == sorted(curve)
    assert curve[-1] > curve[0]
    assert len(set(curve[-3:])) > 1  # полки нет

    verdict = morphism.bounded_delay(6)
    assert verdict.value is None
    assert "растёт" in verdict.reason


def test_a_prefix_code_has_bounded_delay():
    """У префиксного кода задержка выходит на полку, а не растёт.

    Нулевой она бывает только когда первые символы образов все разные;
    у `{0, 10, 110}` образы `10` и `110` начинаются одинаково, и задержка
    равна единице — но дальше не растёт, и это видно по кривой.
    """
    morphism = parse_morphism("x -> 0; y -> 10; z -> 110")
    assert morphism.delay(5)[0] == 1
    assert "вышла на полку" in morphism.bounded_delay(5).reason

    instant = parse_morphism("x -> 0; y -> 1; z -> 2")
    assert instant.delay(4)[0] == 0


def test_the_delay_witness_is_a_real_pair():
    morphism = parse_morphism("x -> 0; y -> 01; z -> 11")
    length, pair = morphism.delay(5)
    first, second = pair
    assert first[0] != second[0]
    images = morphism.apply(first), morphism.apply(second)
    assert images[0][:length] == images[1][:length]
