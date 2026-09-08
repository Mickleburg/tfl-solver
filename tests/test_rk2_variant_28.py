"""РК2 2025, вариант 28, задача 1: язык SRS над базисом.

Разбор — `reports/rk2-variant-28-task1/report.md`, решение —
`tools/rk2_variant_28.py`. Задача была открытой: обе проверенные работы
получили по 1 баллу и дали разные ответы, оба опровергнуты оракулом.

Здесь закреплено и то, что доказано (инварианты, опровержения), и само
описание языка — в двух направлениях сразу. Необходимость: всякое слово
точного замыкания описанию удовлетворяет. Достаточность: всякому слову,
удовлетворяющему описанию, **предъявляется вывод**, и каждый его шаг
сверяется с самой системой.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.srs import parse_srs
from tfl.words import iter_words
from tools.rk2_variant_28 import (
    RULES,
    bases,
    belongs,
    check_derivation,
    closure,
    derivation,
    language,
    slots,
    unslot,
)

LIMIT = 8


@pytest.fixture(scope="module")
def system():
    return parse_srs(RULES)


@pytest.fixture(scope="module")
def exact():
    """Точное `L ∩ Σ^{⩽ LIMIT}` — обходом, независимо от описания."""
    return language(LIMIT)


def words_upto(limit: int) -> list[str]:
    return [
        "".join(letters)
        for length in range(1, limit + 1)
        for letters in itertools.product("ab", repeat=length)
    ]


def prefix_a(word: str) -> int:
    index = 0
    while index < len(word) and word[index] == "a":
        index += 1
    return index


# --------------------------------------------------------------------------
# Модель слотов сверена с самой системой
# --------------------------------------------------------------------------


def test_the_slot_model_agrees_with_the_rewriting_system(system):
    """Векторная модель — не пересказ правил, а их перевод, и он сверен.

    Обход по слотам точен там, где обход по словам требовал бы оговорки:
    число `a` не растёт, число `b` не убывает, поэтому путь к короткому
    слову не уходит за границу по `b`. Именно это здесь и проверяется —
    два независимых обхода дают одно множество.
    """
    for base in (1, 2, 3):
        mine = {
            unslot(vector)
            for vector in closure(base, 8)
            if len(unslot(vector)) <= 8
        }
        found = system.reachable(
            "a" * base + "b" * base + "a" * base, max_len=2 * base + 8
        )
        theirs = {word for word in found.words if len(word) <= 8}
        assert mine == theirs, base


def test_slots_and_words_are_inverse():
    for word in words_upto(5):
        assert unslot(slots(word)) == word


# --------------------------------------------------------------------------
# Описание языка — в обе стороны
# --------------------------------------------------------------------------


def test_the_description_matches_the_closure_exactly(exact):
    """Ни ложных «да», ни ложных «нет» на всех словах длины ⩽ 8.

    До длины 11 (2773 слова) сверено отдельным прогоном, он в тесте
    не держится только из-за времени.
    """
    wrong = [word for word in words_upto(LIMIT) if (word in exact) != belongs(word)]
    assert wrong == []
    assert len(exact) == 323


def test_every_word_of_the_language_gets_a_verified_derivation(exact):
    """Достаточность предъявляется, а не заявляется.

    Для каждого слова строится цепочка от базиса, и каждый её шаг
    проверяется самой системой переписывания.
    """
    for word in sorted(exact):
        chain = derivation(word)
        assert chain is not None, word
        assert chain[-1] == word, word
        base = bases(word)[0]
        assert chain[0] == "a" * base + "b" * base + "a" * base, word
        assert check_derivation(chain), word


def test_a_word_outside_the_language_has_no_derivation():
    for word in ("aabab", "ab", "aabb", "abb", "bbb", ""):
        assert not belongs(word), word
        assert derivation(word) is None, word


def test_the_teacher_question_from_the_margin():
    """Пометка на полях проверенной работы: «можно ли получить `aⁿ⁺¹bⁿa`?»

    Нет. Начальный блок `a` не удлиняется, поэтому базис обязан иметь
    `n+1` букву `a` в начале, то есть быть не меньше `n+1`; но букв `b`
    в слове всего `n`, а их число не убывает. Условие `n+1 ⩽ n` неверно.
    """
    for n in range(1, 6):
        assert not belongs("a" * (n + 1) + "b" * n + "a")
        assert belongs("a" * n + "b" * n + "a")


# --------------------------------------------------------------------------
# Инварианты, на которых держится описание
# --------------------------------------------------------------------------


def test_ba_is_a_factor_of_every_word(system, exact):
    """Ни одно правило не разрушает вхождение `ba`."""
    broken = [
        (word, nxt)
        for word in iter_words("ab", 7)
        if "ba" in word
        for nxt in system.step(word)
        if "ba" not in nxt
    ]
    assert broken == []
    assert all("ba" in word for word in exact)


def test_measure_never_decreases(system, exact):
    """d(w) = 2|w|_b − |w|_a равна нулю на базисе и только растёт."""

    def measure(word: str) -> int:
        return 2 * word.count("b") - word.count("a")

    assert all(measure("a" * n + "b" * n + "a" * n) == 0 for n in range(1, 5))
    dropped = [
        (word, nxt)
        for word in iter_words("ab", 7)
        for nxt in system.step(word)
        if measure(nxt) < measure(word)
    ]
    assert dropped == []
    assert all(measure(word) >= 0 for word in exact)


def test_leading_block_never_grows(system):
    grown = [
        (word, nxt)
        for word in iter_words("ab", 7)
        for nxt in system.step(word)
        if prefix_a(nxt) > prefix_a(word)
    ]
    assert grown == []


def test_the_last_slot_is_never_refilled(system):
    """Ключевой инвариант, которого не хватало: `i_k` растёт только от `a → ab`.

    `ba² → ba` выбрасывает слот с номером `t ⩽ k−1` и последний сохраняет,
    `ab → ba` его только уменьшает. В базисе `i_k = 0`, поэтому в последнем
    слоте могут стоять **только порождённые** буквы `b`.
    """
    grown = [
        (word, nxt)
        for word in iter_words("ab", 7)
        for nxt in system.step(word)
        if slots(nxt)[-1] > slots(word)[-1] and nxt != word + "b"
    ]
    assert grown == []


def test_no_b_ever_moves_right(system):
    """Для каждой `b` число букв `a` слева от неё не растёт.

    Проверяется по мультимножеству: правило `a → ab` добавляет новое
    значение, все прежние обязаны не увеличиться.
    """

    def lambdas(word: str) -> list[int]:
        found, seen = [], 0
        for letter in word:
            if letter == "a":
                seen += 1
            else:
                found.append(seen)
        return sorted(found)

    for word in iter_words("ab", 6):
        before = lambdas(word)
        for nxt in system.step(word):
            after = lambdas(nxt)
            assert len(after) >= len(before)
            # каждое прежнее значение накрывается не большим новым
            assert all(
                mine >= yours for mine, yours in zip(before, after[: len(before)])
            ), (word, nxt)


# --------------------------------------------------------------------------
# Опровержение обоих сданных ответов
# --------------------------------------------------------------------------


def test_both_graded_answers_are_refuted(exact):
    """Оба ответа из проверенных работ ошибаются на конкретных словах."""
    # photo_112: L = {aⁿbᵐ | m ≥ n ≥ 1}
    assert "aaabbba" in exact  # в языке, а гипотеза его не принимает
    assert "aabb" not in exact  # гипотеза принимает, а слова в языке нет

    # photo_166: L = (a|b)*b(a|b)*a
    assert "aaabbbab" in exact  # оканчивается на b — гипотеза отвергает
    assert "aaba" not in exact  # гипотеза принимает, а меры не хватает


def test_the_three_earlier_conditions_were_not_enough(exact):
    """Прежние наработки останавливались здесь; теперь понятно, чего не хватало."""
    word = "aabab"
    assert "ba" in word
    assert word.count("a") <= 2 * word.count("b")
    assert prefix_a(word) <= word.count("b")
    assert word not in exact
    # недостающее: базис вынужден быть n = 2, а тогда порождённых `b` нет
    assert bases(word) == []
    assert slots(word) == [0, 0, 1, 1]


# --------------------------------------------------------------------------
# Класс языка
# --------------------------------------------------------------------------


def test_the_slice_by_a_star_b_star_a():
    """`L ∩ a*b*a = {aᵐbⁿa | m ⩽ n}` — язык не регулярен.

    Пересечение с регулярным работает в нужную сторону: если срез
    не регулярен, то и `L` не регулярен.
    """
    for m in range(0, 7):
        for n in range(0, 7):
            assert belongs("a" * m + "b" * n + "a") == (n >= 1 and m <= n)


def test_the_slice_by_a_star_b_star_a_star():
    """`L ∩ a*b*a* = {aᵖbᵈaᵠ | q ⩾ 1, d ⩾ p, 2d ⩾ p+q}`."""
    for p in range(0, 6):
        for d in range(0, 6):
            for q in range(0, 6):
                expected = q >= 1 and d >= p and 2 * d >= p + q
                assert belongs("a" * p + "b" * d + "a" * q) == expected


def test_the_witness_defeats_pumping_for_every_length():
    """Не КС: свидетель `aᵖbᵖaᵖ` разбирается символьно, а не по срезу.

    Разбиение задаётся тройкой `(α, β, γ)` — сколько букв каждого блока
    попало в накачиваемые куски; условие `|vxy| ⩽ p` означает, что блоки
    1 и 3 одновременно не задеты. Слово после накачки степени `i` лежит
    в срезе тогда и только тогда, когда `(i−1)(β−α) ⩾ 0`
    и `(i−1)(2β−α−γ) ⩾ 0`. Ни одна тройка не переживает обе степени
    `i = 0` и `i = 2`, а значит рассуждение годится при **любом** `p`.
    """

    def stays(alpha: int, beta: int, gamma: int, power: int) -> bool:
        sign = power - 1
        return (
            sign * (beta - alpha) >= 0 and sign * (2 * beta - alpha - gamma) >= 0
        )

    survivors = [
        (alpha, beta, gamma)
        for alpha in range(6)
        for beta in range(6)
        for gamma in range(6)
        if (alpha == 0 or gamma == 0)
        and (alpha or beta or gamma)
        and stays(alpha, beta, gamma, 0)
        and stays(alpha, beta, gamma, 2)
    ]
    assert survivors == []
