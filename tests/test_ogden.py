"""Лемма Огдена в формулировке курса (лекция 6, «Бонус: лемма Огдена»).

Курсовая формулировка сильнее расхожей: отмеченные буквы должны быть
**во всех трёх** частях одной из двух троек — `x₁, y₁, z` либо `z, y₂, x₂`.
Контрольная точка — язык из самой лекции, на котором обычная накачка
бессильна, а Огден работает.
"""

from __future__ import annotations

import re

from tfl.lang import from_predicate
from tfl.pump import (
    cf_splits,
    defeats_ogden,
    noncf_by_ogden,
    noncf_by_pumping,
    ogden_splits,
)


def bad_language():
    """`{aᵐbⁿcⁿdⁿ | m > 0} ∪ {bⁱcʲdᵏ}` — «плохой» язык из лекции 6."""

    def predicate(word: str) -> bool:
        match = re.fullmatch(r"(a*)(b*)(c*)(d*)", word)
        if match is None:
            return False
        a, b, c, d = (len(group) for group in match.groups())
        if a:
            return b == c == d
        return True

    return from_predicate(predicate, "abcd", "{aᵐbⁿcⁿdⁿ} ∪ {bⁱcʲdᵏ}")


def lecture_witness(n: int) -> tuple[str, set[int]]:
    """Слово `ab²ⁿc²ⁿd²ⁿ` с отметкой `n` последних букв `d` — ход из лекции."""
    word = "a" + "b" * (2 * n) + "c" * (2 * n) + "d" * (2 * n)
    return word, set(range(len(word) - n, len(word)))


def test_plain_pumping_fails_on_this_language():
    """Именно поэтому лекция и достаёт Огдена: обычная накачка не берёт.

    Стирание единственной `a` оставляет слово в языке — оно попадает
    во вторую часть объединения, `bⁱcʲdᵏ`.
    """
    verdict = noncf_by_pumping(
        bad_language(),
        lambda p: "a" + "b" * (2 * p) + "c" * (2 * p) + "d" * (2 * p),
        upto=3,
    )
    assert verdict.value is False
    assert "накачивается" in verdict.reason


def test_ogden_holds_where_plain_pumping_fails():
    """Тот же свидетель с отметками отбивает все допустимые разбиения.

    Содержательным перебор становится с `n = 3`: раньше отмеченных букв
    не хватает, чтобы заполнить тройку целиком. При `n = 3, 4, 5` разбиений
    уже сотни и тысячи, и ни одно не остаётся в языке.
    """
    verdict = noncf_by_ogden(bad_language(), lecture_witness, upto=5)
    assert verdict.value is None
    assert "отбивает накачку по Огдену при всех n ≤ 5" in verdict.reason
    for n in (3, 4, 5):
        word, marks = lecture_witness(n)
        assert len(list(ogden_splits(word, marks, n))) > 100
        assert defeats_ogden(bad_language(), word, marks, n).value is True


def test_empty_split_set_is_not_a_pass():
    """Пустой перебор ничего не доказывает — и вердикт это говорит.

    При `n ≤ 2` отмеченных букв меньше трёх, поэтому ни «во всех трёх
    из x₁, y₁, z», ни «во всех трёх из z, y₂, x₂» выполнить нельзя,
    и допустимых разбиений нет вовсе. Возвращать здесь «отбил» было бы
    ложью: проверять оказалось нечего.
    """
    language = bad_language()
    for n in (1, 2):
        word, marks = lecture_witness(n)
        assert list(ogden_splits(word, marks, n)) == []
        verdict = defeats_ogden(language, word, marks, n)
        assert verdict.value is None
        assert "допустимых разбиений нет" in verdict.reason


def test_wholly_vacuous_witness_is_reported_as_such():
    """Если содержательным не оказался ни один `n`, это отдельный исход."""
    verdict = noncf_by_ogden(bad_language(), lecture_witness, upto=2)
    assert verdict.value is None
    assert "отметка вырождена" in verdict.reason


def test_marks_cut_off_exactly_the_split_that_saved_the_adversary():
    """Механизм леммы: отметки запрещают то разбиение, которым он спасался.

    Обычную накачку ломает разбиение, качающее одинокую `a`
    (`v = ε`, `y = a`): при `k = 0` слово уходит во вторую часть
    объединения. Под отметками на хвосте из `d` это разбиение
    недопустимо — ни в `x₁`, ни в `z` отмеченных букв нет.
    """
    n = 3
    word, marks = lecture_witness(n)
    saving = ("", "", "", "a", word[1:])
    assert saving in set(cf_splits(word, n))
    assert saving not in set(ogden_splits(word, marks, n))
    # и при этом разбиения у Огдена есть — лемма не выродилась в пустоту
    assert list(ogden_splits(word, marks, n))


def test_all_three_parts_of_a_triple_must_be_marked():
    """Разбиения, где отмечена лишь часть тройки, не порождаются.

    Это и есть разница с расхожей формулировкой «`y₁` и `y₂` вместе
    содержат отмеченную позицию».
    """
    word = "aabbcc"
    marks = {4, 5}  # отмечены только последние две буквы
    for x1, y1, z, y2, x2 in ogden_splits(word, marks, 2):
        left = bool(_marked(x1, 0, marks) and _marked(y1, len(x1), marks))
        offsets = _offsets(x1, y1, z, y2)
        in_z = _marked(z, offsets[2], marks)
        first = _marked(x1, 0, marks) and _marked(y1, offsets[1], marks) and in_z
        second = (
            in_z
            and _marked(y2, offsets[3], marks)
            and _marked(x2, offsets[3] + len(y2), marks)
        )
        assert first or second
        assert left or True  # обе тройки допустимы, важно что хоть одна полна


def _offsets(x1: str, y1: str, z: str, y2: str) -> tuple[int, int, int, int]:
    return 0, len(x1), len(x1) + len(y1), len(x1) + len(y1) + len(z)


def _marked(part: str, offset: int, marks: set[int]) -> bool:
    return any(offset + i in marks for i in range(len(part)))


def test_witness_must_lie_in_the_language():
    verdict = defeats_ogden(bad_language(), "abbcd", {3, 4}, 2)
    assert verdict.value is False
    assert "не принадлежит" in verdict.reason


def test_not_enough_marked_letters():
    language = bad_language()
    word = "abbccdd"
    verdict = defeats_ogden(language, word, {6}, 3)
    assert verdict.value is False
    assert "не менее 3" in verdict.reason


def test_marks_outside_the_word_are_refused():
    verdict = defeats_ogden(bad_language(), "abbccdd", {2, 99}, 2)
    assert verdict.value is False
    assert "за пределами слова" in verdict.reason


def test_ogden_also_kills_the_classic_non_cf_language():
    """`{aⁿbⁿcⁿ}` — Огден работает и там, где хватало обычной накачки."""
    language = from_predicate(
        lambda w: bool(re.fullmatch(r"(a*)(b*)(c*)", w))
        and len(set(len(g) for g in re.fullmatch(r"(a*)(b*)(c*)", w).groups())) == 1,
        "abc",
        "{aⁿbⁿcⁿ}",
    )

    def witness(n: int) -> tuple[str, set[int]]:
        word = "a" * n + "b" * n + "c" * n
        return word, set(range(n))

    for n in (3, 4):
        word, marks = witness(n)
        assert list(ogden_splits(word, marks, n))
        assert defeats_ogden(language, word, marks, n).value is True
