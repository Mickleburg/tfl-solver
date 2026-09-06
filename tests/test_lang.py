"""Тесты языка-предиката, таблиц различимости и лемм о накачке.

Контрольная точка взята не из учебника, а из **проверенной работы РК2 2025**
(фотография `photo_140` из учебного чата, см. `corpus/chat/FINDINGS.md`):
язык пересечён с регулярным до `a b^n a b^{n+1} aa`, префиксы `a b^i a`,
суффиксы `b^{j+1} aa`, и таблица обязана различить все пары.

Второе, что проверяется всюду: границы вывода. Отбитая накачка и различённая
таблица — **не** доказательства нерегулярности; функции обязаны возвращать
«не выяснено», а не «да». Обратные исходы (негодный свидетель, слипшиеся
префиксы) точны, и их значение — `False`.
"""

from __future__ import annotations

import re

import pytest

from tfl.lang import (
    at_least_classes,
    distinguishing_suffix,
    family,
    from_cfg,
    from_predicate,
    from_regex,
    nonregular_evidence,
    table,
)
from tfl.pump import (
    cf_splits,
    defeats_cf,
    defeats_regular,
    noncf_by_pumping,
    nonregular_by_pumping,
    regular_splits,
)


def counts(pattern: str, alphabet: str, name: str = ""):
    return from_predicate(lambda w: bool(re.fullmatch(pattern, w)), alphabet, name)


ANBN = from_predicate(
    lambda w: bool(re.fullmatch(r"a*b*", w)) and w.count("a") == w.count("b"),
    "ab",
    "aⁿbⁿ",
)
ANBNCN = from_predicate(
    lambda w: bool(re.fullmatch(r"a*b*c*", w))
    and w.count("a") == w.count("b") == w.count("c"),
    "abc",
    "aⁿbⁿcⁿ",
)
EVEN_A = from_predicate(lambda w: w.count("a") % 2 == 0, "ab", "чётное число a")

# Язык из проверенной работы: L ∩ ab⁺ab⁺aa = { a b^n a b^{n+1} aa | n ≥ 0 }.
_GRADED = re.compile(r"^ab(b*)ab(b*)aa$")


def _graded_member(word: str) -> bool:
    match = _GRADED.match(word)
    return bool(match) and len(match.group(2)) == len(match.group(1)) + 1


GRADED = from_predicate(_graded_member, "ab", "ab^n ab^{n+1} aa")


# --------------------------------------------------------------------------
# Язык как предикат
# --------------------------------------------------------------------------


def test_membership_and_words():
    assert "aabb" in ANBN and "abab" not in ANBN
    assert ANBN.words(4) == ["", "ab", "aabb"]


def test_intersection_of_predicates():
    both = ANBN & EVEN_A
    assert "aabb" in both and "ab" not in both


def test_complement():
    assert "ab" in ~EVEN_A and "ab" not in EVEN_A


def test_restrict_to_regular_language():
    """Основной приём курса: сузить язык пересечением с регулярным."""
    narrowed = EVEN_A.restrict("aa*bb*")
    assert "aabb" in narrowed
    assert "baab" not in narrowed  # не подходит под регулярку
    assert "ab" not in narrowed  # нечётное число a


def test_from_regex_agrees_with_predicate():
    assert from_regex("(aa)*", "ab").disagreements(
        from_predicate(lambda w: w in {"", "aa", "aaaa", "aaaaaa"}, "ab"), max_len=6
    ) == []


def test_from_cfg():
    from tfl.cfg import parse_cfg

    language = from_cfg(parse_cfg("S -> a S b | ε"))
    assert "aabb" in language and "aab" not in language


def test_disagreements_finds_the_mismatch():
    """«Равное число a и b» — не то же самое, что `aⁿbⁿ`, и это видно сразу."""
    wrong = from_predicate(lambda w: w.count("a") == w.count("b"), "ab", "равное число")
    bad = ANBN.disagreements(wrong, max_len=4)
    assert bad[0] == "ba"  # кратчайшее расхождение
    assert "abab" in bad


# --------------------------------------------------------------------------
# Таблица различимости
# --------------------------------------------------------------------------


def test_graded_paper_table_separates_every_pair():
    """Воспроизведение таблицы из проверенной работы РК2 2025."""
    verdict = at_least_classes(
        GRADED,
        family(lambda i: "a" + "b" * i + "a", 6),
        family(lambda j: "b" * (j + 1) + "aa", 6),
    )
    assert verdict.value is True
    assert verdict.witness.merged() == []


def test_graded_paper_table_is_diagonal_from_the_first_row():
    """Единицы стоят по диагонали со сдвигом: `pᵢ` подходит ровно к `sᵢ`."""
    built = table(
        GRADED,
        family(lambda i: "a" + "b" * i + "a", 5, start=1),
        family(lambda j: "b" * (j + 1) + "aa", 5, start=1),
    )
    assert built.is_diagonal


def test_anbn_has_at_least_six_classes():
    verdict = at_least_classes(
        ANBN, family(lambda i: "a" * i, 6), family(lambda j: "b" * j, 6)
    )
    assert verdict.value is True and "не меньше 6" in verdict.reason


def test_regular_language_family_collapses():
    """У регулярного языка семейство обязано слипнуться — иначе он не регулярен."""
    verdict = nonregular_evidence(EVEN_A, lambda i: "a" * i, lambda j: "a" * j, upto=5)
    assert verdict.value is False
    assert "не различены" in verdict.reason


def test_nonregular_evidence_is_never_proved():
    """Различив n префиксов, нерегулярность не доказали — только подкрепили.

    Это ровно та подмена, ради которой в проекте заведён `Verdict`:
    таблица до n = 8 выглядит убедительно и доказательством не является.
    """
    verdict = nonregular_evidence(ANBN, lambda i: "a" * i, lambda j: "b" * j, upto=8)
    assert verdict.value is None
    assert "при произвольном n" in verdict.reason


def test_table_markdown_has_a_row_per_prefix():
    built = table(ANBN, ["a", "aa"], ["b", "bb"])
    assert built.markdown().count("\n") == 3  # шапка, разделитель, две строки


def test_distinguishing_suffix_found_and_absent():
    assert distinguishing_suffix(ANBN, "a", "aa") is not None
    assert distinguishing_suffix(EVEN_A, "aa", "") is None


# --------------------------------------------------------------------------
# Накачка
# --------------------------------------------------------------------------


def test_regular_splits_respect_the_lemma():
    got = list(regular_splits("abc", 2))
    assert all(len(x + y) <= 2 and y for x, y, z in got)
    assert all(x + y + z == "abc" for x, y, z in got)


def test_cf_splits_respect_the_lemma():
    got = list(cf_splits("abcd", 3))
    assert all(len(v + x + y) <= 3 and (v or y) for _, v, x, y, _ in got)
    assert all(u + v + x + y + z == "abcd" for u, v, x, y, z in got)


def test_anbn_defeats_regular_pumping():
    assert defeats_regular(ANBN, "aaabbb", 3).value is True


def test_anbn_does_not_defeat_cf_pumping():
    """`aⁿbⁿ` контекстно-свободен, поэтому накачку отбить не может."""
    assert defeats_cf(ANBN, "aaabbb", 3).value is False


def test_anbncn_defeats_cf_pumping():
    assert defeats_cf(ANBNCN, "aaabbbccc", 3).value is True


def test_witness_outside_the_language_is_rejected():
    verdict = defeats_regular(ANBN, "aab", 2)
    assert verdict.value is False and "не принадлежит" in verdict.reason


def test_witness_shorter_than_p_is_rejected():
    verdict = defeats_regular(ANBN, "ab", 5)
    assert verdict.value is False and "неприменима" in verdict.reason


def test_nonregular_by_pumping_is_never_proved():
    verdict = nonregular_by_pumping(ANBN, lambda p: "a" * p + "b" * p, upto=6)
    assert verdict.value is None
    assert "при произвольном p" in verdict.reason


def test_noncf_by_pumping_is_never_proved():
    verdict = noncf_by_pumping(ANBNCN, lambda p: "a" * p + "b" * p + "c" * p, upto=4)
    assert verdict.value is None


def test_bad_witness_for_a_regular_language_is_refuted():
    verdict = nonregular_by_pumping(EVEN_A, lambda p: "a" * (2 * p), upto=4)
    assert verdict.value is False


def test_pumping_checks_every_split_not_a_convenient_one():
    """Смысл модуля: перебор всех разбиений, а не одного удобного.

    Для `aⁿbⁿ` при `p = 3` разбиение `x=ε, y=aaa, z=bbb` накачкой из языка
    не выводится… а вот `y=a` выводится. Ровно на этом ошибаются в отчётах,
    разбирая одно разбиение и объявляя вывод.
    """
    splits = list(regular_splits("aaabbb", 3))
    survivors = [
        (x, y, z) for x, y, z in splits
        if all((x + y * k + z) in ANBN for k in (0, 2, 3))
    ]
    assert splits and not survivors


@pytest.mark.parametrize("power", [0, 2, 3])
def test_zero_power_is_included(power):
    """Стирание (`i = 0`) — самый результативный случай, забывать его нельзя."""
    assert (("aaabbb"[:0] + "a" * power + "aabbb") in ANBN) == (power == 1)
