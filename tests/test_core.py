"""Тесты ядра оракулов: регексы, автоматы, классы эквивалентности.

Главный тест здесь — `test_two_constructions_agree`. Он прогоняет все 28
вариантов ЛР2 2025 через два независимых построения ДКА (Томпсон + подмножества
против производных Брзозовского) и требует, чтобы они совпали по языку и по
числу состояний после минимизации. Совпадение двух разных алгоритмов —
это и есть проверка, ради которой ядро строилось.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl import regex as rx
from tfl.automata import (
    brzozowski_dfa,
    counterexample,
    dfa_of,
    difference,
    disagreements,
    intersection,
    thompson,
    union,
)
from tfl.myhill import class_table, extended_fooling_set, fooling_set
from tfl.words import iter_words

VARIANTS_FILE = pathlib.Path(__file__).parent.parent / "evals" / "lab2_2025_variants.txt"


def load_variants() -> list[tuple[int, str]]:
    out = []
    for line in VARIANTS_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        num, pattern = line.split("\t", 1)
        out.append((int(num), pattern.strip()))
    return out


VARIANTS = load_variants()


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_precedence():
    # '|' слабее конкатенации, конкатенация слабее '*'
    assert rx.parse("ab|c") == rx.alt(rx.cat(rx.Sym("a"), rx.Sym("b")), rx.Sym("c"))
    assert rx.parse("ab*") == rx.cat(rx.Sym("a"), rx.star(rx.Sym("b")))


def test_normalization_makes_equal_asts():
    assert rx.parse("a|a") == rx.parse("a")
    assert rx.parse("a|b") == rx.parse("b|a")
    assert rx.parse("(a*)*") == rx.parse("a*")
    assert rx.parse("εa") == rx.parse("a")


def test_empty_alternative_is_epsilon():
    # В условиях курса встречается запись `(b | )` вместо `(b | ε)`.
    assert rx.parse("(b|)") == rx.parse("(b|ε)")


@pytest.mark.parametrize("bad", ["(a", "a)", "*a", "(a|)b)", "a|*"])
def test_parse_errors(bad):
    with pytest.raises(rx.ParseError):
        rx.parse(bad)


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_all_lab2_variants_parse(num, pattern):
    node = rx.parse(pattern)
    assert node.alphabet(), f"вариант {num}: пустой алфавит"


# --------------------------------------------------------------------------
# Автоматы
# --------------------------------------------------------------------------


def test_known_minimal_sizes():
    # Учебные примеры с известным ответом.
    assert len(dfa_of("(a|b)*abb")) == 4
    assert len(dfa_of("a(a|b)*a|b(a|b)*b")) == 5
    assert len(dfa_of("a*")) == 1
    assert len(dfa_of("ε")) == 1


def test_empty_language():
    dfa = dfa_of("∅")
    assert dfa.is_empty()
    assert not dfa.accepts("")


def test_epsilon_language():
    dfa = dfa_of("ε")
    assert dfa.accepts("")
    assert not dfa.accepts("a")


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_two_constructions_agree(num, pattern):
    """Томпсон+подмножества против производных Брзозовского."""
    node = rx.parse(pattern)
    alphabet = node.alphabet()

    thompson_dfa = thompson(node).determinize().minimize()
    brzozowski = brzozowski_dfa(node, alphabet).minimize()

    assert counterexample(thompson_dfa, brzozowski) is None, (
        f"вариант {num}: построения расходятся"
    )
    assert len(thompson_dfa) == len(brzozowski), (
        f"вариант {num}: разное число состояний после минимизации"
    )


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_dfa_matches_nfa_on_all_short_words(num, pattern):
    """ДКА против прямой симуляции ε-НКА — третий независимый распознаватель."""
    node = rx.parse(pattern)
    nfa = thompson(node)
    dfa = dfa_of(node)
    bad = disagreements(nfa.accepts, dfa.accepts, node.alphabet(), max_len=6)
    assert not bad, f"вариант {num}: расхождения на словах {bad}"


def test_minimize_is_idempotent():
    dfa = dfa_of("(a|b)*abb")
    assert len(dfa.minimize()) == len(dfa)


def test_boolean_operations():
    even_a = dfa_of("(b*ab*ab*)*")  # чётное число a
    ends_b = dfa_of("(a|b)*b")

    both = intersection(even_a, ends_b).minimize()
    assert both.accepts("aab")
    assert not both.accepts("ab")
    assert not both.accepts("aa")

    either = union(even_a, ends_b).minimize()
    assert either.accepts("aa")
    assert either.accepts("ab")

    only_even = difference(even_a, ends_b).minimize()
    assert only_even.accepts("aa")
    assert not only_even.accepts("aab")


def test_complement_roundtrip():
    dfa = dfa_of("(a|b)*abb")
    twice = dfa.complement().complement().minimize()
    assert counterexample(dfa, twice) is None


def test_counterexample_is_shortest():
    left = dfa_of("a*")
    right = dfa_of("a*b?" .replace("b?", "(b|ε)"))
    word = counterexample(left, right)
    assert word == "b"


def test_alphabets_are_unioned_on_comparison():
    """Автоматы над разными алфавитами сравниваются корректно.

    `a*` над {a} и `a*` над {a,b} — это один и тот же язык; а вот `a*` и
    `(a|b)*` различаются словом `b`, даже если первый автомат про `b`
    ничего не знает.
    """
    assert counterexample(dfa_of("a*"), dfa_of("a*")) is None
    assert counterexample(dfa_of("a*"), dfa_of("(a|b)*")) == "b"


# --------------------------------------------------------------------------
# Классы эквивалентности
# --------------------------------------------------------------------------


def test_class_table_proves_minimality():
    dfa = dfa_of("(a|b)*abb")
    table = class_table(dfa)
    assert len(table.prefixes) == dfa.class_count()
    assert table.is_minimal_proof
    assert not table.duplicate_rows()


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_class_table_consistent_for_all_variants(num, pattern):
    dfa = dfa_of(pattern)
    table = class_table(dfa)
    assert len(table.prefixes) == dfa.class_count(), f"вариант {num}: строк ≠ классов"
    assert table.is_minimal_proof, f"вариант {num}: таблица не различает классы"


def test_fooling_set_reproduces_consultation_example():
    """Пример из consa_rk2_2024.pdf: для a(a|b)*a|b(a|b)*b оценка равна 4.

    Симметричная версия теоремы даёт всего 2 — это и есть причина, по которой
    в курсе используется уточнённая, треугольная формулировка.
    """
    dfa = dfa_of("a(a|b)*a|b(a|b)*b")
    triangular = extended_fooling_set(dfa)
    assert triangular.bound == 4
    assert triangular.verify(dfa.accepts)
    assert fooling_set(dfa).bound < triangular.bound


@pytest.mark.parametrize("num,pattern", VARIANTS, ids=[f"v{n}" for n, _ in VARIANTS])
def test_fooling_sets_are_valid_and_bounded(num, pattern):
    """Оценка снизу корректна и не превосходит размер минимального ДКА."""
    dfa = dfa_of(pattern)
    for fs in (fooling_set(dfa), extended_fooling_set(dfa)):
        assert fs.verify(dfa.accepts), f"вариант {num}: {fs.kind} множество неверно"
        assert fs.bound <= len(dfa), f"вариант {num}: оценка НКА выше размера ДКА"


# --------------------------------------------------------------------------
# Сверка с чужими решениями (слой C)
# --------------------------------------------------------------------------
#
# Единственная внешняя проверка, доступная до выхода заданий 2026: работы
# текущего потока по тем же вариантам ЛР2 2025. Совпадение с ними ничего не
# доказывает об абсолютной правильности, но расхождение — сигнал разобраться.


def test_matches_dm800_variant_15():
    """`dm800-TFLlabs/lab2` — вариант 15, таблица классов на 17 строк.

    Расхождение 16 против 17 при первом прогоне оказалось состоянием-ловушкой:
    в их таблице это строка `aabaa` из одних минусов. С учётом ловушки
    представители классов совпадают посимвольно.
    """
    dfa = dfa_of("b*((ab*a)*(aabb|(babb)*))*")
    expected = [
        "", "a", "b", "aa", "ab", "ba", "aab", "aba", "bab",
        "aaba", "abab", "babb", "aabaa", "aabab", "babba", "babbaa", "babbaab",
    ]
    assert dfa.class_count() == 17
    assert len(dfa) == 16, "живых состояний на одно меньше: ловушка отброшена"
    assert sorted(class_table(dfa).prefixes) == sorted(expected)


def test_matches_prrromanssss_variant_2():
    """`Prrromanssss-formal-languages-labs/lab2` — вариант 2, min_dfa.dot на 22 состояния."""
    dfa = dfa_of(
        "((a*b*c*)*ab(a*b*c*)*bc(a|b|c)*)"
        "|((a|b|c)*bc(a*b*c*)*ab(a|bc|cc|bb)*)|abc"
    )
    assert len(dfa) == 22
    assert dfa.class_count() == 22, "ловушки нет: любое слово продолжается до языка"


# --------------------------------------------------------------------------
# Перечисление слов
# --------------------------------------------------------------------------


def test_iter_words_is_shortlex():
    assert list(iter_words("ab", 2)) == ["", "a", "b", "aa", "ab", "ba", "bb"]


def test_iter_words_min_len():
    assert list(iter_words("a", 3, min_len=2)) == ["aa", "aaa"]
