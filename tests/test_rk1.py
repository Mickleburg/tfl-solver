"""Оракулы под РК1.

Контрольные точки взяты из авторского разбора пробной работы
(`corpus/txt/FormalLanguageTheory_2023_RK1_probe_task_solutions.txt`)
и из условий 2025 года (`rk1_tfl_2025.pdf`).
"""

from __future__ import annotations

from tfl.automata import DFA
from tfl.lang import counter_dfa, counter_range, from_predicate
from tfl.myhill import extended_fooling_set
from tfl.words import iter_words


def occurrences(word: str, pattern: str) -> int:
    return sum(
        1
        for i in range(len(word) - len(pattern) + 1)
        if word[i : i + len(pattern)] == pattern
    )


# --------------------------------------------------------------------------
# RK1-B: счётчики подслов
# --------------------------------------------------------------------------


def test_ab_equals_ba_is_regular_and_the_dfa_is_produced():
    """Классика курса: `|w|_ab` и `|w|_ba` различаются не более чем на 1.

    Обход достижимых состояний завершается, значит регулярность доказана,
    а свидетель — готовый ДКА, который можно предъявить в отчёте.
    """
    verdict = counter_dfa({"ab": 1, "ba": -1}, "ab")
    assert verdict.value is True
    machine = verdict.witness
    assert len(machine) == 5
    for word in iter_words("ab", 13):
        assert machine.accepts(word) == (
            occurrences(word, "ab") == occurrences(word, "ba")
        ), word


def test_equal_letters_defeats_the_method():
    """`|w|_a = |w|_b` — разность не ограничена, обход не сходится.

    Про сам язык вердикт молчит: это ограничение приёма, а не свойство.
    """
    verdict = counter_dfa({"a": 1, "b": -1}, "ab", budget=400)
    assert verdict.value is None
    assert "не ограничена" in verdict.reason


def test_range_growth_is_the_quick_diagnostic():
    """Постоянный размах — признак ограниченности, растущий — нет."""
    bounded = counter_range({"ab": 1, "ba": -1}, "ab")
    assert {hi - lo for lo, hi in bounded.values()} == {2}

    growing = counter_range({"ab": 1, "bb": -1}, "ab")
    spans = [hi - lo for _, (lo, hi) in sorted(growing.items())]
    assert spans == sorted(spans) and spans[-1] > spans[0]


def test_variant_6_first_conjunct_is_regular():
    """В6 2025: `|w₁|_ab = |w₁|_ba` — та самая ограниченная разность."""
    assert counter_dfa({"ab": 1, "ba": -1}, "ab").value is True


def test_variant_4_first_conjunct_is_not_handled_by_the_method():
    """В4 2025: `|w|_ab = |w|_bb` — размах растёт линейно, приём не берёт."""
    verdict = counter_dfa({"ab": 1, "bb": -1}, "ab", budget=2000)
    assert verdict.value is None


# --------------------------------------------------------------------------
# RK1-A: автомат остатков (задача II авторского разбора)
# --------------------------------------------------------------------------


def ternary_multiples_of_five() -> DFA:
    """Троичные числа, кратные 5: состояние — остаток, переход `r·3 + d`."""
    delta = {(r, d): (r * 3 + int(d)) % 5 for r in range(5) for d in "012"}
    return DFA(frozenset("012"), 0, frozenset({0}), delta)


def test_remainder_automaton_matches_the_authors_table():
    """Таблица переходов из разбора: строки — остатки, столбцы — цифры."""
    machine = ternary_multiples_of_five()
    expected = {
        0: (0, 1, 2),
        1: (3, 4, 0),
        2: (1, 2, 3),
        3: (4, 0, 1),
        4: (2, 3, 4),
    }
    for state, row in expected.items():
        assert tuple(machine.delta[(state, d)] for d in "012") == row


def test_the_automaton_is_minimal_even_among_nfa():
    """«Никакой НКА для этого языка не может иметь меньше 5 состояний».

    Обоснование в разборе — расширенный критерий Глайстера–Шаллита,
    то есть верхнетреугольная матрица. У нас это `extended_fooling_set`.
    """
    machine = ternary_multiples_of_five().minimize()
    assert len(machine) == 5
    assert machine.class_count() == 5
    assert len(extended_fooling_set(machine).pairs) == 5


def test_the_automaton_really_recognises_multiples_of_five():
    machine = ternary_multiples_of_five()
    for word in iter_words("012", 7):
        value = int(word, 3) if word else 0
        assert machine.accepts(word) == (value % 5 == 0), word


# --------------------------------------------------------------------------
# RK1-B: повторяющиеся подслова (задача I авторского разбора)
# --------------------------------------------------------------------------


def repeated_block_language():
    """`{v₁ z v₂ z | |z| ≥ 2, z, v₂ ∈ {a,b,c}*, v₁ ∈ {a,b}*}`."""

    def predicate(word: str) -> bool:
        n = len(word)
        for start in range(n):
            if any(ch == "c" for ch in word[:start]):
                continue  # v₁ ∈ {a,b}*
            for zlen in range(2, n - start + 1):
                z = word[start : start + zlen]
                if word.endswith(z) and n - zlen >= start + zlen:
                    return True
        return False

    return from_predicate(predicate, "abc", "{v₁ z v₂ z}")


def test_the_authors_witness_family_behaves_as_claimed():
    """`ca^{m+k} c a^m ∈ L`, а `c a^m c a^{m+k+1} ∉ L` — прямая цитата.

    Именно эта пара даёт нижнетреугольную матрицу принадлежности,
    из которой в разборе выводится нерегулярность.
    """
    language = repeated_block_language()
    for m in range(2, 6):
        for k in range(1, 4):
            assert ("c" + "a" * (m + k) + "c" + "a" * m) in language
            assert ("c" + "a" * m + "c" + "a" * (m + k + 1)) not in language


def test_a_bordered_block_lets_the_adversary_absorb_it():
    """Предостережение разбора: не брать `z` с ненулевой префикс-функцией.

    При `z = cⁿ` весь суффикс `z`, кроме двух первых букв, уходит в `v₂`,
    и слово остаётся в языке при любом сдвиге — свидетель негоден.
    """
    language = repeated_block_language()
    assert ("c" * 4 + "c" * 4) in language
    # сдвинутая пара всё равно в языке — различить префиксы не выйдет
    assert ("c" * 5 + "c" * 4) in language
    assert ("c" * 4 + "c" * 5) in language
