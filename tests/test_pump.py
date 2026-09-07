"""Накачка со степенью, зависящей от разбиения.

Остальная накачка проверяется в `tests/test_lang.py` и `tests/test_mfa.py`.
Здесь — приём с семинара 2024 (issue #29), которого фиксированный набор
степеней не берёт: свидетель отбивает разбиения только тогда, когда
степень выбирается **по самому разбиению**.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Степень накачки, зависящая от разбиения (семинар 2024, issue #29)
# --------------------------------------------------------------------------


def middle_marker_language():
    """`{w₁ a w₂ | |w₁| = |w₂|, w₁ ≠ w₂}` — тот самый язык из erratum'а.

    > `{w₁w₂ | |w₁| = |w₂| & w₁ ≠ w₂}` — КС-язык, а вот если добавить
    > между `w₁` и `w₂` букву `a`, как было на семинаре, то уже нет.
    """
    from tfl.lang import from_predicate

    def predicate(word: str) -> bool:
        if len(word) % 2 != 1:
            return False
        half = len(word) // 2
        return word[half] == "a" and word[:half] != word[half + 1 :]

    return from_predicate(predicate, "ab", "{w₁ a w₂ | |w₁|=|w₂|, w₁≠w₂}")


def seminar_witness(n: int) -> str:
    """`b^{n!+n} a b^n a b^n a b^{n!+n}` — свидетель преподавателя."""
    from math import factorial

    tail = "b" * (factorial(n) + n)
    return f"{tail}a{'b' * n}a{'b' * n}a{tail}"


def test_the_witness_belongs_to_the_language():
    language = middle_marker_language()
    for n in (2, 3, 4):
        assert seminar_witness(n) in language


def test_a_fixed_set_of_powers_loses_this_witness():
    """С набором `(0, 2, 3, 4)` свидетель не проходит — и это не его вина.

    Разбиение, которое «выживает», качает по одной букве `b` слева
    и справа от центральной `a`: при малых степенях слово остаётся
    в языке, и лемма зря объявляет свидетеля негодным.
    """
    from tfl.pump import defeats_cf

    verdict = defeats_cf(middle_marker_language(), seminar_witness(3), 3)
    assert verdict.value is False
    assert "накачивается, оставаясь в языке" in verdict.reason


def test_factorial_powers_make_the_witness_work():
    """`p!/i + 1`, где `i` — длина накачиваемого куска.

    Факториал берётся ради делимости: `p!` делится на любое `i ⩽ p`,
    поэтому добавить ровно `p!` букв удаётся при любой длине куска.
    """
    from tfl.pump import defeats_cf, factorial_powers

    language = middle_marker_language()
    for n in (3, 4):
        verdict = defeats_cf(language, seminar_witness(n), n, powers=factorial_powers())
        assert verdict.value is True, verdict.reason


def test_factorial_powers_do_not_break_a_genuine_cf_language():
    """Контроль: `aⁿbⁿ` контекстно-свободен, и никакая степень его не отобьёт."""
    from tfl.lang import from_predicate
    from tfl.pump import defeats_cf, factorial_powers

    good = from_predicate(
        lambda w: len(w) % 2 == 0 and w == "a" * (len(w) // 2) + "b" * (len(w) // 2), "ab"
    )
    for word, p in (("aabb", 2), ("aaabbb", 3)):
        assert defeats_cf(good, word, p, powers=factorial_powers()).value is False


def test_powers_may_also_be_a_plain_callable():
    """Набор степеней — либо кортеж, либо функция от разбиения. Проверяем оба."""
    from tfl.pump import _powers_for

    assert _powers_for((0, 2), ("b",), 3) == (0, 2)
    assert _powers_for(lambda pumped, p: (len(pumped[0]) + p,), ("bb",), 3) == (5,)


def test_unequal_pumped_pieces_fall_back_to_the_default_set():
    """`i` определено только когда обе части одной длины — иначе середина съедет."""
    from tfl.pump import POWERS, factorial_powers

    choose = factorial_powers()
    assert choose(("b", "bb"), 4) == POWERS
    assert choose(("bb", "bb"), 4) == (*POWERS, 13)  # 4!/2 + 1
