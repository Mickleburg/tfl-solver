"""Оракулы под задачи 1 и 2 РК2.

`RK2-A` — язык SRS над базисом либо язык грамматики с числовым условием;
`RK2-B` — язык, заданный теоретико-множественным описанием.

Эталоны берутся из официальных условий, лекций и обезличенных контрольных
примеров.
"""

from __future__ import annotations

from tfl.cfg import parse_cfg
from tfl.lang import from_cfg, from_predicate
from tfl.pump import defeats_dcfl, nondcfl_by_pumping
from tfl.srs import parse_srs

# --------------------------------------------------------------------------
# RK2-A: числовой инвариант грамматики против числового условия
# --------------------------------------------------------------------------


def test_variant_15_language_is_empty():
    """`S → aSbSb | aSa | a`, букв `a` вчетверо больше, чем `b`.

    `I(w) = |w|_a − 2|w|_b` нечётно на всех выводимых словах, а условие
    `|w|_a = 4|w|_b` делает его чётным. Противоречие ⇒ язык пуст.
    """
    grammar = parse_cfg("S -> a S b S b | a S a | a")
    assert sorted(grammar.residues({"a": 1, "b": -2}, 2)["S"]) == [1]
    verdict = grammar.counting_condition({"a": 1, "b": -4})
    assert verdict.value is False
    assert "по модулю 2" in verdict.reason
    # и перебором тоже пусто — согласие двух независимых способов
    language = from_cfg(grammar)
    assert [w for w in language.words(11) if w.count("a") == 4 * w.count("b")] == []


def test_second_work_with_the_same_trick():
    """`S → aSaSa | bSb | b`, букв `a` столько же, сколько `b`."""
    grammar = parse_cfg("S -> a S a S a | b S b | b")
    assert grammar.counting_condition({"a": 1, "b": -1}).value is False


def test_counting_condition_stays_silent_when_the_word_exists():
    """Вариант 7: условие выполнимо, и оракул честно не заявляет обратного."""
    grammar = parse_cfg("S -> a S b S | a S | b a b")
    verdict = grammar.counting_condition({"a": 1, "b": -2})
    assert verdict.value is None
    witnesses = [
        w for w in from_cfg(grammar).words(7) if w.count("a") == 2 * w.count("b")
    ]
    assert "aaabab" in witnesses


def test_residues_are_exact_not_sampled():
    """Неподвижная точка совпадает с перебором слов до длины 12."""
    grammar = parse_cfg("S -> a S b | a a S | b")
    coefficients = {"a": 1, "b": -1}
    computed = grammar.residues(coefficients, 5)["S"]
    enumerated = {
        (w.count("a") - w.count("b")) % 5 for w in from_cfg(grammar).words(12)
    }
    assert enumerated <= computed


# --------------------------------------------------------------------------
# RK2-A: замыкание базиса по SRS
# --------------------------------------------------------------------------


def test_closure_over_a_basis_family():
    """Вариант 19: `abb→abab, bab→baa, a→b, bb→ε` над `aⁿb^{2n}` даёт всё."""
    system = parse_srs("abb -> abab\nbab -> baa\na -> b\nbb -> ε")
    found = system.closure(["a" * n + "b" * (2 * n) for n in (1, 2)], max_len=5)
    assert {"a", "b", "aa", "ab", "ba", "bb"} <= found.words


def test_closure_marks_itself_inexact():
    """Правило `a → ab` удлиняет слово, значит обход упирается в потолок."""
    system = parse_srs("baa -> ba\nab -> ba\na -> ab")
    found = system.closure(["aba"], max_len=6)
    assert not found.exact
    assert "ba" in found.words


# --------------------------------------------------------------------------
# RK2-B: лемма Ю
# --------------------------------------------------------------------------


def union_language():
    """`{aⁿbⁿ} ∪ {aⁿb²ⁿ}` — КС, но не DCFL (пример из лекции 9)."""

    def predicate(word: str) -> bool:
        a, b = word.count("a"), word.count("b")
        if not a or word != "a" * a + "b" * b:
            return False
        return b == a or b == 2 * a

    return from_predicate(predicate, "ab", "{aⁿbⁿ} ∪ {aⁿb²ⁿ}")


def test_yu_witness_from_the_lecture():
    """`x = aⁿbⁿ⁻¹`, `y = b`, `z = bⁿ⁺¹` — свидетель прямо из лекции.

    Оба случая леммы проваливаются; «не выяснено» здесь — потолок метода,
    а не неудача: обобщение на произвольное `p` пишет человек.
    """
    language = union_language()

    def witness(p: int) -> tuple[str, str, str]:
        n = p + 2  # нужно n − 1 > p
        return "a" * n + "b" * (n - 1), "b", "b" * (n + 1)

    verdict = nondcfl_by_pumping(language, witness, upto=3)
    assert verdict.value is None
    assert "при всех p ≤ 3" in verdict.reason


def test_case_one_needs_both_words_to_stay():
    """Слабое прочтение случая 1 расходится с лекцией — проверка на нём.

    Синхронная накачка `a` и `b` внутри префикса оставляет в языке `xy`,
    но выводит из него `xz`. Лекция говорит «в случае 1 нет подходящей
    накачки», значит требуются оба слова.
    """
    language = union_language()
    n = 5
    x, y, z = "a" * n + "b" * (n - 1), "b", "b" * (n + 1)
    head = "a" * (n + 1) + "b" * n  # накачали по разу и a, и b
    assert (head + y) in language  # слабое прочтение случай 1 бы принял
    assert (head + z) not in language  # строгое — нет
    assert defeats_dcfl(language, x, y, z, 2).value is True


def test_deterministic_language_survives_the_lemma():
    """`{aⁿbⁿ}` — DCFL, и свидетеля против него быть не должно."""
    language = from_predicate(
        lambda w: w == "a" * w.count("a") + "b" * w.count("b")
        and w.count("a") == w.count("b")
        and bool(w),
        "ab",
        "{aⁿbⁿ}",
    )
    n = 5
    verdict = defeats_dcfl(language, "a" * n + "b" * (n - 1), "b", "b", 2)
    assert verdict.value is False
    assert "случай 1" in verdict.reason


def test_pair_must_share_the_first_letter_of_the_tails():
    language = union_language()
    verdict = defeats_dcfl(language, "aaaabbb", "b", "b" * 5, 20)
    assert verdict.value is False
    assert "|x|" in verdict.reason  # префикс короче p — лемма неприменима


def test_witness_must_lie_in_the_language():
    language = union_language()
    verdict = defeats_dcfl(language, "aaaa", "b", "bb", 2)
    assert verdict.value is False
    assert "не в языке" in verdict.reason


# --------------------------------------------------------------------------
# RK2-B: следствие теоремы Париха
# --------------------------------------------------------------------------


def test_squares_over_one_letter_are_not_context_free():
    """Гомоморфизм в унарный алфавит и разрыв между квадратами.

    Следствие теоремы Париха из лекции 7: над однобуквенным алфавитом
    регулярные и КС-языки совпадают, поэтому различающая таблица бьёт
    сразу по обоим свойствам.
    """
    import math

    from tfl.lang import family, one_letter_noncf

    squares = from_predicate(
        lambda w: bool(w) and set(w) <= {"a"} and math.isqrt(len(w)) ** 2 == len(w),
        "a",
        "{a^{m²}}",
    )
    assert squares.words(10) == ["a", "aaaa", "aaaaaaaaa"]
    verdict = one_letter_noncf(
        squares, family(lambda i: "a" * i, 9), family(lambda j: "a" * j, 9)
    )
    assert verdict.value is True
    assert "КС-свойство" in verdict.reason


def test_parikh_corollary_needs_one_letter():
    """Над двумя буквами следствие не работает — и оракул это говорит."""
    from tfl.lang import family, one_letter_noncf

    language = from_predicate(lambda w: w.count("a") == w.count("b"), "ab", "равные")
    verdict = one_letter_noncf(
        language, family(lambda i: "a" * i, 5), family(lambda j: "b" * j, 5)
    )
    assert verdict.value is False
    assert "однобуквенный" in verdict.reason
