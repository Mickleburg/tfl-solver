"""Автоматы с памятью и ref-слова — лекция 12.

Приёмочные примеры взяты из самой лекции: ref-слово `[1a[2b]1x1]2x2`
с выписанными значениями переменных и 2-MFA для `{aⁿ²}` с выписанной
трассой памяти. Оба места лекция разбирает по шагам, поэтому сверять
можно не только ответ, но и промежуточные конфигурации.
"""

from __future__ import annotations

import pytest

from tfl.mfa import CLOSE, OPEN, Cell, MFA, parse_mfa, parse_ref_word

# --------------------------------------------------------------------------
# Ref-слова
# --------------------------------------------------------------------------


def test_ref_word_from_the_lecture():
    """`[1a[2b]1x1]2x2` задаёт `ababbab`, `x1 = ab`, `x2 = bab`."""
    ref = parse_ref_word("[1a[2b]1x1]2x2")
    assert ref.expand() == "ababbab"
    assert ref.values() == {1: "ab", 2: "bab"}


def test_blocks_may_be_interleaved():
    """> разные скобочные блоки могут быть перепутаны

    Именно этим ref-слова отличаются от обычной скобочной записи:
    `[1` закрывается внутри блока `[2`, и это законно.
    """
    ref = parse_ref_word("[1a[2b]1x1]2x2")
    kinds = [kind for kind, _ in ref.tokens]
    assert kinds == ["open", "letter", "open", "letter", "close", "read", "close", "read"]


def test_round_trip_printing():
    for text in ("[1a]1x1", "[1a[2b]1x1]2x2", "ab", "[1]1x1"):
        assert str(parse_ref_word(text)) == text


def test_reading_before_closing_is_rejected():
    with pytest.raises(ValueError, match="до закрытия"):
        parse_ref_word("[1ax1]1").expand()


def test_closing_an_unopened_cell_is_rejected():
    with pytest.raises(ValueError, match="не будучи открытой"):
        parse_ref_word("a]1").expand()


def test_empty_cell_expands_to_nothing():
    ref = parse_ref_word("[1]1x1a")
    assert ref.expand() == "a"
    assert ref.values() == {1: ""}


# --------------------------------------------------------------------------
# MFA: автомат лекции для {aⁿ²}
# --------------------------------------------------------------------------


def squares_mfa() -> MFA:
    """2-MFA со слайда 9 лекции 12. Backref-REGEX: `([1x2]1[2x1a]2)+`."""
    return parse_mfa(
        """
        q0 2 q1 o ⋄
        q1 1 q2 c o
        q2 a q3 ⋄ ⋄
        q3 2 q1 o c
        """,
        start="q0",
        accepting="q3",
        cells=2,
        alphabet="a",
    )


def test_squares_language():
    machine = squares_mfa()
    assert machine.words(16) == ["a", "a" * 4, "a" * 9, "a" * 16]


def test_memory_trace_matches_the_lecture():
    """> при k-ом посещении q3 получим состояние памяти вида `⟨⟨aᵏ⁻¹,c⟩,⟨aᵏ,o⟩⟩`"""
    machine = squares_mfa()
    for k in (1, 2, 3, 4):
        memory = machine.memory_after("a" * (k * k))
        assert memory == (Cell("a" * (k - 1), CLOSE), Cell("a" * k, OPEN))


def test_opening_a_closed_cell_resets_it():
    """`u' = v`, а не `u·v`: открытие закрытой ячейки обнуляет содержимое.

    Видно на самом автомате лекции: к третьему посещению `q3` первая
    ячейка содержит `aa`, а не накопленное `a·aa`.
    """
    memory = squares_mfa().memory_after("a" * 9)
    assert memory[0] == Cell("aa", CLOSE)


def test_flags_are_applied_before_reading():
    """Ячейку открывают, потом читают ленту — а не наоборот.

    Первый переход `q0 --2, o, ⋄--> q1` читает **вторую** ячейку, которая
    ещё пуста, и пишет прочитанное `ε` в только что открытую первую.
    Если бы флаги применялись после чтения, первая ячейка осталась бы
    закрытой и запись бы не произошла.
    """
    machine = squares_mfa()
    start = (Cell(), Cell())
    moves = machine.step("q0", 0, start, "a")
    assert moves == [("q1", 0, (Cell("", OPEN), Cell("", CLOSE)))]


def test_reading_from_an_open_cell_is_forbidden():
    """`b ∈ {1,…,k}` требует `r'_b = c`: читать можно только из закрытой."""
    machine = parse_mfa("q0 1 q1 o", start="q0", accepting="q1", cells=1, alphabet="a")
    assert machine.step("q0", 0, (Cell("a", CLOSE),), "a") == []


def test_closed_cell_keeps_its_content():
    machine = parse_mfa("q0 a q1 c", start="q0", accepting="q1", cells=1, alphabet="a")
    moves = machine.step("q0", 0, (Cell("aa", OPEN),), "a")
    assert moves == [("q1", 1, (Cell("aa", CLOSE),))]


# --------------------------------------------------------------------------
# Детерминизм и разбор
# --------------------------------------------------------------------------


def test_lecture_automaton_is_deterministic():
    assert squares_mfa().is_deterministic()


def test_a_letter_and_a_memory_read_from_one_state_break_determinism():
    """Условие лекции складывает ссылки и чтение буквы: `|⋃δ(q,i)| + |δ(q,b)| ⩽ 1`."""
    machine = parse_mfa(
        """
        q0 a q1 ⋄
        q0 1 q1 c
        """,
        start="q0",
        accepting="q1",
        cells=1,
        alphabet="a",
    )
    assert not machine.is_deterministic()


def test_parse_rejects_a_wrong_number_of_flags():
    with pytest.raises(ValueError, match="полей"):
        parse_mfa("q0 a q1 ⋄", start="q0", accepting="q1", cells=2, alphabet="a")


def test_digit_is_a_letter_when_the_alphabet_says_so():
    """В алфавите `{0,1}` цифра — буква, а не номер ячейки."""
    machine = parse_mfa("q0 1 q1 ⋄", start="q0", accepting="q1", cells=1, alphabet="01")
    assert machine.accepts("1")
    assert not machine.accepts("")


def test_agreement_with_a_predicate():
    verdict = squares_mfa().agrees_with(
        lambda w: len(w) > 0 and int(len(w) ** 0.5) ** 2 == len(w), max_len=10
    )
    assert verdict.value is True, verdict.reason


def test_epsilon_cycle_does_not_hang_the_search():
    """Повторная конфигурация отсекается, поэтому ε-петля обход не вешает."""
    machine = parse_mfa(
        """
        q0 ε q0 o
        q0 a q1 ⋄
        """,
        start="q0",
        accepting="q1",
        cells=1,
        alphabet="a",
    )
    assert machine.run("a").value is True
    assert machine.run("").value is False


def test_budget_exhaustion_is_not_a_refusal():
    """Кончился бюджет — вердикт «не выяснено», а не «не принимается».

    Слово `a⁹` автомат принимает, но при бюджете в пять конфигураций
    обход до ответа не доходит, и выдать `False` тут было бы враньём.
    """
    machine = squares_mfa()
    assert machine.run("a" * 9).value is True
    starved = machine.run("a" * 9, budget=5)
    assert starved.value is None
    assert "бюджет" in starved.reason


# --------------------------------------------------------------------------
# Лемма о накачке для CSY-языков (слайд 5 лекции 12)
# --------------------------------------------------------------------------


def anbn_language():
    from tfl.lang import from_predicate

    return from_predicate(
        lambda w: len(w) % 2 == 0 and w == "a" * (len(w) // 2) + "b" * (len(w) // 2),
        "ab",
        "aⁿbⁿ",
    )


def test_anbn_is_not_a_csy_language():
    """> с помощью леммы о накачке для CSY-языков легко доказать,
    > что `aⁿbⁿ` не описывается CSY-регуляркой
    """
    from tfl.pump import defeats_csy, noncsy_by_pumping

    language = anbn_language()
    for n in (2, 3, 4):
        verdict = defeats_csy(language, "a" * n + "b" * n, n)
        assert verdict.value is True, verdict.reason

    overall = noncsy_by_pumping(language, lambda n: "a" * n + "b" * n, upto=5)
    assert overall.value is None
    assert "при произвольном N" in overall.reason


def test_a_genuine_csy_language_survives_the_lemma():
    """Контроль: `a*` описывается регуляркой, и лемма обязана не сработать."""
    from tfl.lang import from_predicate
    from tfl.pump import noncsy_by_pumping

    verdict = noncsy_by_pumping(
        from_predicate(lambda w: set(w) <= {"a"}, "ab"), lambda n: "a" * (n + 1), upto=4
    )
    assert verdict.value is False
    assert "накачивается, оставаясь в языке" in verdict.reason


def test_empty_enumeration_is_not_a_bad_witness():
    """При `N = 1` разбиений нет вовсе: `|y| > 0` и `|x₀y| < 1` несовместимы.

    Такое `N` не говорит ни за, ни против. Засчитать его как «свидетель
    негоден» — ровно та ошибка, из-за которой лемма Огдена однажды
    возвращала `True` на пустом переборе.
    """
    from tfl.pump import csy_splits, defeats_csy, noncsy_by_pumping

    assert list(csy_splits("ab", 1)) == []
    assert defeats_csy(anbn_language(), "ab", 1).value is None

    overall = noncsy_by_pumping(anbn_language(), lambda n: "a" * n + "b" * n, upto=5)
    assert "при N = [1] разбиений нет вовсе" in overall.reason


def test_splits_pick_non_overlapping_occurrences_of_one_y():
    """Качается одно и то же `y` сразу во всех выбранных вхождениях."""
    from tfl.pump import csy_splits

    splits = dict(list(csy_splits("aabb", 2)))
    assert ("", "abb") in splits and splits[("", "abb")] == "a"
    assert ("", "", "bb") in splits and splits[("", "", "bb")] == "a"


def test_zero_power_is_excluded():
    """Лемма говорит только про `i > 0`, стирание здесь неправомерно.

    `a⁺` при стирании выпал бы из языка, и лемма ложно «сработала» бы.
    """
    from tfl.lang import from_predicate
    from tfl.pump import defeats_csy

    positive = from_predicate(lambda w: len(w) > 0 and set(w) <= {"a"}, "ab")
    verdict = defeats_csy(positive, "aaa", 2)
    assert verdict.value is False
