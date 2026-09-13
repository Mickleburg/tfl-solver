"""База структурных образцов: гипотеза о вердикте и методе по форме условия.

Список переписан под локальные оракулы и **размечен по надёжности**. Смысл разметки
проверяется здесь: записи со статусом `WRONG` хранятся вместе
с исполняемым контрпримером, а не на слово.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.automata import NFA, dfa_of, equivalent
from tfl.cfg import parse_cfg
from tfl.hints import HEURISTIC, HINTS, THEOREM, WEAK, WRONG, Hint, match
from tfl.lang import from_cfg, from_predicate
from tfl.pump import defeats_cf

STATUSES = {THEOREM, HEURISTIC, WEAK, WRONG}


def words_upto(limit: int, alphabet: str) -> list[str]:
    return [
        "".join(letters)
        for length in range(0, limit + 1)
        for letters in itertools.product(alphabet, repeat=length)
    ]


# --------------------------------------------------------------------------
# Устройство базы
# --------------------------------------------------------------------------


def test_every_record_is_complete():
    assert len(HINTS) >= 20
    for hint in HINTS:
        assert isinstance(hint, Hint)
        assert hint.status in STATUSES, hint.name
        assert hint.shape and hint.guess and hint.method and hint.example
        assert hint.pattern.pattern


def test_names_are_unique():
    names = [hint.name for hint in HINTS]
    assert len(names) == len(set(names))


def test_doubtful_records_carry_a_caveat():
    """Запись, которой нельзя верить как есть, обязана объяснять почему."""
    for hint in HINTS:
        if hint.status in (WRONG, WEAK, HEURISTIC):
            assert hint.note, hint.name


def test_every_method_names_an_oracle_or_a_construction():
    """Метод — это либо наш модуль, либо явная конструкция, а не «подумать»."""
    for hint in HINTS:
        assert "tfl." in hint.method or "предъявить" in hint.method or (
            "грамматик" in hint.method
        ), hint.name


# --------------------------------------------------------------------------
# Поиск по условию
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Описать язык {w c wᴿ}. Детерминирован ли он?", "палиндром с разделителем"),
        ("Доказать, что {aⁿbⁿcⁿ} не контекстно-свободен", "три равных блока"),
        ("Проверить на регулярность {w | |w|_ab − |w|_ba ⩽ 1}", "счётчики подслов"),
        ("Завершается ли система переписывания ab → ba, a → ab?", "завершимость системы"),
    ],
)
def test_match_finds_the_right_shape(text, expected):
    assert expected in {hint.name for hint in match(text, limit=4)}


def test_match_returns_hypotheses_not_a_verdict():
    """Несколько образцов сразу — норма: база подсказывает, а не решает."""
    text = "Язык {w c w | w ∈ {a,b}*}: контекстно-свободен ли он, детерминирован ли?"
    found = match(text, limit=4)
    assert len(found) >= 2


def test_nothing_matches_an_unrelated_text():
    assert match("Сегодня хорошая погода", limit=4) == []


# --------------------------------------------------------------------------
# Записи, неверные как записаны
# --------------------------------------------------------------------------


def reverse_dfa(automaton):
    """Реверс: рёбра развернуть, конечные сделать начальными."""
    delta: dict[tuple[object, str], frozenset[object]] = {}
    for (source, letter), target in automaton.delta.items():
        delta.setdefault((target, letter), frozenset())
        delta[(target, letter)] |= {source}
    start = "→"
    return NFA(
        automaton.alphabet,
        start,
        frozenset([automaton.start]),
        delta,
        {start: frozenset(automaton.finals)},
    )


def test_reversal_of_a_dcfl_can_be_a_dcfl():
    """Контрпример к записи «реверс DCFL — не DCFL».

    Реверс регулярного языка регулярен, а всякий регулярный язык
    детерминированно контекстно-свободен. Строится явно: `a*b*` — ДКА,
    его реверс — тоже ДКА, и он распознаёт ровно `b*a*`.
    """
    forward = dfa_of("a*b*", "ab").minimize()
    backward = reverse_dfa(forward).determinize().minimize()
    assert equivalent(backward, dfa_of("b*a*", "ab").minimize())

    for word in words_upto(6, "ab"):
        assert forward.accepts(word) == backward.accepts(word[::-1])

    record = next(hint for hint in HINTS if hint.name == "реверс")
    assert record.status == WRONG
    assert "регулярн" in record.note


def test_context_free_is_not_closed_under_complement():
    """Контрпример к записи «дополнение известного языка — DCFL».

    Внутри `a*b*c*` дополнение языка `{aⁿbⁿcⁿ}` контекстно-свободно —
    грамматика строится явно и сверяется с предикатом. Сам же `{aⁿbⁿcⁿ}`
    не контекстно-свободен. Значит из «язык задан дополнением» не следует
    ни КС, ни тем более детерминизм: класс дополнением не наследуется.

    Дальше уже теорема, а не перебор: DCFL замкнут относительно
    дополнения, поэтому будь `¬{aⁿbⁿcⁿ}` детерминированным, был бы
    детерминированным и `{aⁿbⁿcⁿ}` — а он не КС.
    """
    grammar = parse_cfg(
        """
        S -> A C | D B
        A -> M | L
        M -> a M b | P
        P -> a P | a
        L -> a L b | Q
        Q -> b Q | b
        B -> U | V
        U -> b U c | R
        R -> b R | b
        V -> b V c | W
        W -> c W | c
        C -> c C | ε
        D -> a D | ε
        """
    )
    unequal = from_cfg(grammar)
    expected = from_predicate(
        lambda w: (
            w == "".join(sorted(w, key="abc".index))
            and not (w.count("a") == w.count("b") == w.count("c"))
        ),
        "abc",
    )
    for word in words_upto(7, "abc"):
        if word != "".join(sorted(word, key="abc".index)):
            continue  # вне a*b*c* — этот кусок регулярный и КС очевидно
        assert (word in unequal) == (word in expected), word

    record = next(hint for hint in HINTS if hint.name == "дополнение")
    assert record.status == WRONG
    assert "замкнут" in record.note


# --------------------------------------------------------------------------
# Записи, слабее истины
# --------------------------------------------------------------------------


def test_three_blocks_deserve_a_stronger_verdict():
    """`{aⁿbⁿcⁿ}` в исходной базе значился «не DCFL» — он не КС.

    Свидетель `aᵖbᵖcᵖ` отбивает накачку для КС-языков; конечный перебор
    доказательством не является, но вердикт «не DCFL» здесь заведомо мельче
    достижимого, и запись помечена соответственно.
    """
    language = from_predicate(
        lambda w: w == "a" * w.count("a") + "b" * w.count("b") + "c" * w.count("c")
        and w.count("a") == w.count("b") == w.count("c"),
        "abc",
    )
    for p in (2, 3, 4):
        verdict = defeats_cf(language, "a" * p + "b" * p + "c" * p, p)
        assert verdict.value is True, (p, verdict.reason)

    record = next(hint for hint in HINTS if hint.name == "три равных блока")
    assert record.status == WEAK
    assert "не КС" in record.guess


def test_grammar_properties_are_not_language_properties():
    """Две записи держатся на правиле проекта, и оно здесь названо явно."""
    for name in ("грамматика без конфликтов", "неоднозначное объединение в грамматике"):
        record = next(hint for hint in HINTS if hint.name == name)
        assert record.status == HEURISTIC
        assert "язык" in record.note


# --------------------------------------------------------------------------
# Встраивание в приём задачи
# --------------------------------------------------------------------------


def test_intake_reports_hints():
    """Приём задачи называет не только класс, но и гипотезу с методом."""
    from tfl.intake import analyse

    analysis = analyse("Описать язык {aⁿbⁿcⁿ}. К какому классу он относится?")
    assert analysis.hints
    text = analysis.report()
    assert "Структурные образцы" in text
    assert "гипотезы, не вердикты" in text


def test_intake_no_longer_confuses_determinism_with_termination():
    """`терминир` ловило «де**терминир**ован» — вопрос подменялся.

    Из-за этого всякая задача про детерминизм получала ещё и признак
    «завершимость», а он тянет вес к ЛР1 и SRS-вопросам билета.
    """
    from tfl.intake import find_asks

    asks = {evidence.name for evidence in find_asks("Детерминирован ли язык?")}
    assert asks == {"детерминизм"}
    assert "завершимость" in {
        e.name for e in find_asks("Завершается ли система переписывания?")
    }
