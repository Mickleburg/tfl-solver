r"""Расширенные регулярки ЛР4: группы захвата, ссылки, каркас.

Приёмочная таблица — вердикты преподавателя из issue #35
(`corpus/issues/lab4-issue35.md`). Языковые примеры — из условия
(`corpus/tasks/lab4-2024-task.md`).
"""

from __future__ import annotations

import pytest

from tfl.extre import (
    ExprRef,
    Group,
    ParseError,
    StrRef,
    groups,
    matches,
    parse_extended,
    skeleton_cfg,
    validate,
)
from tfl.parse import recognize
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Разбор и нумерация
# --------------------------------------------------------------------------


def test_round_trip_printing():
    for text in ("a", "ab", "a|b", "a*", "(a|b)", "(?:ab)", "(?1)", r"\1", "(?=ab)"):
        assert str(parse_extended(text)) == text


def test_groups_are_numbered_by_opening_bracket():
    """«Нумерация по номеру открывающей скобки» — считаются только захваты.

    В `((?:a(?2)|(bb))(?1))` вторая группа — это `(bb)`, а не `(?:…)`:
    иначе вердикт преподавателя «захват выражения из другой альтернативы»
    не имел бы смысла.
    """
    table = groups(parse_extended("((?:a(?2)|(bb))(?1))"))
    assert sorted(table) == [1, 2]
    assert str(table[2]) == "bb"


def test_references_do_not_consume_numbers():
    table = groups(parse_extended("(?1)(a|b)"))
    assert sorted(table) == [1]
    assert str(table[1]) == "a|b"


def test_nested_numbering_follows_the_text():
    table = groups(parse_extended("(a(bb)|b(cc))"))
    assert [str(table[n]) for n in (1, 2, 3)] == ["a(bb)|b(cc)", "bb", "cc"]


def test_parse_errors():
    for text in ("(a", "a)", "", "(?x)", "\\0", "a**|", "(?:)"):
        with pytest.raises(ParseError):
            parse_extended(text)


# --------------------------------------------------------------------------
# Приёмочная таблица issue #35
# --------------------------------------------------------------------------

VERDICTS = [
    ("(a|(bb))(?2)", True, "захват выражения, инициализация не требуется"),
    (r"(a(bb)|b(cc))\2", False, "группа 2 есть только в одной альтернативе"),
    ("((?:a(?2)|(bb))(?1))", True, "захват выражения из другой альтернативы"),
    ("(a(?1))", True, "синтаксически корректно, семантически бессмысленно"),
    (r"(a|(bb))\2", False, "на пути «a» группа 2 не заполнена"),
    ("(a|(bb))(a|(?3))", True, "пример из условия"),
    (r"(a|(?2))(a|(bb\1))", False, "группа 1 ещё не дочитана до конца"),
    ("(?1)(a|b)", True, "ссылки на выражение априори инициализированы"),
    (r"(\1)(a|b)", False, "нужно распарсить префикс, зайдя в группу"),
    ("(?1)(a|b)*(?1)", True, "под звёздочкой можно ссылаться на выражение"),
    (r"(a|b)*\1", False, "звёздочка может раскрыться в ноль итераций"),
    ("((?1)a|b)", True, "левая рекурсия синтаксически корректна"),
]


@pytest.mark.parametrize("text,expected,why", VERDICTS)
def test_teacher_verdicts(text: str, expected: bool, why: str):
    verdict = validate(parse_extended(text))
    assert verdict.value is expected, f"{text} — {why}: {verdict.reason}"


def test_uninitialised_reference_names_the_group():
    verdict = validate(parse_extended(r"(a|(bb))\2"))
    assert verdict.value is False
    assert "группе 2" in verdict.reason


# --------------------------------------------------------------------------
# Ограничения условия
# --------------------------------------------------------------------------


def test_at_most_nine_capture_groups():
    ok = "(a)" * 9
    too_many = "(a)" * 10
    assert validate(parse_extended(ok)).value is True
    verdict = validate(parse_extended(too_many))
    assert verdict.value is False
    assert "разрешено 9" in verdict.reason


def test_lookahead_may_not_contain_a_capture_group():
    verdict = validate(parse_extended("(?=(a))b"))
    assert verdict.value is False
    assert "содержит группу захвата" in verdict.reason


def test_lookahead_may_not_be_nested():
    verdict = validate(parse_extended("(?=a(?=b))c"))
    assert verdict.value is False
    assert "вложенную проверку" in verdict.reason


def test_reference_to_a_missing_group():
    verdict = validate(parse_extended("(a)(?5)"))
    assert verdict.value is False
    assert "несуществующую группу" in verdict.reason


# --------------------------------------------------------------------------
# Каркас: КС-грамматика
# --------------------------------------------------------------------------


def test_expression_reference_is_plain_substitution():
    """«`(?1)(a|b)` — не что иное, как `(a|b)(a|b)`, и это известно сразу»."""
    node = parse_extended("(?1)(a|b)")
    accepted = {w for w in iter_words("ab", 3) if matches(node, w)}
    assert accepted == {"aa", "ab", "ba", "bb"}


def test_example_from_the_task_statement():
    """`(aa|bb)(?1)` распознаёт `{aabb, aaaa, bbaa, bbbb}` — цитата условия."""
    node = parse_extended("(aa|bb)(?1)")
    accepted = {w for w in iter_words("ab", 4) if matches(node, w)}
    assert accepted == {"aabb", "aaaa", "bbaa", "bbbb"}


def test_recursive_reference_gives_a_nested_language():
    """`(a(?1)b|c)` задаёт `{aⁿ c bⁿ}`, а не `{aⁿcⁿbⁿ}`, как в условии.

    Выражение с одними лишь ссылками на выражение порождает КС-язык,
    а `{aⁿcⁿbⁿ}` не контекстно-свободен, так что в условии опечатка.
    Каркас здесь точен: ссылок на строку и проверок нет.
    """
    grammar = skeleton_cfg(parse_extended("(a(?1)b|c)"))
    inside = {w for w in iter_words("abc", 5) if recognize(grammar, w)}
    assert inside == {"c", "acb", "aacbb"}
    assert not recognize(grammar, "aaccbb")


def test_skeleton_is_an_upper_bound_when_a_string_reference_is_present():
    """`(a|b)\\1` требует одинаковых букв, каркас же допускает любые.

    Каркас — приближение сверху: `False` доказывает непринадлежность,
    `True` не доказывает ничего.
    """
    node = parse_extended(r"(a|b)\1")
    assert matches(node, "ab") is True  # каркас шире языка
    assert matches(node, "aa") is True
    assert matches(node, "abb") is False  # а вот это уже доказано


def test_lookahead_is_erased_in_the_skeleton():
    node = parse_extended("(?=a)ab")
    assert matches(node, "ab") is True
    assert matches(node, "b") is False


def test_skeleton_grammar_is_well_formed():
    grammar = skeleton_cfg(parse_extended("((?:a(?2)|(bb))(?1))"))
    assert grammar.start == "S"
    assert "G1" in grammar.nonterminals and "G2" in grammar.nonterminals
    assert grammar.terminals == frozenset("ab")
