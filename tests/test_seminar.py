"""Тесты по разбору семинара 05.09.2026 (issue #44 репозитория курса).

Отчёт — `reports/seminar-2026-09-05/report.md`. Здесь закреплены выводы,
которые получены исполнением, чтобы они не разошлись с кодом.

Главная содержательная находка: наивная раскодирующая система для
контекстно-зависимой кодировки **не конфлюэнтна**, и критическую пару
(`ada`) находит оракул, а не глаз.
"""

from __future__ import annotations

import pytest

from tfl.srs import parse_srs
from tfl.words import iter_words

CODE = "acdaabcbaddbdb"

NAIVE = """
ab -> (b
ac -> (c
ad -> (d
da -> )a
db -> )b
dc -> )c
"""

COMPLETED = NAIVE + """
a) -> ()
d( -> )(
"""

WITH_END = COMPLETED + """
a$ -> ($
d$ -> )$
"""


def decode(code: str) -> str:
    """Позиционное декодирование: a перед не-a — это «(», d перед не-d — «)»."""
    out = []
    for i, ch in enumerate(code):
        nxt = code[i + 1] if i + 1 < len(code) else ""
        if ch == "a":
            out.append("a" if nxt == "a" else "(")
        elif ch == "d":
            out.append("d" if nxt == "d" else ")")
        else:
            out.append(ch)
    return "".join(out)


def balanced(word: str) -> bool:
    depth = 0
    for ch in word:
        depth += (ch == "(") - (ch == ")")
        if depth < 0:
            return False
    return depth == 0


def normal_forms(system, word: str) -> set[str]:
    return set(system.normal_forms(word, max_len=len(word) + 5).words)


# --------------------------------------------------------------------------
# Задача 1
# --------------------------------------------------------------------------


def test_homomorphic_decoding_is_balanced():
    """Простейший ответ: ( → a, ) → d, остальное само."""
    plain = CODE.replace("a", "(").replace("d", ")")
    assert plain == "(c)((bcb())b)b"
    assert balanced(plain)
    assert set(plain) - set("()") == {"b", "c"}  # два других типа термов


def test_context_decoding_keeps_a_and_d_as_terms():
    """Выигрыш контекстной кодировки: a и d остаются термами."""
    result = decode(CODE)
    assert result == "(c)a(bcb(d)b)b"
    assert balanced(result)
    assert set(result) - set("()") == {"a", "b", "c", "d"}


def test_context_decoding_has_nesting():
    """`(d)` лежит внутри `(bcb(d)b)` — вложенность, а не просто две пары."""
    result = decode(CODE)
    depth, deepest = 0, 0
    for ch in result:
        depth += (ch == "(") - (ch == ")")
        deepest = max(deepest, depth)
    assert deepest == 2


# --------------------------------------------------------------------------
# Задача 2
# --------------------------------------------------------------------------


def test_naive_system_terminates():
    assert parse_srs(NAIVE).terminates(max_len=8, context=1).value is True


def test_naive_system_is_not_confluent():
    """Критическая пара `ada`: правила `ad → (d` и `da → )a` спорят за `d`."""
    verdict = parse_srs(NAIVE).locally_confluent(max_len=10)
    assert verdict.value is False
    critical = verdict.witness[0]
    assert critical == "ada"


def test_naive_system_has_many_ambiguous_words():
    system = parse_srs(NAIVE)
    ambiguous = [w for w in iter_words("abcd", 4) if len(normal_forms(system, w)) > 1]
    assert len(ambiguous) > 20
    assert "ada" in ambiguous


def test_completion_restores_confluence():
    """Два правила — `a) → ()` и `d( → )(` — закрывают все критические пары."""
    system = parse_srs(COMPLETED)
    assert len(system) == 8
    assert system.terminates(max_len=8, context=1).value is True
    assert system.locally_confluent(max_len=12).value is True


def test_completed_system_is_unambiguous():
    system = parse_srs(COMPLETED)
    assert all(len(normal_forms(system, w)) == 1 for w in iter_words("abcd", 5))


def test_end_marker_matches_positional_decoding():
    """Без маркера конца последний символ переписать нечем.

    `ad` пополненная система приводит к `(d`, а позиционное декодирование
    даёт `()`: у последней буквы нет следующей. Маркер `$` — тот же приём,
    что в задании 52-Б с `^w$`.
    """
    system = parse_srs(WITH_END)
    assert system.locally_confluent(max_len=12).value is True
    for word in iter_words("abcd", 5):
        assert normal_forms(system, word + "$") == {decode(word) + "$"}


def test_not_every_string_is_encodable():
    """Кодируемость — ограничение на соседние пары, значит язык регулярен."""
    allowed = {
        "a": set("a("),
        "d": set("d)"),
        "(": set(")bcd"),
        ")": set("(abc"),
        "b": set("abcd()"),
        "c": set("abcd()"),
    }

    def encodable_by_pairs(word: str) -> bool:
        if not word:
            return True
        if word[-1] not in set("()bc"):
            return False
        return all(word[i + 1] in allowed[word[i]] for i in range(len(word) - 1))

    def round_trips(word: str) -> bool:
        return decode(word.replace("(", "a").replace(")", "d")) == word

    assert all(
        encodable_by_pairs(w) == round_trips(w) for w in iter_words("abcd()", 5)
    )
    # Прямое следствие: подряд идущие открывающие скобки закодировать нельзя.
    assert not round_trips("((")
    assert not round_trips("ab")
    assert round_trips("(c)a(bcb(d)b)b")


# --------------------------------------------------------------------------
# Задачи 3 и 4
# --------------------------------------------------------------------------


def rotate(word: str) -> str:
    return word[1:] + word[0] if word else word


def test_rotation_is_injective_but_not_a_morphism():
    images = {}
    for word in iter_words("ab", 9):
        assert rotate(word) not in images, "циклический сдвиг обязан быть инъективным"
        images[rotate(word)] = word
    assert rotate("ab") != rotate("a") + rotate("b")


def test_rotation_needs_the_last_symbol():
    """Первая буква оригинала — последняя буква кода."""
    left, right = "abbbb", "bbbbb"
    assert rotate(left)[:-1] == rotate(right)[:-1]
    assert left[0] != right[0]


MORPHISM = {"x": "0", "y": "01", "z": "11"}


def morph(word: str) -> str:
    return "".join(MORPHISM[c] for c in word)


def test_morphism_is_injective_on_short_words():
    images = {}
    for word in iter_words("xyz", 8):
        assert morph(word) not in images, f"коллизия на {word}"
        images[morph(word)] = word


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_morphism_delay_grows_without_bound(k):
    """Общий префикс образов растёт как 2k, а первые буквы прообраза разные."""
    left, right = "x" + "z" * k, "y" + "z" * (k - 1)
    a, b = morph(left), morph(right)
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    assert common == 2 * k
    assert left[0] != right[0]


# --------------------------------------------------------------------------
# Группа 52-Б
# --------------------------------------------------------------------------


def step_pattern(word: str) -> set[str]:
    """Один шаг системы образцов `aXb → bXa`, `Xb → aaX`."""
    out = set()
    for i, left in enumerate(word):
        if left == "a":
            for j in range(i + 1, len(word)):
                if word[j] == "b":
                    out.add(word[:i] + "b" + word[i + 1 : j] + "a" + word[j + 1 :])
    for j, mid in enumerate(word):
        if mid == "b":
            for i in range(j + 1):
                out.add(word[:i] + "aa" + word[i:j] + word[j + 1 :])
    return out - {word}


def measure(word: str) -> int:
    return word.count("a") + 2 * word.count("b")


def test_pattern_measure_is_invariant():
    """μ(w) = |w|_a + 2|w|_b сохраняется обоими правилами."""
    for word in iter_words("ab", 5):
        assert all(measure(nxt) == measure(word) for nxt in step_pattern(word))


def test_pattern_system_normal_form_is_unique():
    """Нормальная форма — a^μ(w), значит система конфлюэнтна."""
    for word in iter_words("ab", 4):
        if not word:
            continue
        seen, queue = {word}, [word]
        while queue:
            current = queue.pop()
            for nxt in step_pattern(current):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        forms = {w for w in seen if not step_pattern(w)}
        assert forms == {"a" * measure(word)}


def step_brackets(word: str) -> set[str]:
    """Один шаг правила `(X) → X`: удалить любую `(` и любую `)` правее неё."""
    return {
        word[:i] + word[i + 1 : j] + word[j + 1 :]
        for i, left in enumerate(word)
        if left == "("
        for j in range(i + 1, len(word))
        if word[j] == ")"
    }


def reduces_to_empty(word: str) -> bool:
    seen, queue = {word}, [word]
    while queue:
        current = queue.pop()
        if not current:
            return True
        for nxt in step_brackets(current):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def test_bracket_rule_empties_exactly_the_dyck_words():
    """`w →* ε` тогда и только тогда, когда `w` — правильная скобочная."""
    for word in iter_words("()", 9):
        assert reduces_to_empty(word) == balanced(word), word
