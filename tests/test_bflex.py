"""Brainfuck-Рефал и Brainfuck-Лисп: сторона МАТа в ЛР2 2024.

Две приёмочные точки, и обе независимы от того, как собран автомат:

* **вывод по грамматике против автомата.** Программы порождаются
  рекурсивным выводом по правилам, а цель собирается операциями над
  автоматами. Если они расходятся, ошибка видна на конкретном слове;
* **ограничения ТЗ считаются операциями над языками.** Пересечения,
  склейки и конечность проверяются автоматами, а не рассуждением
  о том, как устроен генератор.

Отдельно закреплено найденное: набор лексем может удовлетворять всем
ограничениям ТЗ и при этом давать **вырожденную** цель, где скобочные
конструкции целиком поглощаются языком имён.
"""

from __future__ import annotations

import itertools
import random

import pytest

from tfl.automata import equivalent
from tfl.bflex import (
    LISP,
    REFAL,
    Lexicon,
    alternative,
    automaton_for,
    concat,
    epsilon,
    plus,
    random_lexicon,
    refal_automaton,
    repeat,
    sample_program,
    teacher_for,
    words_automaton,
)
from tfl.lstar import learn


def as_nfa(words, alphabet):
    from tfl.bflex import _as_nfa

    return _as_nfa(words_automaton(words, set(alphabet)))


def language(machine, limit=5, letters="ab"):
    return [
        word
        for length in range(limit + 1)
        for word in ("".join(x) for x in itertools.product(letters, repeat=length))
        if machine.accepts(word)
    ]


# --------------------------------------------------------------------------
# Комбинаторы
# --------------------------------------------------------------------------


def test_words_automaton_accepts_exactly_the_listed_words():
    machine = words_automaton(["ab", "b"], set("ab"))
    assert language(machine) == ["b", "ab"]


def test_concatenation_glues_languages():
    left, right = as_nfa(["ab", "b"], "ab"), as_nfa(["ba"], "ab")
    assert language(concat(left, right).determinize().minimize()) == ["bba", "abba"]


def test_star_and_plus_differ_only_in_the_empty_word():
    machine = as_nfa(["ab", "b"], "ab")
    star = language(repeat(machine).determinize().minimize(), 3)
    more = language(plus(machine).determinize().minimize(), 3)
    assert star[0] == ""
    assert star[1:] == more


def test_alternative_is_the_union():
    left, right = as_nfa(["ab"], "ab"), as_nfa(["ba", "b"], "ab")
    assert language(alternative(left, right).determinize().minimize(), 2) == [
        "b",
        "ab",
        "ba",
    ]


def test_epsilon_accepts_only_the_empty_word():
    assert language(epsilon("ab").determinize().minimize(), 2) == [""]


# --------------------------------------------------------------------------
# Ограничения ТЗ
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [REFAL, LISP])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_generator_satisfies_the_statement(kind, seed):
    verdict = random_lexicon(kind, seed=seed).check()
    assert verdict.value is True, verdict.reason


def broken_lisp(**changes) -> Lexicon:
    machines = dict(random_lexicon(LISP, seed=1).machines)
    machines.update(changes)
    return Lexicon(LISP, machines)


def test_a_shared_alphabet_for_eol_is_refused():
    """> автомат для [eol] имеет алфавит, отличный от всех остальных лексем."""
    verdict = broken_lisp(eol=words_automaton(["31"], set("0123456789"))).check()
    assert verdict.value is False
    assert "алфавит `[eol]`" in verdict.reason


def test_overlapping_brackets_are_refused():
    same = words_automaton(["121"], set("12"))
    verdict = broken_lisp(lbr=same, rbr=same).check()
    assert verdict.value is False
    assert "пересекаются" in verdict.reason


def test_a_bracket_that_is_two_brackets_glued_is_refused():
    """Условие про склейку — не украшение: `1`, `2` и `12` его нарушают.

    Языки `{1}` и `{2}` не пересекаются, но `[lbr][rbr]` даёт `12`,
    а это в точности язык третьей скобки, если её так задать.
    """
    machines = dict(random_lexicon(LISP, seed=1).machines)
    machines["lbr"] = words_automaton(["1", "12"], set("12"))
    machines["rbr"] = words_automaton(["2"], set("12"))
    verdict = Lexicon(LISP, machines).check()
    assert verdict.value is False
    assert "конкатенация" in verdict.reason


def test_a_finite_atom_language_is_refused():
    """> язык лексемы атомов бесконечный."""
    verdict = broken_lisp(atom=words_automaton(["345"], set("3456789"))).check()
    assert verdict.value is False
    assert "конечен" in verdict.reason


def test_an_infinite_separator_is_refused_for_refal():
    """> языки у этих лексем конечные."""
    machines = dict(random_lexicon(REFAL, seed=1).machines)
    from tfl.bflex import _prefixed_words

    machines["sep"] = _prefixed_words("2", "2")
    verdict = Lexicon(REFAL, machines).check()
    assert verdict.value is False
    assert "бесконечен" in verdict.reason


def test_a_missing_lexeme_is_named():
    machines = dict(random_lexicon(REFAL, seed=1).machines)
    del machines["lbr-2"]
    verdict = Lexicon(REFAL, machines).check()
    assert verdict.value is False
    assert "lbr-2" in verdict.reason


# --------------------------------------------------------------------------
# Цель: вывод по грамматике против собранного автомата
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,depth", [(REFAL, 0), (REFAL, 1), (REFAL, 2), (LISP, 0), (LISP, 1), (LISP, 2)]
)
def test_every_derived_program_is_accepted(kind, depth):
    """Главная сверка: вывод по правилам и сборка автоматами сходятся."""
    lexicon = random_lexicon(kind, seed=1)
    machine = automaton_for(lexicon, depth)
    for seed in range(40):
        word = sample_program(lexicon, depth, seed)
        assert machine.accepts(word), (kind, depth, word)


@pytest.mark.parametrize("kind", [REFAL, LISP])
def test_a_single_corrupted_letter_is_almost_always_caught(kind):
    """Порча одной буквы должна ломать разбор — иначе цель слишком щедра."""
    lexicon = random_lexicon(kind, seed=1)
    machine = automaton_for(lexicon, 1)
    rng = random.Random(7)
    caught = 0
    for seed in range(40):
        word = sample_program(lexicon, 1, seed)
        position = rng.randrange(len(word))
        others = sorted(lexicon.alphabet - {word[position]})
        spoiled = word[:position] + rng.choice(others) + word[position + 1 :]
        caught += not machine.accepts(spoiled)
    # Ловятся не все: буква внутри имени, заменённая на другую букву того же
    # подалфавита, оставляет имя именем. Это свойство языка, а не недосмотр.
    assert caught >= 30


def test_deeper_nesting_gives_a_bigger_language():
    lexicon = random_lexicon(LISP, seed=1)
    shallow = automaton_for(lexicon, 0)
    deeper = automaton_for(lexicon, 1)
    assert not equivalent(shallow, deeper)
    assert len(deeper) > len(shallow)


def test_the_builder_refuses_a_lexicon_of_the_other_kind():
    with pytest.raises(ValueError, match="не от Brainfuck-Рефала"):
        refal_automaton(random_lexicon(LISP, seed=1), 1)


def test_negative_depth_is_refused():
    with pytest.raises(ValueError, match="вложенность"):
        automaton_for(random_lexicon(LISP, seed=1), -1)


# --------------------------------------------------------------------------
# Найденное: ТЗ выполнено, а цель вырождена
# --------------------------------------------------------------------------


def degenerate_lisp() -> Lexicon:
    """Набор лексем, законный по ТЗ, у которого имена глотают скобки.

    `[atom]` — слова длины ⩾ 3, начинающиеся на `1` и кончающиеся на `2`.
    Тогда `[lbr][atom][dot][atom][rbr]` = `1` + `1…2` + `33` + `1…2` + `2`
    само начинается на `1` и кончается на `2`, то есть **само является
    атомом**, и скобки в нём не видны.
    """
    from tfl.automata import DFA

    letters = frozenset("123")
    delta = {("q0", "1"): "q1"}
    for char in "123":
        delta[("q1", char)] = "q2"
        delta[("q2", char)] = "q3" if char == "2" else "q2"
        delta[("q3", char)] = "q3" if char == "2" else "q2"
    atom = DFA(letters, "q0", frozenset({"q3"}), delta).minimize()
    return Lexicon(
        LISP,
        {
            "eol": words_automaton(["0"], {"0"}),
            "lbr": words_automaton(["1"], set("12")),
            "rbr": words_automaton(["2"], set("12")),
            "dot": words_automaton(["33"], {"3"}),
            "atom": atom,
        },
    )


def test_the_statement_allows_names_to_swallow_brackets():
    """Ограничения ТЗ выполнены, а скобочная конструкция — сама атом."""
    lexicon = degenerate_lisp()
    assert lexicon.check().value is True
    nested = "1" + "112" + "33" + "112" + "2"  # [lbr][atom][dot][atom][rbr]
    assert automaton_for(lexicon, 1).accepts(nested)
    # И то же слово принимается там, где скобок нет вовсе: это один атом.
    assert automaton_for(lexicon, 0).accepts(nested)


def test_the_generator_keeps_brackets_visible():
    """У генератора имена и скобки живут в разных подалфавитах.

    Поэтому конструкция со скобками на нулевой вложенности отвергается —
    без этого вся лабораторная теряет смысл.
    """
    lexicon = random_lexicon(LISP, seed=1)
    nested = (
        lexicon.words("lbr", 4)[0]
        + lexicon.words("atom", 1)[0]
        + lexicon.words("dot", 4)[0]
        + lexicon.words("atom", 1)[0]
        + lexicon.words("rbr", 4)[0]
    )
    assert automaton_for(lexicon, 1).accepts(nested)
    assert not automaton_for(lexicon, 0).accepts(nested)

    brackets = set("".join(lexicon.words("lbr", 4) + lexicon.words("rbr", 4)))
    names = set("".join(lexicon.words("atom", 2)))
    assert brackets and names and not (brackets & names)


# --------------------------------------------------------------------------
# МАТ и угадыватель
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind,depth", [(LISP, 0), (LISP, 1), (REFAL, 0)])
def test_lstar_learns_the_lexer_automaton(kind, depth):
    """Ради этого МАТ и делается: угадыватель обязан сойтись на цели."""
    lexicon = random_lexicon(kind, seed=1)
    target = automaton_for(lexicon, depth)
    teacher = teacher_for(lexicon, depth)
    result = learn(teacher, "".join(sorted(lexicon.alphabet)), max_rounds=60)
    assert result.converged, result.summary()
    assert equivalent(result.dfa, target)
    assert teacher.equivalence_queries == result.equivalence_queries


def test_the_lexicon_can_be_drawn():
    """> визуализация — автоматы для лексем, и общий автомат для лексера."""
    lexicon = random_lexicon(LISP, seed=1)
    picture = lexicon.to_dot()
    assert picture.count("digraph") == len(lexicon.names)
    assert "digraph" in automaton_for(lexicon, 1).to_dot()
    assert "[atom]" in lexicon.markdown()
