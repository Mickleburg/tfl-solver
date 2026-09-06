"""Тесты КС-грамматик и разбора.

Главная перекрёстная проверка — `test_recognizer_agrees_with_enumeration`:
принадлежность считается двумя независимыми способами (алгоритм Эрли и
неподвижная точка по выводимым словам) и на всех 14 грамматиках ЛР3 они
обязаны совпасть.

Числовые контрольные точки взяты из классики: у грамматики арифметических
выражений ровно 12 состояний LR(0), она SLR(1), но не LL(1) и не LR(0).
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.cfg import CFG, Production, parse_cfg
from tfl.parse import cyk, derivation, language, prefix_free, recognize
from tfl.words import iter_words

LAB3 = pathlib.Path(__file__).parent.parent / "evals" / "lab3_2025"

EXPRESSION = "E -> E + T | T\nT -> T * F | F\nF -> ( E ) | i"
DANGLING_ELSE = "S -> i S e S | i S | a"


def lab3_grammars() -> list[tuple[int, CFG]]:
    out = []
    for path in sorted(LAB3.glob("variant-*.cfg")):
        number = int(path.stem.split("-")[1])
        out.append((number, parse_cfg(path.read_text(encoding="utf-8"))))
    return out


GRAMMARS = lab3_grammars()
IDS = [f"v{n}" for n, _ in GRAMMARS]


# --------------------------------------------------------------------------
# Разбор текста грамматики
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arrow", ["->", "→", "::=", "-->", "=>"])
def test_arrows(arrow):
    assert parse_cfg(f"S {arrow} a").productions == (Production("S", ("a",)),)


def test_alternatives_split():
    grammar = parse_cfg("S -> a S b | ε")
    assert len(grammar) == 2
    assert Production("S", ()) in grammar.productions


def test_spacing_does_not_matter():
    """`S → b T a a T` и `S → bTaaT` — одно и то же правило.

    В условиях курса встречаются обе записи, иногда в одном задании.
    """
    spaced = parse_cfg("S -> b T a a T\nT -> a")
    tight = parse_cfg("S -> bTaaT\nT -> a")
    assert spaced.productions == tight.productions


def test_terminal_runs_are_split_into_symbols():
    """`S → ab S bb S` из варианта 25 — это отдельные буквы, не токены `ab`, `bb`."""
    grammar = parse_cfg("S -> ab S bb S\nS -> ε")
    assert grammar.terminals == frozenset("ab")
    assert grammar.productions[0].rhs == ("a", "b", "S", "b", "b", "S")


def test_primed_nonterminal_stays_one_symbol():
    grammar = parse_cfg("S -> a S' b\nS' -> ε")
    assert "S'" in grammar.nonterminals
    assert grammar.productions[0].rhs == ("a", "S'", "b")


def test_missing_arrow_rejected():
    with pytest.raises(ValueError):
        parse_cfg("S a b")


# --------------------------------------------------------------------------
# Чистка и nullable
# --------------------------------------------------------------------------


def test_unproductive_removed():
    grammar = parse_cfg("S -> a | A\nA -> A b")
    assert "A" not in grammar.clean().nonterminals


def test_unreachable_removed():
    grammar = parse_cfg("S -> a\nB -> b")
    assert "B" not in grammar.clean().nonterminals


def test_clean_order_matters():
    """Сначала непорождающие, потом недостижимые.

    `B` порождающий, но достижим только через непорождающий `A`, так что
    после первой чистки становится мусором. В обратном порядке он бы остался.
    """
    grammar = parse_cfg("S -> a\nS -> A B\nA -> A a\nB -> b")
    cleaned = grammar.clean()
    assert cleaned.nonterminals == frozenset({"S"})


def test_empty_language_grammar():
    assert parse_cfg("S -> S a").clean().productions == ()


def test_nullable():
    grammar = parse_cfg("S -> A B\nA -> ε\nB -> b | ε")
    assert grammar.nullable() == {"S", "A", "B"}


# --------------------------------------------------------------------------
# First / Follow
# --------------------------------------------------------------------------


def test_first_follow_of_expression_grammar():
    """Контрольные значения из классического учебника."""
    grammar = parse_cfg(EXPRESSION)
    first = grammar.first_k(1)
    assert {t[0] for t in first["E"]} == {"(", "i"}
    assert {t[0] for t in first["T"]} == {"(", "i"}
    assert {t[0] for t in first["F"]} == {"(", "i"}

    follow = grammar.follow_k(1)
    assert {t[0] for t in follow["E"]} == {"$", "+", ")"}
    assert {t[0] for t in follow["T"]} == {"$", "+", ")", "*"}
    assert {t[0] for t in follow["F"]} == {"$", "+", ")", "*"}


def test_first_k_longer_than_one():
    grammar = parse_cfg("S -> a b c")
    assert grammar.first_k(2)["S"] == {("a", "b")}


def test_first_contains_epsilon_as_empty_tuple():
    assert () in parse_cfg("S -> ε | a").first_k(1)["S"]


# --------------------------------------------------------------------------
# LL(1)
# --------------------------------------------------------------------------


def test_simple_grammar_is_ll1():
    assert parse_cfg("S -> a S b | ε").is_ll1()


def test_expression_grammar_is_not_ll1():
    grammar = parse_cfg(EXPRESSION)
    assert not grammar.is_ll1()
    assert grammar.ll1_table()[1], "конфликты должны быть перечислены"


def test_direct_left_recursion_detected():
    assert parse_cfg(EXPRESSION).left_recursive() == {"E", "T"}


def test_indirect_left_recursion_detected():
    """Косвенная рекурсия `A → B c`, `B → A d` тоже должна находиться."""
    grammar = parse_cfg("A -> B c | a\nB -> A d | b")
    assert grammar.left_recursive() == {"A", "B"}


def test_left_recursion_through_nullable_prefix():
    """`A → B A c` с аннулируемым `B` — это тоже левая рекурсия."""
    grammar = parse_cfg("A -> B A c | a\nB -> ε")
    assert "A" in grammar.left_recursive()


# --------------------------------------------------------------------------
# LR(0) / SLR(1)
# --------------------------------------------------------------------------


def test_expression_grammar_has_twelve_lr0_states():
    """Классическая контрольная точка из учебника."""
    states, _ = parse_cfg(EXPRESSION).lr0_states()
    assert len(states) == 12


def test_expression_grammar_is_slr1_but_not_lr0():
    grammar = parse_cfg(EXPRESSION)
    assert not grammar.is_lr0()
    assert grammar.is_slr1()


def test_dangling_else_has_shift_reduce_conflict():
    conflicts = parse_cfg(DANGLING_ELSE).slr1_conflicts()
    assert any(c.kind == "shift/reduce" for c in conflicts)


def test_augmented_start_is_fresh():
    grammar, extra = parse_cfg("S -> a").augmented()
    assert extra.lhs not in {"S"}
    assert grammar.start == extra.lhs


def test_epsilon_rule_breaks_lr0_but_not_slr1():
    """`S → aSb | ε`: свёртка ε конфликтует со сдвигом, но Follow их разводит."""
    grammar = parse_cfg("S -> a S b | ε")
    assert not grammar.is_lr0()
    assert grammar.is_slr1()


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_earley_on_balanced_pairs():
    grammar = parse_cfg("S -> a S b | ε")
    assert recognize(grammar, "")
    assert recognize(grammar, "aaabbb")
    assert not recognize(grammar, "aab")
    assert not recognize(grammar, "ba")


def test_earley_handles_left_recursion():
    """Левая рекурсия зацикливает наивный спуск, но не Эрли."""
    grammar = parse_cfg(EXPRESSION)
    assert recognize(grammar, "i+i*i")
    assert not recognize(grammar, "i+")


def test_earley_handles_cyclic_grammar():
    """`S → S | a` — цикл в выводе; распознаватель обязан завершаться."""
    assert recognize(parse_cfg("S -> S | a"), "a")


def test_language_enumeration():
    grammar = parse_cfg("S -> a S b | ε")
    assert language(grammar, 6) == {"", "ab", "aabb", "aaabbb"}


def test_language_of_empty_grammar():
    assert language(parse_cfg("S -> S a"), 5) == set()


def test_derivation_is_returned_and_valid():
    grammar = parse_cfg("S -> a S b | ε")
    steps = derivation(grammar, "aabb")
    assert steps is not None
    assert steps[0] == "S" and steps[-1] == "aabb"


def test_derivation_absent_for_nonmember():
    assert derivation(parse_cfg("S -> a S b | ε"), "abab") is None


# --------------------------------------------------------------------------
# Перекрёстная проверка на грамматиках ЛР3
# --------------------------------------------------------------------------


def test_all_14_lab3_grammars_extracted():
    assert len(GRAMMARS) == 14
    assert [n for n, _ in GRAMMARS] == list(range(1, 28, 2))


@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_grammar_alphabet_is_ab(number, grammar):
    """Все грамматики ЛР3 2025 — над {a, b}; иное означает сбой извлечения."""
    assert grammar.terminals <= frozenset("ab"), f"вариант {number}"
    assert grammar.nonterminals <= frozenset({"S", "T"}), f"вариант {number}"


@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_recognizer_agrees_with_enumeration(number, grammar):
    """Эрли против неподвижной точки — два независимых способа.

    Эрли идёт от слова к грамматике, перечисление — от грамматики к словам.
    Расхождение означает ошибку в одном из них.
    """
    words = set(language(grammar, 7))
    for word in iter_words("ab", 7):
        assert recognize(grammar, word) == (word in words), (
            f"вариант {number}: расхождение на «{word or 'ε'}»"
        )


@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_cleaning_preserves_language(number, grammar):
    """Чистка убирает бесполезные нетерминалы, но не меняет язык."""
    assert language(grammar.clean(), 7) == language(grammar, 7), f"вариант {number}"


def test_epsilon_rules_do_not_get_lost_in_earley():
    """Регрессия: приём Айкока–Хорспула для аннулируемых нетерминалов.

    Грамматика варианта 27 ЛР3. Пустое слово выводится как `S → TT` при
    `T → ε` дважды. Наивный Эрли его терял: завершение `T → •` происходит
    в том же столбце, и ситуация `S → T•T`, добавленная позже, оставалась
    непродвинутой.

    Ошибку нашла перекрёстная проверка с перечислением языка — сама по себе
    реализация выглядела правдоподобно.
    """
    grammar = parse_cfg("S -> S a S b | T T | b a b\nT -> b b T | ε")
    assert recognize(grammar, "")
    assert "" in language(grammar, 3)


def test_minimal_nullable_chain():
    """Ещё короче: `S → A B`, оба нетерминала аннулируемы."""
    grammar = parse_cfg("S -> A B\nA -> ε\nB -> ε")
    assert recognize(grammar, "")


# --------------------------------------------------------------------------
# Нормальная форма Хомского и алгоритм Кока–Янгера–Касами
# --------------------------------------------------------------------------


def test_epsilon_removal_keeps_the_empty_word():
    """Регрессия: `S₀ → ε` терялся из-за того, что nullable не пересчитывался.

    После выноса стартового нетерминала наружу (`S₀ → S`) новый стартовый
    тоже аннулируем, но в старом множестве nullable его нет — и правило
    `S₀ → ε` не добавлялось, а вместе с ним из языка пропадало пустое слово.
    """
    grammar = parse_cfg("S -> a S b | ε").remove_epsilon()
    assert "" in language(grammar, 4)
    assert language(grammar, 6) == {"", "ab", "aabb", "aaabbb"}


def test_epsilon_removal_drops_all_other_epsilon_rules():
    grammar = parse_cfg("S -> A B\nA -> a | ε\nB -> b | ε").remove_epsilon()
    empty = [p for p in grammar.productions if not p.rhs]
    assert [p.lhs for p in empty] == [grammar.start]


def test_epsilon_free_grammar_gets_no_fresh_start():
    grammar = parse_cfg("S -> a S | a")
    assert grammar.remove_epsilon().start == "S"


def test_unit_rules_removed():
    grammar = parse_cfg("S -> A\nA -> B\nB -> b").remove_unit()
    assert all(not (len(p.rhs) == 1 and p.rhs[0] in grammar.nonterminals)
               for p in grammar.productions)
    assert language(grammar, 3) == {"b"}


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | ε",
        "S -> A\nA -> B\nB -> b",
        EXPRESSION,
        "S -> S a S b | T T | b a b\nT -> b b T | ε",
    ],
)
def test_chomsky_normal_form_shape_and_language(text):
    grammar = parse_cfg(text)
    normal = grammar.chomsky_normal_form()
    assert normal.is_chomsky_normal_form()
    assert language(normal, 6) == language(grammar, 6)


def test_pair_nonterminals_are_shared_between_rules():
    """Одна и та же пара символов не должна плодить разные нетерминалы.

    В `S → abc | xabc` пара `⟨b⟩⟨c⟩` возникает дважды. Без общей таблицы
    вторая получит имя вроде `⟨⟨b⟩⟨c⟩⟩1`, и грамматика распухнет на ровном
    месте. Совпадение тела с телом правила исходного нетерминала при этом
    нормально и проверкой не считается.
    """
    grammar = parse_cfg("S -> a b c | x a b c")
    normal = grammar.chomsky_normal_form()
    added = set(normal.nonterminals) - set(grammar.nonterminals)
    bodies = [p.rhs for p in normal.productions if len(p.rhs) == 2 and p.lhs in added]
    assert len(bodies) == len(set(bodies))
    assert len(bodies) == 2  # ⟨b⟩⟨c⟩ и ⟨a⟩⟨⟨b⟩⟨c⟩⟩


@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_cyk_agrees_with_earley(number, grammar):
    """Третий независимый распознаватель против первого.

    Эрли идёт сверху вниз по исходной грамматике, CYK — снизу вверх
    по нормальной форме Хомского. Совпадение проверяет заодно и само
    приведение к нормальной форме.
    """
    normal = grammar.chomsky_normal_form()
    for word in iter_words("ab", 6):
        assert cyk(grammar, word, normal) == recognize(grammar, word), (
            f"вариант {number}: расхождение на «{word or 'ε'}»"
        )


# --------------------------------------------------------------------------
# Беспрефиксность
# --------------------------------------------------------------------------


def test_prefix_free_refuted_with_witness():
    verdict = prefix_free(parse_cfg("S -> a S b | ε"), 6)
    assert verdict.value is False
    short, long = verdict.witness
    assert long.startswith(short) and short != long


def test_prefix_free_is_never_proved():
    """«Да» среди исходов нет: беспрефиксность КС-языка неразрешима.

    Язык aⁿbⁿ (n > 0) беспрефиксен, но подтвердить это перебором нельзя,
    и честный ответ — «не выяснено».
    """
    assert prefix_free(parse_cfg("S -> a S b | a b"), 8).value is None
