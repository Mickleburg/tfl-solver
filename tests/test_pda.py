"""Тесты магазинных автоматов.

Главная перекрёстная проверка — `test_round_trip_preserves_language`:
грамматика переводится в PDA нисходящей конструкцией, обратно — тройной,
и язык обязан сохраниться. Конструкции устроены совершенно по-разному,
поэтому совпадение не тавтологично.

Вторая проверка того же рода — `test_simulation_agrees_with_earley`:
симуляция автомата сравнивается с алгоритмом Эрли на исходной грамматике.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.cfg import parse_cfg
from tfl.parse import language, recognize
from tfl.pda import PDA, Transition, disagreements, from_cfg, parse_pda, to_cfg
from tfl.words import iter_words

LAB3 = pathlib.Path(__file__).parent.parent / "evals" / "lab3_2025"

# Классический DPDA для aⁿbⁿ, n > 0: допуск по пустому стеку.
BALANCED = """
start: q
stack: Z
accept: empty
q, a, Z -> q, A Z
q, a, A -> q, A A
q, b, A -> r, ε
r, b, A -> r, ε
r, ε, Z -> r, ε
"""

# Тот же язык, но с лишним недетерминизмом: два перехода по `a` из q.
FORKED = """
start: q
stack: Z
accept: empty
q, a, Z -> q, A Z
q, a, Z -> p, A Z
q, b, A -> q, ε
p, b, A -> p, ε
q, ε, Z -> q, ε
"""


def lab3_grammars():
    out = []
    for path in sorted(LAB3.glob("variant-*.cfg")):
        out.append((int(path.stem.split("-")[1]), parse_cfg(path.read_text(encoding="utf-8"))))
    return out


GRAMMARS = lab3_grammars()
IDS = [f"v{n}" for n, _ in GRAMMARS]


# --------------------------------------------------------------------------
# Чтение и печать
# --------------------------------------------------------------------------


def test_parse_pda_reads_header_and_transitions():
    pda = parse_pda(BALANCED)
    assert pda.start == "q"
    assert pda.start_stack == ("Z",)
    assert pda.accept_by == "empty"
    assert len(pda) == 5


def test_stack_chain_without_spaces_is_split_char_wise():
    """`A Z` и `AZ` — одна и та же цепочка, как и в правых частях правил."""
    spaced = parse_pda("start: q\nstack: Z\nq, a, Z -> q, A Z")
    tight = parse_pda("start: q\nstack: Z\nq, a, Z -> q, AZ")
    assert spaced.transitions == tight.transitions


def test_epsilon_tokens_recognized():
    pda = parse_pda("start: q\nstack: Z\nq, ε, Z -> q, ε")
    assert pda.transitions[0].read == ""
    assert pda.transitions[0].push == ()


def test_missing_start_rejected():
    with pytest.raises(ValueError):
        parse_pda("stack: Z\nq, a, Z -> q, Z")


def test_accept_by_state_requires_final_states():
    with pytest.raises(ValueError):
        PDA("q", ("Z",), (), frozenset(), "state")


# --------------------------------------------------------------------------
# Симуляция
# --------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["ab", "aabb", "aaabbb"])
def test_balanced_accepts(word):
    assert parse_pda(BALANCED).accepts(word).value is True


@pytest.mark.parametrize("word", ["", "a", "b", "aab", "abab", "ba"])
def test_balanced_rejects(word):
    assert parse_pda(BALANCED).accepts(word).value is False


def test_rejection_is_definite_when_search_completes():
    """Отказ обязан быть обоснован полным перебором, а не исчерпанием бюджета."""
    verdict = parse_pda(BALANCED).accepts("ba")
    assert verdict.value is False
    assert "перебран" in verdict.reason


def test_tight_budget_yields_unknown_not_false():
    """С урезанным стеком автомат не успевает — и обязан сказать «не выяснено».

    Это ровно та подмена, ради которой заведён `Verdict`: слово в языке,
    но при бюджете 1 принимающий путь недостижим, и `False` здесь был бы
    уверенно выглядящим враньём.
    """
    verdict = parse_pda(BALANCED).accepts("aaabbb", max_stack=1)
    assert verdict.value is None


def test_verdict_is_not_a_bool():
    with pytest.raises(TypeError):
        bool(parse_pda(BALANCED).accepts("ab"))


def test_accepting_path_is_returned_as_witness():
    verdict = parse_pda(BALANCED).accepts("aabb")
    assert all(isinstance(t, Transition) for t in verdict.witness)


# --------------------------------------------------------------------------
# Детерминизм
# --------------------------------------------------------------------------


def test_balanced_is_deterministic():
    assert parse_pda(BALANCED).is_deterministic()


def test_two_transitions_on_same_symbol_are_a_fork():
    conflicts = parse_pda(FORKED).nondeterminism()
    assert any("по символу «a»" in c.reason for c in conflicts)


def test_epsilon_next_to_symbol_transition_is_a_fork():
    """Критерий из лекции 9: есть ε-переход ⇒ других из этого состояния нет."""
    pda = parse_pda("start: q\nstack: Z\nq, a, Z -> q, Z\nq, ε, Z -> q, ε")
    assert not pda.is_deterministic()
    assert "ε-переход" in pda.nondeterminism()[0].reason


def test_incompatible_pops_are_not_a_fork():
    """Переходы, снимающие разные символы, вместе сработать не могут."""
    pda = parse_pda("start: q\nstack: Z\nq, a, Z -> q, Z\nq, a, A -> q, A")
    assert pda.is_deterministic()


def test_prefix_pops_are_a_fork():
    """`Z` — префикс `Z Z`, значит есть стек, к которому применимы оба."""
    pda = parse_pda("start: q\nstack: Z\nq, a, Z -> q, Z\nq, a, Z Z -> q, Z")
    assert not pda.is_deterministic()


def test_cfg_pda_is_not_deterministic():
    """У нисходящего автомата раскрытие нетерминала — всегда ε-развилка."""
    assert not from_cfg(parse_cfg("S -> a S b | ε")).is_deterministic()


# --------------------------------------------------------------------------
# Грамматика ⟷ автомат
# --------------------------------------------------------------------------


def test_from_cfg_shape():
    pda = from_cfg(parse_cfg("S -> a S b | ε"))
    assert len(pda.states) == 1
    assert pda.accept_by == "empty"
    # два правила плюс по переходу на каждый терминал
    assert len(pda) == 4


def test_to_cfg_requires_empty_stack_acceptance():
    pda = PDA("q", ("Z",), (Transition("q", "a", ("Z",), (), "f"),), frozenset({"f"}), "state")
    with pytest.raises(ValueError):
        to_cfg(pda)


def test_to_cfg_of_handmade_dpda():
    """Тройная конструкция на автомате, написанном руками, а не выведенном из грамматики."""
    grammar = to_cfg(parse_pda(BALANCED))
    assert language(grammar, 6) == {"ab", "aabb", "aaabbb"}


@pytest.mark.parametrize(
    "text",
    [
        "S -> a S b | ε",
        "S -> a S b | a b",
        "S -> S a S b | T T | b a b\nT -> b b T | ε",
    ],
)
def test_round_trip_preserves_language(text):
    grammar = parse_cfg(text)
    assert language(to_cfg(from_cfg(grammar)), 6) == language(grammar, 6)


@pytest.mark.parametrize("number,grammar", GRAMMARS, ids=IDS)
def test_simulation_agrees_with_earley(number, grammar):
    """Симуляция автомата против Эрли на всех грамматиках ЛР3."""
    pda = from_cfg(grammar)
    bad, undecided = disagreements(pda, lambda w: recognize(grammar, w), "ab", max_len=5)
    assert not bad, f"вариант {number}: расхождения на {bad}"
    assert not undecided, f"вариант {number}: бюджета не хватило на {undecided}"


def test_disagreements_separates_unknown_from_wrong():
    """Слова без вердикта попадают во второй список, а не в первый.

    Смешать их — значит объявить ошибкой то, что просто не досчиталось.
    """
    pda = parse_pda(BALANCED)
    bad, undecided = disagreements(
        pda, lambda w: True, "ab", max_len=3
    )
    assert bad and not undecided


def test_dot_output_mentions_every_transition():
    dot = parse_pda(BALANCED).to_dot()
    assert dot.count("->") == len(parse_pda(BALANCED)) + 1  # переходы плюс стрелка старта


def test_iter_words_covers_alphabet():
    """Страховка от опечатки в алфавите тестов выше."""
    assert set(iter_words("ab", 1)) == {"", "a", "b"}


# --------------------------------------------------------------------------
# Real-time свойство и слияние снимаемых символов
# --------------------------------------------------------------------------
#
# Авторский разбор РК2 (`2023/RK2_probe_tasks_solutions`) на задаче
# про real-time формулирует и ограничение, и приём:
#
# > А сбросить сразу несколько символов за один шаг мы не можем
# > по определению. Остаётся ввести символы M2, соответствующие парным M
# > <…> Слияние нескольких снимаемых подряд стековых символов и введение
# > символа, находящегося непосредственно перед дном — довольно типичные
# > приёмы построения таких автоматов.
#
# Ниже три автомата для одного языка `{a²ⁿbⁿ}`, различающиеся ровно тем,
# какой ценой они обходят это ограничение.


def _double_a_language(word: str) -> bool:
    letters = word.count("a")
    return (
        word == "a" * letters + "b" * (len(word) - letters)
        and letters % 2 == 0
        and len(word) - letters == letters // 2
    )


def multi_pop_pda() -> PDA:
    """Кладёт `A` на каждую `a` и снимает две за шаг — так курс не разрешает."""
    return PDA("q", (), (
        Transition("q", "a", (), ("A",), "q"),
        Transition("q", "b", ("A", "A"), (), "r"),
        Transition("r", "b", ("A", "A"), (), "r"),
    ), frozenset({"q", "r"}), "both")


def merged_pda() -> PDA:
    """Слияние: `A` кладётся на каждую вторую `a`, снимается по одному."""
    return PDA("qe", (), (
        Transition("qe", "a", (), (), "qo"),
        Transition("qo", "a", (), ("A",), "qe"),
        Transition("qe", "b", ("A",), (), "qb"),
        Transition("qb", "b", ("A",), (), "qb"),
    ), frozenset({"qe", "qb"}), "both")


def epsilon_pop_pda() -> PDA:
    """Второй символ снимается ε-переходом — real-time теряется."""
    return PDA("q", (), (
        Transition("q", "a", (), ("A",), "q"),
        Transition("q", "b", ("A",), (), "r1"),
        Transition("r1", "", ("A",), (), "r"),
        Transition("r", "b", ("A",), (), "r1"),
    ), frozenset({"q", "r"}), "both")


@pytest.mark.parametrize(
    "build", [multi_pop_pda, merged_pda, epsilon_pop_pda], ids=["multi", "merged", "eps"]
)
def test_all_three_recognise_the_same_language(build):
    """Автоматы различаются устройством, а не языком — иначе сравнивать нечего."""
    machine = build()
    for word in iter_words("ab", 9):
        assert (machine.accepts(word).value is True) == _double_a_language(word), word


def test_merging_buys_both_properties_at_once():
    """Слияние даёт и real-time, и снятие по одному символу."""
    machine = merged_pda()
    assert machine.is_deterministic()
    assert machine.is_real_time()
    assert machine.multi_pop_transitions() == []


def test_multi_pop_is_real_time_but_outside_the_definition():
    machine = multi_pop_pda()
    assert machine.is_real_time()
    assert len(machine.multi_pop_transitions()) == 2


def test_epsilon_pop_loses_real_time():
    """Снять второй символ ε-переходом можно, но автомат перестаёт быть real-time."""
    machine = epsilon_pop_pda()
    assert not machine.is_real_time()
    assert machine.multi_pop_transitions() == []
    assert len(machine.epsilon_transitions()) == 1
    assert machine.is_deterministic()  # DPDA, но не Real-Time DPDA
