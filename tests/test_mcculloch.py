"""Сети Мак-Каллока–Питтса: сеть как автомат.

Две вещи, ради которых файл и написан.

* **Торможение абсолютное.** Один активный тормозящий вход глушит нейрон
  при любом возбуждении — это не вычитание из суммы, и разница
  проверяется на конкретных наборах.
* **Синхронность — это задержка.** Схема со слайда 2 выглядит
  комбинационной, но каждый слой стоит такта: `x ⊕ y` появляется
  на выходе через два шага. Ловушка закреплена явно, потому что именно
  на ней проще всего ошибиться, читая картинку.

Язык сети сверяется с независимо построенным ДКА, а не с самим собой.
"""

from __future__ import annotations

import itertools

import pytest

from tfl.automata import DFA, equivalent
from tfl.mcculloch import Network, Neuron, xor_network


def words(alphabet: str, limit: int):
    for length in range(limit + 1):
        for letters in itertools.product(alphabet, repeat=length):
            yield "".join(letters)


# --------------------------------------------------------------------------
# Нейрон
# --------------------------------------------------------------------------


def test_inhibition_is_absolute_and_not_subtractive():
    """Три возбуждения при пороге 1 глушатся одним торможением."""
    neuron = Neuron(frozenset("abc"), frozenset("z"), 1)
    assert neuron.fires({"a": 1, "b": 1, "c": 1, "z": 0}) == 1
    assert neuron.fires({"a": 1, "b": 1, "c": 1, "z": 1}) == 0


def test_the_threshold_is_a_sum_over_excitatory_inputs():
    neuron = Neuron(frozenset("abc"), frozenset(), 2)
    assert neuron.fires({"a": 1, "b": 0, "c": 0}) == 0
    assert neuron.fires({"a": 1, "b": 1, "c": 0}) == 1


def test_a_zero_threshold_neuron_fires_on_its_own():
    """Порог ноль — сумма всегда его достигает, и нейрон активен всегда."""
    assert Neuron(frozenset(), frozenset(), 0).fires({}) == 1
    assert Neuron(frozenset(), frozenset("z"), 0).fires({"z": 1}) == 0


def test_a_contradictory_input_is_refused():
    with pytest.raises(ValueError, match="тормозящий сразу"):
        Neuron(frozenset("a"), frozenset("a"))
    with pytest.raises(ValueError, match="натуральное"):
        Neuron(threshold=-1)


# --------------------------------------------------------------------------
# Сеть
# --------------------------------------------------------------------------


def test_a_dangling_input_is_named():
    with pytest.raises(ValueError, match="вход из ниоткуда"):
        Network(("x",), {"N": Neuron(frozenset({"q"}))})


def test_a_node_cannot_be_both_input_and_neuron():
    with pytest.raises(ValueError, match="и входом, и нейроном"):
        Network(("x",), {"x": Neuron()})


def test_the_alphabet_covers_every_combination_of_input_bits():
    net = xor_network()
    assert net.alphabet == "0123"
    assert net.decode("2") == {"x": 1, "y": 0}
    assert all(net.encode(net.decode(s)) == s for s in net.alphabet)


def test_the_update_is_synchronous():
    """Все нейроны читают значения **предыдущего** такта, а не соседа сейчас."""
    net = xor_network()
    after = net.step(net.zero(), "2")  # x = 1, y = 0
    assert dict(zip(net.names, after)) == {"N1": 1, "N2": 0, "N3": 0}
    later = net.step(after, "0")
    assert dict(zip(net.names, later)) == {"N1": 0, "N2": 0, "N3": 1}


def test_xor_appears_two_ticks_later_and_not_at_once():
    """Схема со слайда выглядит комбинационной, но каждый слой — такт."""
    net = xor_network()
    for first in net.alphabet:
        bits = net.decode(first)
        expected = bits["x"] ^ bits["y"]
        assert net.value(first, "N3") == 0, "мгновенного ответа быть не может"
        assert net.value(first + "0", "N3") == expected, first


# --------------------------------------------------------------------------
# Сеть как автомат
# --------------------------------------------------------------------------


def delayed_xor_dfa(alphabet: str) -> DFA:
    """Независимый распознаватель: предпоследняя буква имеет `x ≠ y`.

    Строится по описанию языка, а не по сети, — иначе сверка была бы
    сверкой сети с самой собой.
    """
    marks = {"0": "0", "1": "1", "2": "1", "3": "0"}  # x ≠ y
    states = ["", "0", "1", "00", "01", "10", "11"]
    delta = {}
    for state in states:
        for letter in alphabet:
            delta[(state, letter)] = (state + marks[letter])[-2:]
    finals = frozenset(s for s in states if len(s) == 2 and s[0] == "1")
    return DFA(frozenset(alphabet), "", finals, delta)


def test_the_language_of_the_network_matches_an_independent_recogniser():
    net = xor_network()
    assert equivalent(net.to_dfa("N3"), delayed_xor_dfa(net.alphabet))


def test_the_automaton_agrees_with_the_simulation_word_by_word():
    net = xor_network()
    machine = net.to_dfa("N3")
    for word in words(net.alphabet, 5):
        assert machine.accepts(word) is (net.value(word, "N3") == 1), word


def test_only_reachable_state_vectors_appear():
    """Достижимых векторов меньше, чем `2³`: `N₁` и `N₂` вместе не горят."""
    net = xor_network()
    moore = net.to_moore("N3")
    assert len(moore) < 2 ** len(net.names)
    assert all(not (state[0] and state[1]) for state in moore.states)


def test_a_missing_output_neuron_is_named():
    with pytest.raises(KeyError, match="N9"):
        xor_network().to_moore("N9")


def test_a_memory_cell_gives_a_non_trivial_language():
    """Нейрон с обратной связью помнит: язык — «был хоть один сигнал».

    Условие `x + N ⩾ 1` включается от входа и держится само, поэтому
    автомат ровно на два состояния. Задержки здесь нет ни на такт —
    слой один, а тактом стоит **каждый слой**, а не сама синхронность.
    """
    net = Network(("x",), {"N": Neuron(frozenset({"x", "N"}), frozenset(), 1)})
    machine = net.to_dfa("N").minimize()
    assert len(machine) == 2
    for word in words("01", 6):
        assert machine.accepts(word) is ("1" in word), word


def test_the_network_can_be_drawn_and_printed():
    net = xor_network()
    assert "digraph" in net.to_dot()
    assert "arrowhead=dot" in net.to_dot()  # тормозящие рисуются иначе
    assert "| `N1` |" in net.markdown()
