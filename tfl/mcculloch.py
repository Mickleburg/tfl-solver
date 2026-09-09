"""Нейронные сети Мак-Каллока–Питтса: сеть как автомат.

Слайды 1–2 конспекта лекции 1 курса 2026, они же — начало слайдов
курса 2025 (`corpus/txt/FormalLanguageTheory_2025_Lecture1_Slides.txt`,
«McCulloch & Pitts Neural Networks»).

Сеть состоит из бинарных нейронов. У нейрона `v` заданы множество
возбуждающих входов `E⁺_v`, множество тормозящих `E⁻_v` и порог
активации `θ_v ∈ ℕ`. Обновление **синхронное**:

$$x_v(t+1) = \\begin{cases}
1, & \\sum_{u \\in E^{+}_v} x_u(t) \\geqslant \\theta_v
     \\;\\wedge\\; \\forall u \\in E^{-}_v:\\; x_u(t) = 0,\\\\
0, & \\text{иначе.}
\\end{cases}$$

## Торможение здесь абсолютное

`∀u ∈ E⁻_v : x_u(t) = 0` — **один** активный тормозящий вход глушит
нейрон при любом возбуждении. Это не то же самое, что вычитание
из суммы, и разница видна уже на примере лекции: `N₁ = x ∧ ¬y` при
абсолютном торможении получается сразу, а при вычитающем понадобился бы
подбор веса. В курсе принято абсолютное, так и сделано.

## Синхронность делает из картинки задержку

Схема на слайде выглядит комбинационной: `N₁ = x ∧ ¬y`, `N₂ = y ∧ ¬x`,
`N₃ = N₁ ∨ N₂ = x ⊕ y`. Но обновление синхронное, поэтому `N₃`
в момент `t+1` смотрит на `N₁, N₂` в момент `t`, а те — на входы
в момент `t−1`. **Каждый слой это такт задержки**, и `x ⊕ y` появляется
на выходе через два шага, а не мгновенно. Ловушка закреплена тестом.

## Зачем это в ТФЯ

Состояний у сети конечное число — вектор значений нейронов, — а вход
берётся из конечного алфавита наборов входных битов. Значит сеть это
**автомат Мура** (выход висит на состоянии), и язык слов, после которых
выходной нейрон активен, регулярен. Перевод `to_moore` и `to_dfa`
это и предъявляет: дальше работают все распознаватели проекта.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tfl.automata import DFA
from tfl.mealy import Moore

__all__ = ["Neuron", "Network", "SIGNALS", "xor_network"]

#: Пул символов под наборы входных битов: `k` входов дают `2ᵏ` букв.
SIGNALS = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Neuron:
    """Бинарный нейрон: возбуждающие входы, тормозящие и порог."""

    excitatory: frozenset[str] = frozenset()
    inhibitory: frozenset[str] = frozenset()
    threshold: int = 1

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("порог активации — натуральное число")
        both = self.excitatory & self.inhibitory
        if both:
            raise ValueError(
                f"вход {sorted(both)} и возбуждающий, и тормозящий сразу"
            )

    def fires(self, values: dict[str, int]) -> int:
        """Значение нейрона на следующем такте по значениям на текущем."""
        if any(values[source] for source in self.inhibitory):
            return 0  # торможение абсолютное: хватает одного активного
        return int(sum(values[source] for source in self.excitatory) >= self.threshold)

    def __str__(self) -> str:
        plus = "+".join(sorted(self.excitatory)) or "—"
        minus = "+".join(sorted(self.inhibitory)) or "—"
        return f"θ={self.threshold}, E⁺={{{plus}}}, E⁻={{{minus}}}"


@dataclass(frozen=True)
class Network:
    """Сеть: входные узлы задаются снаружи, нейроны считаются."""

    inputs: tuple[str, ...]
    neurons: dict[str, Neuron] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = set(self.inputs) | set(self.neurons)
        if set(self.inputs) & set(self.neurons):
            raise ValueError("узел не может быть и входом, и нейроном")
        for name, neuron in self.neurons.items():
            outside = (neuron.excitatory | neuron.inhibitory) - known
            if outside:
                raise ValueError(
                    f"у нейрона «{name}» вход из ниоткуда: {''.join(sorted(outside))}"
                )
        if len(self.inputs) > len(SIGNALS).bit_length() + 4 and 2 ** len(self.inputs) > len(SIGNALS):
            raise ValueError(f"входов больше, чем букв под наборы: {len(self.inputs)}")

    @property
    def names(self) -> tuple[str, ...]:
        """Нейроны в устойчивом порядке — это и есть координаты состояния."""
        return tuple(sorted(self.neurons))

    @property
    def alphabet(self) -> str:
        """Буква на каждый набор входных битов."""
        return SIGNALS[: 2 ** len(self.inputs)]

    def decode(self, symbol: str) -> dict[str, int]:
        """Набор входных битов, отвечающий букве. Старший бит — первый вход."""
        number = self.alphabet.index(symbol)
        width = len(self.inputs)
        return {
            name: (number >> (width - 1 - position)) & 1
            for position, name in enumerate(self.inputs)
        }

    def encode(self, bits: dict[str, int]) -> str:
        """Буква, отвечающая набору битов."""
        number = 0
        for name in self.inputs:
            number = number * 2 + (1 if bits.get(name) else 0)
        return self.alphabet[number]

    def step(self, state: tuple[int, ...], symbol: str) -> tuple[int, ...]:
        """Один синхронный такт: все нейроны обновляются одновременно."""
        values = dict(self.decode(symbol))
        values.update(dict(zip(self.names, state)))
        return tuple(self.neurons[name].fires(values) for name in self.names)

    def zero(self) -> tuple[int, ...]:
        """Покоящееся состояние: все нейроны молчат."""
        return tuple(0 for _ in self.names)

    def trace(self, word: str, state: tuple[int, ...] | None = None):
        """Значения нейронов после каждого такта, включая начальное."""
        current = self.zero() if state is None else state
        found = [current]
        for symbol in word:
            current = self.step(current, symbol)
            found.append(current)
        return found

    def value(self, word: str, neuron: str) -> int:
        """Значение нейрона после прочтения слова."""
        return self.trace(word)[-1][self.names.index(neuron)]

    def to_moore(self, output: str) -> Moore:
        """Сеть как автомат Мура: состояние — вектор значений нейронов.

        Достижимых состояний может быть сильно меньше, чем `2ⁿ`, поэтому
        обход идёт от покоящегося, а не по всем векторам подряд.
        """
        if output not in self.neurons:
            raise KeyError(f"нейрона «{output}» в сети нет")
        index = self.names.index(output)
        start = self.zero()
        delta: dict[tuple[object, str], object] = {}
        label: dict[object, str] = {start: str(start[index])}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for symbol in self.alphabet:
                nxt = self.step(current, symbol)
                delta[(current, symbol)] = nxt
                if nxt not in label:
                    label[nxt] = str(nxt[index])
                    queue.append(nxt)
        return Moore(self.alphabet, start, delta, label, ("0", "1"))

    def to_dfa(self, output: str) -> DFA:
        """Язык слов, после которых выходной нейрон активен. Он регулярен."""
        return self.to_moore(output).to_dfa("1")

    def markdown(self) -> str:
        lines = [
            f"Входы: {', '.join(self.inputs)}. Нейронов: {len(self.neurons)}.",
            "",
            "| нейрон | порог | возбуждающие | тормозящие |",
            "|---|---|---|---|",
        ]
        for name in self.names:
            neuron = self.neurons[name]
            lines.append(
                f"| `{name}` | {neuron.threshold} | "
                f"{', '.join(f'`{x}`' for x in sorted(neuron.excitatory)) or '—'} | "
                f"{', '.join(f'`{x}`' for x in sorted(neuron.inhibitory)) or '—'} |"
            )
        return "\n".join(lines)

    def to_dot(self, name: str = "mcculloch") -> str:
        lines = [f"digraph {name} {{", "  rankdir=LR;"]
        for node in self.inputs:
            lines.append(f'  "{node}" [shape=box];')
        for node in self.names:
            lines.append(f'  "{node}" [shape=circle,label="{node}\\nθ={self.neurons[node].threshold}"];')
        for node in self.names:
            neuron = self.neurons[node]
            for source in sorted(neuron.excitatory):
                lines.append(f'  "{source}" -> "{node}";')
            for source in sorted(neuron.inhibitory):
                lines.append(f'  "{source}" -> "{node}" [arrowhead=dot,style=dashed];')
        lines.append("}")
        return "\n".join(lines)


def xor_network() -> Network:
    """Пример со слайда 2: `N₁ = x ∧ ¬y`, `N₂ = y ∧ ¬x`, `N₃ = x ⊕ y`.

    Тормозящие связи абсолютные, пороги единичные. Из-за синхронности
    `N₃` показывает `x ⊕ y` **через два такта** после подачи входа.
    """
    return Network(
        ("x", "y"),
        {
            "N1": Neuron(frozenset({"x"}), frozenset({"y"}), 1),
            "N2": Neuron(frozenset({"y"}), frozenset({"x"}), 1),
            "N3": Neuron(frozenset({"N1", "N2"}), frozenset(), 1),
        },
    )
