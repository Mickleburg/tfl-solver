"""Автоматы Мили и Мура: преобразователи и алгоритмы над ними.

Первым пунктом в списке «что освежить к лекции 12 сентября» стоят
«автоматы Мили (и алгоритмы над ними) из ДМ». Здесь они и лежат вместе
с тем набором алгоритмов, который в ДМ и разбирают: минимизация
разбиением, проверка эквивалентности с **различающим словом**, перевод
Мили ↔ Мура.

## Чем они отличаются от распознавателей

Распознаватель отвечает «да/нет» на всё слово, преобразователь выдаёт
выходное слово. У **Мили** выход висит на переходе, у **Мура** —
на состоянии.

Разница не косметическая, и её стоит помнить при переводе: автомат Мура
на слове длины `n` выдаёт `n+1` символ (первый — выход начального
состояния, ещё до чтения), автомат Мили — ровно `n`. Поэтому
`Moore.to_mealy` теряет начальный символ, а `Mealy.to_moore` его
добавляет, и `run` у них согласованы только с точностью до этого сдвига.
Проверка стоит тестом, а не оговоркой.

## Что здесь считается

* `minimize` — разбиение по Муру: состояния сначала делятся по «реакции
  на одну букву», потом дробятся, пока разбиение меняется. Для Мили
  начальное разбиение — по строке выходов, для Мура — по выходу
  состояния.
* `distinguishing` — кратчайшее слово, на котором два автомата выдают
  разное. Это и есть содержательный ответ: не «не эквивалентны»,
  а конкретное слово, как и в `tfl.automata.counterexample`.
* `to_dfa` — язык слов, после которых выход попал в заданное множество.
  Он регулярен по построению, и через него преобразователь связывается
  со всем остальным проектом.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tfl.automata import DFA

__all__ = ["Mealy", "Moore"]


def _refine(states, alphabet, initial, successor):
    """Разбиение по Муру: дробить классы, пока разбиение меняется."""
    colour = dict(initial)
    while True:
        signature = {
            state: (colour[state], tuple(colour[successor(state, x)] for x in alphabet))
            for state in states
        }
        groups: dict = {}
        for state in states:
            groups.setdefault(signature[state], []).append(state)
        fresh = {}
        for number, key in enumerate(sorted(groups, key=repr)):
            for state in groups[key]:
                fresh[state] = number
        if fresh == colour:
            return colour
        colour = fresh


@dataclass(frozen=True)
class Mealy:
    """Автомат Мили: выход на переходе."""

    alphabet: str
    start: object
    delta: dict[tuple[object, str], object]
    output: dict[tuple[object, str], str]

    def __post_init__(self) -> None:
        missing = set(self.delta) - set(self.output)
        if missing:
            raise ValueError(f"у переходов нет выхода: {sorted(missing, key=repr)[:3]}")

    @property
    def states(self) -> list:
        found = {self.start}
        for source, _ in self.delta:
            found.add(source)
        found |= set(self.delta.values())
        return sorted(found, key=repr)

    def __len__(self) -> int:
        return len(self.states)

    def is_complete(self) -> bool:
        return all(
            (state, letter) in self.delta
            for state in self.states
            for letter in self.alphabet
        )

    def run(self, word: str) -> str:
        """Выходное слово. Длина совпадает с длиной входа."""
        current, out = self.start, []
        for letter in word:
            if (current, letter) not in self.delta:
                raise KeyError(f"перехода из {current!r} по «{letter}» нет")
            out.append(self.output[(current, letter)])
            current = self.delta[(current, letter)]
        return "".join(out)

    def reachable(self) -> list:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            current = queue.popleft()
            for letter in self.alphabet:
                nxt = self.delta.get((current, letter))
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return sorted(seen, key=repr)

    def trim(self) -> Mealy:
        """Выбросить недостижимые состояния."""
        alive = set(self.reachable())
        return Mealy(
            self.alphabet,
            self.start,
            {k: v for k, v in self.delta.items() if k[0] in alive},
            {k: v for k, v in self.output.items() if k[0] in alive},
        )

    def minimize(self) -> Mealy:
        """Минимизация разбиением: класс задаётся реакцией на слова.

        Начальное разбиение — по строке выходов на однобуквенных словах;
        дальше классы дробятся, пока меняется разбиение. Недостижимые
        состояния выбрасываются заранее: они на язык не влияют,
        а классы плодят.
        """
        machine = self.trim()
        states = machine.states
        initial = {
            state: tuple(machine.output.get((state, x)) for x in machine.alphabet)
            for state in states
        }
        numbered = {key: n for n, key in enumerate(sorted(set(initial.values()), key=repr))}
        colour = _refine(
            states,
            machine.alphabet,
            {state: numbered[initial[state]] for state in states},
            lambda state, letter: machine.delta[(state, letter)],
        )
        delta = {}
        output = {}
        for state in states:
            for letter in machine.alphabet:
                if (state, letter) in machine.delta:
                    delta[(colour[state], letter)] = colour[machine.delta[(state, letter)]]
                    output[(colour[state], letter)] = machine.output[(state, letter)]
        return Mealy(machine.alphabet, colour[machine.start], delta, output)

    def distinguishing(self, other: Mealy) -> str | None:
        """Кратчайшее слово, на котором автоматы выдают разное. None — равны.

        Обход в ширину по парам состояний: если пара выдала разные
        выходы по букве, слово найдено; иначе пара переходит в пару.
        """
        if self.alphabet != other.alphabet:
            raise ValueError("автоматы над разными алфавитами")
        start = (self.start, other.start)
        seen = {start}
        queue = deque([(start, "")])
        while queue:
            (mine, yours), word = queue.popleft()
            for letter in self.alphabet:
                if self.output.get((mine, letter)) != other.output.get((yours, letter)):
                    return word + letter
                nxt = (self.delta.get((mine, letter)), other.delta.get((yours, letter)))
                if nxt[0] is None and nxt[1] is None:
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, word + letter))
        return None

    def equivalent(self, other: Mealy) -> bool:
        return self.distinguishing(other) is None

    def to_moore(self) -> Moore:
        """Перевод в автомат Мура: состояние помнит, каким выходом в него вошли.

        Состояния — пары `(состояние, выход)`. Начальное помечается
        «выхода нет» (`None`) нарочно: до первого чтения автомат Мили
        ничего не выдаёт, и если приписать начальному состоянию настоящий
        символ, выходное слово станет длиннее на единицу — то есть это
        будет уже другой преобразователь, а не тот же в другой записи.
        """
        outputs = sorted({value for value in self.output.values()})
        states = {(self.start, None)}
        queue = deque([(self.start, None)])
        delta: dict[tuple[object, str], object] = {}
        label: dict[object, str | None] = {(self.start, None): None}
        while queue:
            current = queue.popleft()
            base = current[0]
            for letter in self.alphabet:
                if (base, letter) not in self.delta:
                    continue
                nxt = (self.delta[(base, letter)], self.output[(base, letter)])
                delta[(current, letter)] = nxt
                if nxt not in states:
                    states.add(nxt)
                    label[nxt] = nxt[1]
                    queue.append(nxt)
        return Moore(self.alphabet, (self.start, None), delta, label, tuple(outputs))

    def to_dfa(self, accepting: str) -> DFA:
        """Язык слов, после которых выход попал в `accepting`.

        Состояния ДКА — пары «куда пришли, с каким выходом»; стартовое
        отдельное, у него выхода ещё нет. Пустое слово в язык не входит
        никогда: выхода до чтения не было.

        Именно эта конструкция связывает преобразователь с распознавателями
        и показывает, что язык регулярен.
        """
        machine = self.trim()
        start = ("старт", None)
        delta: dict[tuple[object, str], object] = {}
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            base = machine.start if current == start else current[0]
            for letter in machine.alphabet:
                if (base, letter) not in machine.delta:
                    continue
                target = (machine.delta[(base, letter)], machine.output[(base, letter)])
                delta[(current, letter)] = target
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        finals = frozenset(
            state for state in seen if state != start and state[1] in accepting
        )
        return DFA(frozenset(machine.alphabet), start, finals, delta)

    def to_dot(self, name: str = "mealy") -> str:
        lines = [f"digraph {name} {{", "  rankdir=LR;"]
        for state in self.states:
            lines.append(f'  "{state}" [shape=circle];')
        lines.append('  __start [shape=point];')
        lines.append(f'  __start -> "{self.start}";')
        for (source, letter), target in sorted(self.delta.items(), key=repr):
            mark = self.output[(source, letter)]
            lines.append(f'  "{source}" -> "{target}" [label="{letter}/{mark}"];')
        lines.append("}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Moore:
    """Автомат Мура: выход на состоянии."""

    alphabet: str
    start: object
    delta: dict[tuple[object, str], object]
    label: dict[object, str | None]
    outputs: tuple[str, ...] = ()

    @property
    def states(self) -> list:
        return sorted(self.label, key=repr)

    def __len__(self) -> int:
        return len(self.label)

    def run(self, word: str) -> str:
        """Выходное слово. Длина на единицу больше входа: первый символ —
        выход начального состояния, выданный до всякого чтения."""
        current = self.start
        out = [self.label[current]]
        for letter in word:
            if (current, letter) not in self.delta:
                raise KeyError(f"перехода из {current!r} по «{letter}» нет")
            current = self.delta[(current, letter)]
            out.append(self.label[current])
        return "".join("" if symbol is None else symbol for symbol in out)

    def to_mealy(self) -> Mealy:
        """Перевод в автомат Мили: выход состояния переезжает на входящие рёбра.

        Начальный символ при этом теряется — у Мили его негде выдать.
        """
        output = {
            (source, letter): self.label[target]
            for (source, letter), target in self.delta.items()
        }
        return Mealy(self.alphabet, self.start, dict(self.delta), output)

    def minimize(self) -> Moore:
        """Минимизация разбиением: начальные классы — по выходу состояния."""
        alive = {self.start}
        queue = deque([self.start])
        while queue:
            current = queue.popleft()
            for letter in self.alphabet:
                nxt = self.delta.get((current, letter))
                if nxt is not None and nxt not in alive:
                    alive.add(nxt)
                    queue.append(nxt)
        states = sorted(alive, key=repr)
        marks = sorted({self.label[state] for state in states}, key=repr)
        colour = _refine(
            states,
            self.alphabet,
            {state: marks.index(self.label[state]) for state in states},
            lambda state, letter: self.delta[(state, letter)],
        )
        delta = {}
        label = {}
        for state in states:
            label[colour[state]] = self.label[state]
            for letter in self.alphabet:
                if (state, letter) in self.delta:
                    delta[(colour[state], letter)] = colour[self.delta[(state, letter)]]
        return Moore(self.alphabet, colour[self.start], delta, label, self.outputs)

    def to_dfa(self, accepting: str) -> DFA:
        """Язык слов, после которых автомат стоит в состоянии с нужным выходом."""
        finals = frozenset(
            state for state in self.states if (self.label[state] or "") in accepting
        )
        return DFA(frozenset(self.alphabet), self.start, finals, dict(self.delta))

    def to_dot(self, name: str = "moore") -> str:
        lines = [f"digraph {name} {{", "  rankdir=LR;"]
        for state in self.states:
            mark = self.label[state]
            shown = "" if mark is None else f"/{mark}"
            lines.append(f'  "{state}" [shape=circle,label="{state}{shown}"];')
        lines.append("  __start [shape=point];")
        lines.append(f'  __start -> "{self.start}";')
        for (source, letter), target in sorted(self.delta.items(), key=repr):
            lines.append(f'  "{source}" -> "{target}" [label="{letter}"];')
        lines.append("}")
        return "\n".join(lines)
