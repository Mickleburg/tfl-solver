"""Лабиринты из ЛР2 2024: язык путей, выводящих наружу.

Вариант «планарный лабиринт» (слайд 8) со всеми словами условия:

> Входной автомат — произвольный планарный граф с бинарным ветвлением.
> Все тупиковые ветки — выходы из лабиринта. Путь в лабиринте представляет
> собой последовательность инструкций в алфавите `{L, R}`: указывающих
> на каждой развилке, идти направо или налево. Путь принадлежит языку,
> если строго выводит к выходу… Если путь остаётся в лабиринте или содержит
> дополнительные инструкции после попадания в финальное состояние,
> считается, что он языку не принадлежит.

Отсюда автомат получается сразу: состояния — узлы, у развилки два перехода,
выходы принимающие, а любая буква после выхода ведёт в ловушку («содержит
дополнительные инструкции после попадания в финальное состояние»).

Угадыватель знает только оценки сверху на число развилок и выходов
(`parameters.txt`), сам лабиринт — за МАТом. Поэтому генератор здесь тоже
есть: он и играет роль МАТа в связке с `tfl/lstar.py`.

**Ортогональный лабиринт (слайд 9) не реализован, и на то есть причина.**
При буквальном чтении условия язык не регулярен, а значит, и ДКА, который
угадывают, не существует. Снаружи прямоугольника стен нет, ходить там
можно свободно, вернуться внутрь — только через проём; поэтому две разные
внешние клетки различаются словом «дойти до проёма и зайти», и классов
Майхилла–Нероуда бесконечно много. Нужен либо запрет возвращаться, либо
ограничение на блуждание снаружи — в условии этого нет. Записано
в `docs/OPEN-GAPS.md` как вопрос преподавателю.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tfl.automata import DFA

__all__ = ["PlanarMaze", "random_planar_maze", "LEFT", "RIGHT", "TRAP"]

LEFT, RIGHT = "L", "R"
ALPHABET = LEFT + RIGHT

#: Состояние «шаг после выхода» — из него уже ничего не принимается.
TRAP = "ловушка"


@dataclass(frozen=True)
class PlanarMaze:
    """Планарный лабиринт: развилки с двумя ходами и тупики-выходы."""

    forks: dict[int, tuple[int, int]]
    exits: frozenset[int]
    start: int = 0

    def __post_init__(self) -> None:
        nodes = set(self.forks) | set(self.exits)
        if self.start not in nodes:
            raise ValueError(f"стартовый узел {self.start} не описан")
        overlap = set(self.forks) & set(self.exits)
        if overlap:
            raise ValueError(f"узлы {sorted(overlap)} и развилки, и выходы сразу")
        for node, (left, right) in self.forks.items():
            missing = {left, right} - nodes
            if missing:
                raise ValueError(f"развилка {node} ведёт в неописанные {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.forks) + len(self.exits)

    def walk(self, path: str) -> int | str | None:
        """Куда приводит путь. `TRAP` — лишние инструкции после выхода."""
        node: int | str = self.start
        for letter in path:
            if node == TRAP or node in self.exits:
                return TRAP
            if letter not in ALPHABET:
                return None
            left, right = self.forks[node]
            node = left if letter == LEFT else right
        return node

    def accepts(self, path: str) -> bool:
        """Путь строго выводит к выходу."""
        return self.walk(path) in self.exits

    def to_dfa(self) -> DFA:
        delta: dict[tuple[object, str], object] = {}
        for node, (left, right) in self.forks.items():
            delta[(node, LEFT)] = left
            delta[(node, RIGHT)] = right
        for node in self.exits:
            for letter in ALPHABET:
                delta[(node, letter)] = TRAP
        for letter in ALPHABET:
            delta[(TRAP, letter)] = TRAP
        return DFA(frozenset(ALPHABET), self.start, frozenset(self.exits), delta)

    def markdown(self) -> str:
        """Описание лабиринта — то, что МАТ обязан визуализировать."""
        lines = [f"Старт: {self.start}. Развилок {len(self.forks)}, "
                 f"выходов {len(self.exits)}.", ""]
        for node in sorted(self.forks):
            left, right = self.forks[node]
            mark = lambda n: f"**{n}** (выход)" if n in self.exits else str(n)  # noqa: E731
            lines.append(f"- развилка {node}: `L` → {mark(left)}, `R` → {mark(right)}")
        return "\n".join(lines)


def random_planar_maze(
    max_forks: int, max_exits: int, seed: int = 0, tries: int = 200
) -> PlanarMaze:
    """Сгенерировать лабиринт в пределах оценок сверху — работа МАТа.

    > Угадыватель заранее знает оценку сверху на число ветвлений в лабиринте
    > и число выходов из него… По этому же файлу MAT генерирует граф,
    > используя данные числа как оценку сверху.

    Строится дерево развилок (это и обеспечивает планарность), затем часть
    ходов заворачивается назад — иначе лабиринт получается без циклов
    и выучивается слишком легко. Все узлы обязаны быть достижимы
    и хотя бы один выход — тоже, иначе генерация повторяется.
    """
    if max_forks < 1 or max_exits < 1:
        raise ValueError("развилок и выходов должно быть хотя бы по одному")
    rng = random.Random(seed)
    for _ in range(tries):
        # Условие разрешает лабиринт меньше оценки, но совсем мелкий
        # ничего не проверяет, поэтому размер берётся из верхней половины.
        forks_count = rng.randint(max(1, max_forks // 2), max_forks)
        exits_count = rng.randint(1, max_exits)
        fork_nodes = list(range(forks_count))
        exit_nodes = list(range(forks_count, forks_count + exits_count))
        targets = fork_nodes + exit_nodes

        forks: dict[int, tuple[int, int]] = {}
        for node in fork_nodes:
            # Ходы вперёд по дереву плюс, изредка, назад — для циклов.
            forward = [t for t in targets if t > node] or exit_nodes
            pair = [
                rng.choice(forward) if rng.random() < 0.75 else rng.choice(targets)
                for _ in range(2)
            ]
            forks[node] = (pair[0], pair[1])

        maze = PlanarMaze(forks, frozenset(exit_nodes))
        reachable = _reachable(maze)
        if reachable >= set(forks) | set(exit_nodes):
            return maze
    raise RuntimeError(
        f"за {tries} попыток не вышло построить связный лабиринт "
        f"в пределах {max_forks} развилок и {max_exits} выходов"
    )


def _reachable(maze: PlanarMaze) -> set[int]:
    seen = {maze.start}
    frontier = [maze.start]
    while frontier:
        node = frontier.pop()
        if node in maze.exits:
            continue
        for following in maze.forks[node]:
            if following not in seen:
                seen.add(following)
                frontier.append(following)
    return seen
