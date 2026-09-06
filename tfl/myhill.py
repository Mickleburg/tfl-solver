"""Таблица классов эквивалентности и нижние оценки размера автомата.

Два инструмента, которые в курсе требуют предъявлять почти всегда:

* **Таблица классов по Майхиллу–Нероуду** — обоснование минимальности ДКА.
  Строки — префиксы-представители классов, столбцы — различающие суффиксы,
  в клетке `+`, если `префикс+суффикс ∈ L`. Минимальность доказана, если
  все строки попарно различны (в норме получается `+` только на диагонали).

* **Обманывающее множество (fooling set)** — нижняя оценка размера
  *недетерминированного* автомата по теореме Глайстера–Шаллита.
  Уточнённая формулировка из `consa_rk2_2024.pdf`: если найдены N префиксов
  γ₁..γ_N и N суффиксов ω₁..ω_N такие, что γ_i ω_i ∈ L, а γ_j ω_i ∉ L при
  j ≠ i, то в любом НКА не меньше N состояний.

Оценка **односторонняя**: найденное множество даёт честную нижнюю границу,
но её недостижимость не доказывает, что НКА меньше не бывает. Преподаватель
это оговаривает отдельно: «бывают языки с неточной оценкой КЭ НКА».
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

from tfl.automata import DFA, State
from tfl.words import iter_words

__all__ = [
    "ClassTable",
    "class_table",
    "shortest_prefixes",
    "distinguishing_suffix",
    "fooling_set",
    "extended_fooling_set",
    "nfa_lower_bound",
    "FoolingSet",
]


# --------------------------------------------------------------------------
# Таблица классов эквивалентности
# --------------------------------------------------------------------------


@dataclass
class ClassTable:
    """Таблица классов: строки — префиксы, столбцы — суффиксы."""

    prefixes: list[str]
    suffixes: list[str]
    cells: list[list[bool]]

    @property
    def is_minimal_proof(self) -> bool:
        """Все ли строки попарно различны — то есть доказана ли минимальность."""
        rows = [tuple(row) for row in self.cells]
        return len(set(rows)) == len(rows)

    def duplicate_rows(self) -> list[tuple[str, str]]:
        """Пары префиксов, которые таблица не сумела различить."""
        seen: dict[tuple[bool, ...], str] = {}
        dupes: list[tuple[str, str]] = []
        for prefix, row in zip(self.prefixes, self.cells):
            key = tuple(row)
            if key in seen:
                dupes.append((seen[key], prefix))
            else:
                seen[key] = prefix
        return dupes

    def to_markdown(self, empty: str = "ε") -> str:
        """Markdown-таблица для вставки в отчёт."""

        def show(word: str) -> str:
            return empty if word == "" else word

        header = "| | " + " | ".join(show(s) for s in self.suffixes) + " |"
        sep = "|---" * (len(self.suffixes) + 1) + "|"
        lines = [header, sep]
        for prefix, row in zip(self.prefixes, self.cells):
            marks = " | ".join("+" if c else "−" for c in row)
            lines.append(f"| **{show(prefix)}** | {marks} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def shortest_prefixes(dfa: DFA) -> dict[State, str]:
    """Кратчайшее слово, ведущее в каждое достижимое состояние.

    Обход в ширину с сортировкой символов даёт шортлекс-минимальные
    представители — те самые, что естественно выписать в отчёте.
    """
    reps: dict[State, str] = {dfa.start: ""}
    queue = deque([dfa.start])
    letters = sorted(dfa.alphabet)
    while queue:
        cur = queue.popleft()
        for ch in letters:
            nxt = dfa.delta.get((cur, ch))
            if nxt is not None and nxt not in reps:
                reps[nxt] = reps[cur] + ch
                queue.append(nxt)
    return reps


def distinguishing_suffix(dfa: DFA, left: State, right: State) -> str | None:
    """Кратчайший суффикс, различающий два состояния; None — неразличимы.

    Обход в ширину по парам состояний: пара считается различённой, когда
    ровно одно из состояний финально.
    """
    full = dfa.complete()
    start = (left, right)
    seen = {start}
    queue = deque([(start, "")])
    letters = sorted(full.alphabet)
    while queue:
        (a, b), word = queue.popleft()
        if (a in full.finals) != (b in full.finals):
            return word
        for ch in letters:
            nxt = (full.delta[(a, ch)], full.delta[(b, ch)])
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, word + ch))
    return None


def class_table(dfa: DFA, include_trap: bool = True) -> ClassTable:
    """Построить таблицу классов эквивалентности минимального ДКА.

    Суффиксы подбираются жадно: пока какие-то две строки совпадают, берём
    различающий их суффикс и добавляем столбец. Так столбцов получается
    ровно столько, сколько нужно, — лишних в отчёте не будет.

    `include_trap=True` (по умолчанию) считает класс непродолжаемых слов
    полноценным классом Майхилла–Нероуда — он даёт строку из одних минусов.
    Это соответствует определению Σ*/≡_L и тому, как таблицу оформляют
    в сдаваемых работах.
    """
    minimal = dfa.minimize(keep_trap=include_trap)
    reps = shortest_prefixes(minimal)
    states = sorted(reps, key=lambda s: (len(reps[s]), reps[s]))
    prefixes = [reps[s] for s in states]

    suffixes: list[str] = []
    while True:
        rows = {
            s: tuple(minimal.accepts(reps[s] + suf) for suf in suffixes)
            for s in states
        }
        collision: tuple[State, State] | None = None
        seen: dict[tuple[bool, ...], State] = {}
        for s in states:
            key = rows[s]
            if key in seen:
                collision = (seen[key], s)
                break
            seen[key] = s
        if collision is None:
            break
        suffix = distinguishing_suffix(minimal, *collision)
        if suffix is None:  # pragma: no cover — минимальный ДКА так не умеет
            raise AssertionError(
                f"состояния {collision} неразличимы в минимальном автомате"
            )
        suffixes.append(suffix)

    cells = [[minimal.accepts(p + s) for s in suffixes] for p in prefixes]
    return ClassTable(prefixes, suffixes, cells)


# --------------------------------------------------------------------------
# Нижняя оценка размера НКА
# --------------------------------------------------------------------------


@dataclass
class FoolingSet:
    """Обманывающее множество пар (префикс, суффикс).

    `kind` различает две формулировки теоремы:

    * `"symmetric"` — классическая: γ_i ω_j ∉ L при **любом** i ≠ j;
    * `"triangular"` — уточнённая (та, что в `consa_rk2_2024.pdf`):
      условие требуется только для j < i, то есть матрица принадлежности
      нижнетреугольная. Она допускает больше пар и потому обычно даёт
      более сильную оценку.

    Обе дают корректную нижнюю границу на число состояний НКА.
    """

    pairs: list[tuple[str, str]]
    kind: str = "symmetric"

    @property
    def bound(self) -> int:
        """Нижняя граница на число состояний любого НКА для языка."""
        return len(self.pairs)

    def to_markdown(self, empty: str = "ε") -> str:
        def show(word: str) -> str:
            return empty if word == "" else word

        lines = ["| i | γ_i | ω_i |", "|---|---|---|"]
        for i, (prefix, suffix) in enumerate(self.pairs, 1):
            lines.append(f"| {i} | {show(prefix)} | {show(suffix)} |")
        return "\n".join(lines)

    def verify(self, accepts: Callable[[str], bool]) -> bool:
        """Пересчитать определение целиком — дешёвая страховка от ошибки."""
        for i, (xi, yi) in enumerate(self.pairs):
            if not accepts(xi + yi):
                return False
            for j, (xj, _) in enumerate(self.pairs):
                if i == j:
                    continue
                if self.kind == "triangular" and j > i:
                    continue  # требуется только для предшествующих
                if accepts(xj + yi):
                    return False
        return True


def fooling_set(
    dfa: DFA,
    max_suffix_len: int = 6,
    suffixes_per_state: int = 12,
    restarts: int = 40,
    seed: int = 0,
) -> FoolingSet:
    """Построить максимальное найденное обманывающее множество.

    Наивная жадность здесь бесполезна: первый же неудачный выбор суффикса
    ограничивает всё множество. Классический пример — `(a|b)*abb`. Если для
    префикса ε взять суффикс `abb`, то условие γ_j ω_1 ∉ L требует, чтобы
    никакое γ_j·abb не лежало в языке, а туда попадает вообще всё, что
    оканчивается на `abb`. Множество застревает на размере 1.

    Поэтому задача ставится как поиск максимальной клики в графе
    совместимости пар: вершины — кандидаты (γ, ω) с γω ∈ L, ребро —
    выполнение условия в обе стороны. Клика ищется жадно по возрастанию
    степени с воспроизводимыми случайными рестартами: результат — всегда
    корректное обманывающее множество (а значит, честная нижняя оценка),
    даже если не максимальное.
    """
    minimal = dfa.minimize()
    candidates = _candidates(minimal, max_suffix_len, suffixes_per_state)
    n = len(candidates)
    compatible = [[False] * n for _ in range(n)]
    for i in range(n):
        xi, yi = candidates[i]
        for j in range(i + 1, n):
            xj, yj = candidates[j]
            ok = not minimal.accepts(xi + yj) and not minimal.accepts(xj + yi)
            compatible[i][j] = compatible[j][i] = ok

    degree = [sum(row) for row in compatible]
    rng = random.Random(seed)
    best: list[int] = []

    for attempt in range(restarts):
        if attempt == 0:
            order = sorted(range(n), key=lambda i: -degree[i])
        else:
            order = list(range(n))
            rng.shuffle(order)
        clique: list[int] = []
        for i in order:
            if all(compatible[i][j] for j in clique):
                clique.append(i)
        if len(clique) > len(best):
            best = clique

    return FoolingSet([candidates[i] for i in sorted(best)], kind="symmetric")


def extended_fooling_set(
    dfa: DFA,
    max_suffix_len: int = 6,
    suffixes_per_state: int = 12,
    restarts: int = 200,
    seed: int = 0,
) -> FoolingSet:
    """Уточнённая (треугольная) теорема Глайстера–Шаллита.

    Ищется **последовательность** пар, а не множество: достаточно, чтобы
    γ_j ω_i ∉ L для всех предшествующих j < i. Условие одностороннее, поэтому
    пар набирается больше.

    Пример из `consa_rk2_2024.pdf`: для `a(a|b)*a|b(a|b)*b` симметричная
    версия даёт всего 2, а треугольная — 4, что совпадает с размером
    настоящего минимального НКА.
    """

    minimal = dfa.minimize()
    candidates = _candidates(minimal, max_suffix_len, suffixes_per_state)
    n = len(candidates)

    # can_follow[j][i] — можно ли поставить i после j: γ_j ω_i ∉ L.
    can_follow = [[False] * n for _ in range(n)]
    for j, (xj, _) in enumerate(candidates):
        for i, (_, yi) in enumerate(candidates):
            if i != j:
                can_follow[j][i] = not minimal.accepts(xj + yi)

    rng = random.Random(seed)
    best: list[int] = []
    for attempt in range(restarts):
        order = list(range(n))
        if attempt:
            rng.shuffle(order)
        seq: list[int] = []
        for i in order:
            if all(can_follow[j][i] for j in seq):
                seq.append(i)
        if len(seq) > len(best):
            best = seq

    return FoolingSet([candidates[i] for i in best], kind="triangular")


def nfa_lower_bound(dfa: DFA, **kwargs) -> FoolingSet:
    """Лучшая из двух оценок снизу на размер НКА, вместе со свидетелем."""
    symmetric = fooling_set(dfa, **kwargs)
    triangular = extended_fooling_set(dfa, **kwargs)
    return max(symmetric, triangular, key=lambda fs: fs.bound)


def _candidates(
    minimal: DFA, max_suffix_len: int, suffixes_per_state: int
) -> list[tuple[str, str]]:
    """Пары (γ, ω) с γω ∈ L: по нескольку кратчайших суффиксов на класс.

    Больше одного представителя-префикса на класс брать бессмысленно: слова
    из одного класса Майхилла–Нероуда не различает никакой суффикс.
    """
    reps = shortest_prefixes(minimal)
    letters = sorted(minimal.alphabet)
    out: list[tuple[str, str]] = []
    for state in sorted(reps, key=lambda s: (len(reps[s]), reps[s])):
        prefix = reps[state]
        taken = 0
        for suffix in iter_words(letters, max_suffix_len):
            if minimal.run(prefix + suffix) in minimal.finals:
                out.append((prefix, suffix))
                taken += 1
                if taken >= suffixes_per_state:
                    break
    return out
