"""Пары зависимостей для строковых систем переписывания.

Метод Арца и Гизля, приложенный к одноместной сигнатуре: слово
`w₁w₂…wₙ` — это терм `w₁(w₂(…wₙ(x)))`, поэтому у него ровно один корень
(первая буква) и ровно одна переменная (хвост).

## Зачем

Прямая интерпретация требует, чтобы **каждое** правило строго убывало.
Это дорого: одно неудобное правило хоронит весь поиск. Пары зависимостей
меняют требование на пару более слабых:

* все правила убывают **нестрого** (`⩾`);
* строго убывает **хотя бы одна пара** в каждой циклической компоненте
  графа зависимостей.

Разница не косметическая. Нестрогое убывание не требует монотонности,
поэтому коэффициенты можно брать нулевыми, а строгое требуется только
там, где действительно закручивается рекурсия.

## Как строятся пары

`D` — множество **определённых** символов: первые буквы левых частей.
Для правила `l → r` парой зависимостей объявляется

    ⟨l♯, (rᵢ…rₙ)♯⟩   для каждого i, у которого rᵢ ∈ D,

то есть каждый суффикс правой части, начинающийся с определённой буквы.
Диез помечает **корень**: правила системы диезов не содержат, поэтому
переписывать помеченную букву нельзя, и шаги цепочки идут строго ниже
корня. На этом всё и держится.

## Граф зависимостей

Ребро из ⟨l, v⟩ в ⟨l′, v′⟩ проводится, если `v` с каким-то хвостом может
переписаться во что-то, начинающееся с `l′`. Точно это неразрешимо,
поэтому берётся стандартная оценка сверху: `v` обрезается на первой
определённой букве **после корня** (`cap`), и остаток сравнивается
с `l′` на общем префиксе. Оценка сверху безопасна: лишние рёбра могут
помешать доказать завершимость, но не могут доказать её ложно.

Система завершима, если в графе нет бесконечных цепочек. Достаточно
разобрать каждую нетривиальную компоненту сильной связности отдельно:
найти редукционную пару, выкинуть строго убывшие пары и повторить
на том, что осталось.

## Редукционная пара

Букве сопоставляется аффинное отображение `ℕ^d → ℕ^d`, `x ↦ Ax + b`,
слову — композиция; помеченной корневой букве — своё отображение.
При `d = 1` это обычная линейная интерпретация `x ↦ mx + c`.

* нестрогое убывание `l ⩾ r`: `A_l ⩾ A_r` и `b_l ⩾ b_r` покоординатно;
* строгое убывание пары: то же плюс `b_l[0] > b_r[0]`.

Все элементы неотрицательны, поэтому отображения слабо монотонны,
и нестрогое убывание переносится в контекст. Строгий порядок
фундирован: значения лежат в ℕ.

**Поиск и проверка разделены**, как и в `tfl/matrix.py`. Поиск идёт
через Z3 и без него честно возвращает «не выяснено»; проверка
(`DependencyProof.check`) считает целыми числами и от решателя
не зависит вовсе. Наружу выдаётся только проверенное доказательство.

## Что метод не делает

Опровергать. `prove_termination` возвращает либо «доказано», либо
«не выяснено»: неудача поиска не значит, что система не завершима.
Опровержение — это петля, и она ищется отдельно (`SRS.find_loop`).

Замер на вариантах ЛР1 2025 — `python tools/lab1_termination.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from tfl.verdict import Verdict, proved, unknown

__all__ = [
    "MARK",
    "brute_force_pair",
    "DependencyPair",
    "Affine",
    "ReductionPair",
    "Step",
    "DependencyProof",
    "defined_symbols",
    "dependency_pairs",
    "cap",
    "estimated_graph",
    "cycles",
    "find_reduction_pair",
    "prove_termination",
    "have_solver",
]

#: Диез, помечающий корневую букву. В правилах системы его нет никогда.
MARK = "♯"


# --------------------------------------------------------------------------
# Пары зависимостей и граф
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DependencyPair:
    """Пара `⟨l♯, v♯⟩`: `v` — суффикс правой части с определённой буквой."""

    lhs: str
    rhs: str
    source: str

    def __str__(self) -> str:
        return f"{self.lhs}{MARK} → {self.rhs}{MARK}"


def defined_symbols(system) -> frozenset[str]:
    """Первые буквы левых частей — корни редексов."""
    return frozenset(rule.lhs[0] for rule in system.rules if rule.lhs)


def dependency_pairs(system) -> tuple[DependencyPair, ...]:
    """Все пары зависимостей системы.

    Суффикс берётся **со всякой** определённой позиции, включая первую:
    правило `ab → ba` даёт и `ab♯ → ba♯`, и `ab♯ → a♯`, если обе буквы
    определены.
    """
    defined = defined_symbols(system)
    found = []
    for rule in system.rules:
        for index, letter in enumerate(rule.rhs):
            if letter in defined:
                found.append(
                    DependencyPair(rule.lhs, rule.rhs[index:], str(rule))
                )
    return tuple(found)


def cap(word: str, defined: frozenset[str]) -> str:
    """Обрезать слово на первой определённой букве **после корня**.

    Это `CAP` из оценки графа зависимостей: всё, что стоит под
    определённой буквой, может переписаться во что угодно, поэтому
    заменяется переменной, а сравнивать остаётся только начало.
    """
    for index in range(1, len(word)):
        if word[index] in defined:
            return word[:index]
    return word


def _compatible(left: str, right: str) -> bool:
    """Совпадают ли слова на общем префиксе — то есть унифицируются ли."""
    shared = min(len(left), len(right))
    return left[:shared] == right[:shared]


def estimated_graph(
    pairs: tuple[DependencyPair, ...], defined: frozenset[str]
) -> dict[int, list[int]]:
    """Оценка графа зависимостей сверху: рёбра по совпадению начал."""
    return {
        index: [
            other
            for other, target in enumerate(pairs)
            if _compatible(cap(pair.rhs, defined), target.lhs)
        ]
        for index, pair in enumerate(pairs)
    }


def cycles(graph: dict[int, list[int]]) -> list[tuple[int, ...]]:
    """Нетривиальные компоненты сильной связности — алгоритм Тарьяна.

    Нетривиальная — та, в которой есть цикл: либо больше одной вершины,
    либо петля. Только такие и могут дать бесконечную цепочку.
    """
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    found: list[tuple[int, ...]] = []
    counter = 0

    for root in graph:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work: list[tuple[int, object]] = [(root, iter(graph[root]))]
        while work:
            node, walker = work[-1]
            descended = False
            for nxt in walker:  # type: ignore[union-attr]
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(graph[nxt])))
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    top = stack.pop()
                    on_stack.discard(top)
                    component.append(top)
                    if top == node:
                        break
                if len(component) > 1 or node in graph.get(node, ()):
                    found.append(tuple(sorted(component)))
    return found


# --------------------------------------------------------------------------
# Редукционная пара
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Affine:
    """Отображение `ℕ^d → ℕ^d`, `x ↦ Ax + b`, с неотрицательными элементами."""

    matrix: tuple[tuple[int, ...], ...]
    shift: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.shift)

    @staticmethod
    def identity(dimension: int) -> Affine:
        return Affine(
            tuple(
                tuple(1 if i == j else 0 for j in range(dimension))
                for i in range(dimension)
            ),
            tuple(0 for _ in range(dimension)),
        )

    def then(self, inner: Affine) -> Affine:
        """Композиция `self ∘ inner`: сначала `inner`, потом `self`."""
        size = self.dimension
        matrix = tuple(
            tuple(
                sum(self.matrix[i][k] * inner.matrix[k][j] for k in range(size))
                for j in range(size)
            )
            for i in range(size)
        )
        shift = tuple(
            self.shift[i]
            + sum(self.matrix[i][k] * inner.shift[k] for k in range(size))
            for i in range(size)
        )
        return Affine(matrix, shift)

    def above(self, other: Affine) -> bool:
        """Покоординатное `⩾`: значит `self(x) ⩾ other(x)` при всех `x ⩾ 0`."""
        return all(
            self.matrix[i][j] >= other.matrix[i][j]
            for i in range(self.dimension)
            for j in range(self.dimension)
        ) and all(
            self.shift[i] >= other.shift[i] for i in range(self.dimension)
        )

    def strictly_above(self, other: Affine) -> bool:
        """Ещё и `self(x)[0] > other(x)[0]` при всех `x ⩾ 0`."""
        return self.above(other) and self.shift[0] > other.shift[0]

    def __str__(self) -> str:
        if self.dimension == 1:
            slope, free = self.matrix[0][0], self.shift[0]
            body = "x" if slope == 1 else ("0" if slope == 0 else f"{slope}x")
            if free or body == "0":
                body = f"{body} + {free}" if body != "0" else str(free)
            return f"x ↦ {body}"
        rows = "; ".join(" ".join(str(v) for v in row) for row in self.matrix)
        return f"x ↦ [{rows}]·x + [{' '.join(str(v) for v in self.shift)}]"


@dataclass(frozen=True)
class ReductionPair:
    """Интерпретация букв: обычная и для помеченного корня."""

    plain: dict[str, Affine]
    marked: dict[str, Affine]
    dimension: int

    def value(self, word: str) -> Affine:
        """Значение слова: композиция букв слева направо."""
        result = Affine.identity(self.dimension)
        for letter in reversed(word):
            result = self.plain[letter].then(result)
        return result

    def value_marked(self, word: str) -> Affine:
        """То же, но корневая буква взята из помеченной интерпретации."""
        if not word:
            raise ValueError("помеченное слово не может быть пустым")
        return self.marked[word[0]].then(self.value(word[1:]))

    def weakly_decreases(self, left: str, right: str) -> bool:
        return self.value(left).above(self.value(right))

    def pair_decreases(self, pair: DependencyPair, strict: bool) -> bool:
        left = self.value_marked(pair.lhs)
        right = self.value_marked(pair.rhs)
        return left.strictly_above(right) if strict else left.above(right)

    def markdown(self) -> str:
        lines = ["| буква | `[·]` | `[·♯]` |", "|---|---|---|"]
        for letter in sorted(self.plain):
            lines.append(f"| `{letter}` | {self.plain[letter]} | {self.marked[letter]} |")
        return "\n".join(lines)


@dataclass(frozen=True)
class Step:
    """Один шаг разбора: компонента, найденная пара, выкинутые пары."""

    component: tuple[int, ...]
    removed: tuple[int, ...]
    pair: ReductionPair

    def __str__(self) -> str:
        return (
            f"компонента из {len(self.component)} пар: "
            f"строго убыли {len(self.removed)}"
        )


@dataclass
class DependencyProof:
    """Доказательство завершимости парами зависимостей.

    Хранит ровно то, что нужно, чтобы перепроверить вывод с нуля:
    сами пары и последовательность шагов. `check` ничего не берёт
    на веру — ни граф, ни компоненты, ни арифметику.
    """

    pairs: tuple[DependencyPair, ...]
    steps: tuple[Step, ...] = field(default_factory=tuple)

    def check(self, system) -> Verdict:
        """Арбитр: пересобрать всё заново и проверить каждый шаг.

        Возвращает «доказано» только если разобраны **все** циклические
        компоненты. От решателя не зависит: целочисленная арифметика.
        """
        rebuilt = dependency_pairs(system)
        if rebuilt != self.pairs:
            return unknown(
                "пары зависимостей в доказательстве не совпадают "
                "с парами самой системы"
            )
        defined = defined_symbols(system)
        graph = estimated_graph(self.pairs, defined)
        todo = [set(component) for component in cycles(graph)]
        spare = list(range(len(self.steps)))
        used = 0

        # Шаг ищется по компоненте, а не берётся по порядку: порядок обхода
        # компонент — деталь реализации поиска, и проверка от неё зависеть
        # не должна.
        while todo:
            component = todo.pop()
            found = [
                number
                for number in spare
                if set(self.steps[number].component) == component
            ]
            if not found:
                return unknown(
                    "в доказательстве нет шага для компоненты "
                    f"{sorted(component)}"
                )
            spare.remove(found[0])
            step = self.steps[found[0]]
            used += 1
            if not step.removed:
                return unknown(f"шаг {found[0] + 1} не выкидывает ни одной пары")
            for rule in system.rules:
                if not step.pair.weakly_decreases(rule.lhs, rule.rhs):
                    return unknown(
                        f"шаг {used}: правило «{rule}» не убывает нестрого"
                    )
            for index in sorted(component):
                strict = index in step.removed
                if not step.pair.pair_decreases(self.pairs[index], strict):
                    kind = "строго" if strict else "нестрого"
                    return unknown(
                        f"шаг {used}: пара «{self.pairs[index]}» "
                        f"не убывает {kind}"
                    )
            rest = component - set(step.removed)
            narrowed = {
                index: [target for target in graph[index] if target in rest]
                for index in sorted(rest)
            }
            todo.extend(set(part) for part in cycles(narrowed))

        if used != len(self.steps):
            return unknown(
                f"в доказательстве {len(self.steps)} шагов, "
                f"а понадобилось {used} — лишние шаги"
            )
        return proved(
            f"пары зависимостей: всего пар {len(self.pairs)}, "
            f"циклических компонент разобрано {len(self.steps)}, "
            "в каждой строго убывает хотя бы одна пара",
            self,
        )

    def markdown(self) -> str:
        lines = [
            f"Пар зависимостей: **{len(self.pairs)}**, "
            f"разобрано компонент: **{len(self.steps)}**.",
            "",
        ]
        for number, step in enumerate(self.steps, 1):
            lines.append(f"### Компонента {number}")
            lines.append("")
            lines.append("Пары компоненты:")
            lines.append("")
            for index in step.component:
                mark = " — **строго убывает**" if index in step.removed else ""
                lines.append(f"* `{self.pairs[index]}`{mark}")
            lines.append("")
            lines.append(step.pair.markdown())
            lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Поиск редукционной пары
# --------------------------------------------------------------------------


def have_solver() -> bool:
    """Есть ли Z3. Без него поиск честно отвечает «не выяснено»."""
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def find_reduction_pair(
    system,
    pairs: tuple[DependencyPair, ...],
    component: tuple[int, ...],
    dimension: int = 1,
    ceiling: int = 4,
    timeout_ms: int = 20_000,
) -> tuple[ReductionPair, tuple[int, ...]] | None:
    """Найти редукционную пару для одной компоненты.

    Требования к решателю: правила нестрого, пары компоненты нестрого,
    хотя бы одна из них строго. Возвращается сама интерпретация и номера
    строго убывших пар — либо `None`, если решателя нет или модели нет.

    Строгих пар берётся **сколько получится**: чем больше выкинуто
    за шаг, тем меньше остаётся работы. Решателю это ничего не стоит —
    условие «строго» на каждой паре стоит под своим флагом.
    """
    try:
        import z3
    except ImportError:
        return None

    letters = sorted(system.alphabet)
    if not letters:
        return None
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    def cells(name: str, rows: int, columns: int):
        table = [
            [z3.Int(f"{name}_{i}_{j}") for j in range(columns)] for i in range(rows)
        ]
        for row in table:
            for cell in row:
                solver.add(cell >= 0, cell <= ceiling)
        return table

    plain = {a: (cells(f"A{a}", dimension, dimension), cells(f"b{a}", dimension, 1))
             for a in letters}
    marked = {a: (cells(f"M{a}", dimension, dimension), cells(f"c{a}", dimension, 1))
              for a in letters}

    unit = [
        [z3.IntVal(1 if i == j else 0) for j in range(dimension)]
        for i in range(dimension)
    ]
    origin = [[z3.IntVal(0)] for _ in range(dimension)]

    def compose(outer, inner):
        table, vector = outer
        matrix, shift = inner
        product_matrix = [
            [
                z3.Sum([table[i][k] * matrix[k][j] for k in range(dimension)])
                for j in range(dimension)
            ]
            for i in range(dimension)
        ]
        product_shift = [
            [vector[i][0] + z3.Sum([table[i][k] * shift[k][0] for k in range(dimension)])]
            for i in range(dimension)
        ]
        return product_matrix, product_shift

    def value(word: str, head: str | None = None):
        current = (unit, origin)
        for letter in reversed(word):
            current = compose(plain[letter], current)
        if head is not None:
            current = compose(marked[head], current)
        return current

    def weak(left, right):
        left_matrix, left_shift = left
        right_matrix, right_shift = right
        return [
            left_matrix[i][j] >= right_matrix[i][j]
            for i in range(dimension)
            for j in range(dimension)
        ] + [left_shift[i][0] >= right_shift[i][0] for i in range(dimension)]

    for rule in system.rules:
        solver.add(*weak(value(rule.lhs), value(rule.rhs)))

    flags = []
    for index in component:
        pair = pairs[index]
        left = value(pair.lhs[1:], pair.lhs[0])
        right = value(pair.rhs[1:], pair.rhs[0])
        solver.add(*weak(left, right))
        flag = z3.Bool(f"strict_{index}")
        flags.append((index, flag))
        solver.add(z3.Implies(flag, left[1][0][0] > right[1][0][0]))
    solver.add(z3.Or([flag for _, flag in flags]))

    if solver.check() != z3.sat:
        return None
    model = solver.model()

    def read(table) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(model.eval(cell, model_completion=True).as_long() for cell in row)
            for row in table
        )

    def column(table) -> tuple[int, ...]:
        return tuple(row[0] for row in read(table))

    interpretation = ReductionPair(
        {a: Affine(read(plain[a][0]), column(plain[a][1])) for a in letters},
        {a: Affine(read(marked[a][0]), column(marked[a][1])) for a in letters},
        dimension,
    )
    strict = tuple(
        index for index, flag in flags
        if z3.is_true(model.eval(flag, model_completion=True))
    )
    return interpretation, strict


def prove_termination(
    system,
    dimension: int = 1,
    ceiling: int = 4,
    timeout_ms: int = 20_000,
) -> Verdict:
    """Доказать завершимость парами зависимостей.

    Исходов два: «доказано» (и тогда доказательство **уже проверено**
    арбитром `DependencyProof.check`) либо «не выяснено». Опровержения
    метод не даёт никогда — для этого есть поиск петли.
    """
    if not have_solver() and (dimension > 1 or len(system.alphabet) > 3):
        return unknown(
            "для поиска редукционной пары нужен Z3 (`pip install .[smt]`): "
            f"перебор годится лишь при d = 1 и алфавите до трёх букв, "
            f"а здесь d = {dimension} и букв {len(system.alphabet)}. "
            "Проверка готового доказательства от решателя не зависит"
        )
    pairs = dependency_pairs(system)
    if not pairs:
        return proved(
            "пар зависимостей нет вовсе: ни одна правая часть не начинает "
            "нового редекса, поэтому цепочек не бывает",
            DependencyProof((), ()),
        )
    defined = defined_symbols(system)
    graph = estimated_graph(pairs, defined)
    todo = [tuple(component) for component in cycles(graph)]
    if not todo:
        return _verified(system, DependencyProof(pairs, ()))

    steps: list[Step] = []
    while todo:
        component = todo.pop()
        found = (
            find_reduction_pair(
                system, pairs, component, dimension, ceiling, timeout_ms
            )
            if have_solver()
            else brute_force_pair(system, pairs, component, min(ceiling, 2))
        )
        if found is None:
            return unknown(
                f"редукционной пары размерности {dimension} с элементами "
                f"до {ceiling} для компоненты из {len(component)} пар нет; "
                "выше размерность, шире потолок либо другой аргумент",
                tuple(str(pairs[index]) for index in component),
            )
        interpretation, strict = found
        steps.append(Step(component, strict, interpretation))
        rest = set(component) - set(strict)
        narrowed = {
            index: [target for target in graph[index] if target in rest]
            for index in sorted(rest)
        }
        todo.extend(tuple(part) for part in cycles(narrowed))

    return _verified(system, DependencyProof(pairs, tuple(steps)))


def _verified(system, proof: DependencyProof) -> Verdict:
    """Прогнать доказательство через арбитра прежде, чем отдать наружу."""
    verdict = proof.check(system)
    if verdict.value is not True:
        return unknown(
            "поиск выдал доказательство, но арбитр его не принял: "
            f"{verdict.reason}"
        )
    return verdict


def brute_force_pair(
    system,
    pairs: tuple[DependencyPair, ...],
    component: tuple[int, ...],
    ceiling: int = 3,
) -> tuple[ReductionPair, tuple[int, ...]] | None:
    """Тот же поиск перебором — на случай, когда решателя нет.

    Годится только для `d = 1` и алфавита из двух-трёх букв: перебирается
    `(ceiling+1)⁴` наборов на букву, то есть `81³ ≈ 531 000` при трёх
    буквах и потолке 2. Нужен затем же, зачем в `tfl/matrix.py` отдельный
    арбитр: чтобы у метода была часть, ни от чего не зависящая.

    `prove_termination` вызывает его сам, когда Z3 не установлен.
    """
    letters = sorted(system.alphabet)
    values = range(ceiling + 1)
    space = list(product(values, repeat=4))
    for choice in product(space, repeat=len(letters)):
        plain = {
            letter: Affine(((slope,),), (free,))
            for letter, (slope, free, _, _) in zip(letters, choice)
        }
        marked = {
            letter: Affine(((slope,),), (free,))
            for letter, (_, _, slope, free) in zip(letters, choice)
        }
        candidate = ReductionPair(plain, marked, 1)
        if any(
            not candidate.weakly_decreases(rule.lhs, rule.rhs)
            for rule in system.rules
        ):
            continue
        if any(
            not candidate.pair_decreases(pairs[index], False) for index in component
        ):
            continue
        strict = tuple(
            index
            for index in component
            if candidate.pair_decreases(pairs[index], True)
        )
        if strict:
            return candidate, strict
    return None
