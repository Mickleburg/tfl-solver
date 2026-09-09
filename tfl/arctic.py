"""Арктические (max-plus) интерпретации для строковых систем переписывания.

Обычная матричная интерпретация (`tfl/matrix.py`) живёт над полукольцом
`(ℕ, +, ·)`. Арктическая — над `𝔸 = (ℕ ∪ {−∞}, max, +)`: «сложение» это
максимум, «умножение» это сложение. Поэтому произведение матриц

    (A ⊗ B)[i][j] = max_k (A[i][k] + B[k][j])

считает не сумму по всем путям, а **самый длинный путь**. Мера слова
перестаёт быть суммой вкладов и становится максимумом — и системы,
у которых убывает не сумма чего-нибудь, а наибольшее из нескольких
значений, берутся именно здесь.

Единица по `⊗` — матрица с `0` на диагонали и `−∞` вне её; ей отвечает
пустое слово, поэтому интерпретация согласована с конкатенацией
по построению, как и в обычном случае.

## Что именно проверяется

Два порядка на элементах `𝔸`:

* `a ⊒ b` — обычное `a ⩾ b`, где `−∞` меньше всего;
* `a ⊐ b` — **`a > b` либо `b = −∞`**.

Второй выглядит странно (он допускает `−∞ ⊐ −∞`), но именно он нужен:
за строгость отвечает не он сам, а конечность меры, см. доказательство
ниже. Мера слова — элемент `M(w)[0][d−1]`.

Правило `l → r` **убывает**, если

* `M(l) ⊐ M(r)` покоординатно и
* `M(l)[0][d−1] ≠ −∞`.

Каждая буквенная матрица обязана иметь конечные углы `[0][0]`
и `[d−1][d−1]`.

## Почему это доказательство

Пусть `w = u·l·v` переписывается в `w′ = u·r·v`, `U = M(u)`, `V = M(v)`,
`L = M(l)`, `R = M(r)`. Тогда

$$\\mu(w) = \\max_{i,j}\\bigl(U[0][i] + L[i][j] + V[j][d-1]\\bigr).$$

Пусть максимум для `w′` достигается на паре `(i₀, j₀)` и конечен. Тогда
`R[i₀][j₀] ≠ −∞`, значит по определению `⊐` имеем `L[i₀][j₀] > R[i₀][j₀]`,
и уже одно слагаемое с той же парой даёт `μ(w) > μ(w′)`.

Если же `μ(w′) = −∞`, достаточно показать, что `μ(w)` конечна. Возьмём
путь `i = 0`, `j = d−1`: `μ(w) ⩾ U[0][0] + L[0][d−1] + V[d−1][d−1]`,
а все три слагаемых конечны — углы у буквенных матриц конечны
по требованию, а произведение конечных углов конечно.

Итого мера лежит в `ℕ ∪ {−∞}` и на каждом шаге строго убывает, начиная
с конечного значения; как только она стала `−∞`, ни одно правило больше
неприменимо (применимость влечёт конечность). Значит вывод конечен.

## Граница метода

Арктические интерпретации **линейно** ограничивают длину вывода: мера
слова длины `n` не больше `n · max(элемент)`, а каждый шаг уменьшает её
хотя бы на единицу. Обычные матричные дают полиномиальную границу
степени `d`. Поэтому методы несравнимы: там, где вывод растёт квадратично,
арктика бессильна **в принципе**, а не «не нашлось».

**Поиск и проверка разделены**, как и в `tfl/matrix.py`:
`ArcticInterpretation.check` — арбитр, считает точно и ни от чего
не зависит; `find_arctic_interpretation` обращается к Z3, и без него
честно отвечает «не выяснено».
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl.verdict import Verdict, proved, unknown

__all__ = [
    "NEG",
    "ArcticMatrix",
    "ArcticInterpretation",
    "find_arctic_interpretation",
    "relative_step",
    "have_solver",
]

#: `−∞` — нейтральный элемент по `max` и поглощающий по `+`.
NEG = float("-inf")


def _show(value: float) -> str:
    return "−∞" if value == NEG else str(int(value))


@dataclass(frozen=True)
class ArcticMatrix:
    """Квадратная матрица над `𝔸 = (ℕ ∪ {−∞}, max, +)`."""

    rows: tuple[tuple[float, ...], ...]

    @property
    def dimension(self) -> int:
        return len(self.rows)

    @staticmethod
    def unit(dimension: int) -> ArcticMatrix:
        """Единица по `⊗`: нули на диагонали, `−∞` вне её."""
        return ArcticMatrix(
            tuple(
                tuple(0 if i == j else NEG for j in range(dimension))
                for i in range(dimension)
            )
        )

    def __mul__(self, other: ArcticMatrix) -> ArcticMatrix:
        size = self.dimension
        return ArcticMatrix(
            tuple(
                tuple(
                    max(self.rows[i][k] + other.rows[k][j] for k in range(size))
                    for j in range(size)
                )
                for i in range(size)
            )
        )

    def dominates(self, other: ArcticMatrix) -> bool:
        """`A ⊒ B` покоординатно: обычное `⩾`, где `−∞` меньше всего."""
        return all(
            mine >= yours
            for row, other_row in zip(self.rows, other.rows)
            for mine, yours in zip(row, other_row)
        )

    def strictly_dominates(self, other: ArcticMatrix) -> bool:
        """`A ⊐ B` покоординатно: `a > b` **либо** `b = −∞`.

        Допущение `−∞ ⊐ −∞` не ошибка: за фундированность отвечает
        конечность меры, а не иррефлексивность этого отношения.
        """
        return all(
            mine > yours or yours == NEG
            for row, other_row in zip(self.rows, other.rows)
            for mine, yours in zip(row, other_row)
        )

    @property
    def top_right(self) -> float:
        """Элемент `[0][d−1]` — мера слова."""
        return self.rows[0][self.dimension - 1]

    def __str__(self) -> str:
        width = max(len(_show(v)) for row in self.rows for v in row)
        body = "; ".join(
            " ".join(f"{_show(v):>{width}}" for v in row) for row in self.rows
        )
        return f"[{body}]"


@dataclass(frozen=True)
class ArcticInterpretation:
    """Буква → арктическая матрица. Слово → арктическое произведение."""

    matrices: dict[str, ArcticMatrix]

    @property
    def dimension(self) -> int:
        return next(iter(self.matrices.values())).dimension

    def __str__(self) -> str:
        return "; ".join(
            f"{letter} ↦ {matrix}" for letter, matrix in sorted(self.matrices.items())
        )

    def markdown(self) -> str:
        lines = []
        for letter, matrix in sorted(self.matrices.items()):
            rows = r" \\ ".join(
                " & ".join(
                    r"-\infty" if value == NEG else str(int(value)) for value in row
                )
                for row in matrix.rows
            )
            lines.append(rf"$[{letter}] = \begin{{pmatrix}} {rows} \end{{pmatrix}}$")
        return "\n\n".join(lines)

    def value(self, word: str) -> ArcticMatrix:
        result = ArcticMatrix.unit(self.dimension)
        for letter in word:
            result = result * self.matrices[letter]
        return result

    def is_monotone(self) -> bool:
        """Конечны ли углы `[0][0]` и `[d−1][d−1]` у каждой буквы.

        Без этого мера контекста может оказаться `−∞`, и строгое убывание
        в контекст не переносится.
        """
        last = self.dimension - 1
        return all(
            matrix.rows[0][0] != NEG and matrix.rows[last][last] != NEG
            for matrix in self.matrices.values()
        )

    def weakly_decreases(self, left: str, right: str) -> bool:
        """Не возрастает ли правило: `M(l) ⊒ M(r)` покоординатно."""
        return self.value(left).dominates(self.value(right))

    def decreases(self, left: str, right: str) -> bool:
        """Строго ли убывает правило `left → right`."""
        first, second = self.value(left), self.value(right)
        return first.top_right != NEG and first.strictly_dominates(second)

    def check(self, system) -> Verdict:
        """Доказывает ли интерпретация завершимость системы.

        Арбитр: считает точно, ни от какого решателя не зависит.
        """
        if not self.matrices:
            return unknown("интерпретация пуста")
        if not self.is_monotone():
            return unknown(
                "интерпретация не монотонна: у какой-то буквы угол [0][0] "
                "или [d−1][d−1] равен −∞, и мера контекста может обратиться "
                "в −∞ — вывод неправомерен"
            )
        missing = {
            letter
            for rule in system.rules
            for letter in rule.lhs + rule.rhs
            if letter not in self.matrices
        }
        if missing:
            return unknown(f"нет матриц для букв: {', '.join(sorted(missing))}")
        for rule in system.rules:
            left, right = self.value(rule.lhs), self.value(rule.rhs)
            if left.top_right == NEG:
                return unknown(
                    f"мера левой части правила «{rule}» равна −∞: {left}. "
                    f"Убывать нечему",
                    rule,
                )
            if not left.strictly_dominates(right):
                return unknown(
                    f"правило «{rule}» не убывает: {left} против {right} "
                    f"(нужно покоординатно «больше либо справа −∞»)",
                    rule,
                )
        return proved(
            f"все {len(system.rules)} правил убывают в арктической интерпретации "
            f"размерности {self.dimension}: {self}",
            self,
        )


def have_solver() -> bool:
    """Установлен ли Z3. Обязательной зависимостью проекта он не является."""
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def _search(rules, dimension: int, max_entry: int, timeout_ms: int, relative: bool):
    """Общая часть поиска: одна кодировка на два режима.

    `−∞` кодируется числом `−1`: конечные элементы лежат в `[0, max_entry]`,
    и арктическое умножение насыщается, `x ⊗ y = −1`, как только один
    из сомножителей отрицателен. Кодировка точная, а не приближённая:
    все `−∞` склеены в одно значение нарочно, иначе решатель начинает
    их различать и `⩾` перестаёт быть арктическим.

    `relative=True` требует строгого убывания не от всех правил,
    а хотя бы от одного — для удаления правил.
    """
    import z3

    letters = sorted({letter for rule in rules for letter in rule.lhs + rule.rhs})
    if not letters:
        return "в системе нет букв"
    last = dimension - 1
    cells = {
        letter: [
            [z3.Int(f"{letter}_{i}_{j}") for j in range(dimension)]
            for i in range(dimension)
        ]
        for letter in letters
    }

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for letter in letters:
        for i in range(dimension):
            for j in range(dimension):
                cell = cells[letter][i][j]
                solver.add(cell >= -1, cell <= max_entry)
        solver.add(cells[letter][0][0] >= 0)
        solver.add(cells[letter][last][last] >= 0)

    def times(x, y):
        return z3.If(z3.Or(x < 0, y < 0), z3.IntVal(-1), x + y)

    def plus(x, y):
        return z3.If(x >= y, x, y)

    def product(word: str):
        result = [
            [z3.IntVal(0 if i == j else -1) for j in range(dimension)]
            for i in range(dimension)
        ]
        for letter in word:
            fresh = []
            for i in range(dimension):
                row = []
                for j in range(dimension):
                    term = times(result[i][0], cells[letter][0][j])
                    for k in range(1, dimension):
                        term = plus(term, times(result[i][k], cells[letter][k][j]))
                    row.append(term)
                fresh.append(row)
            result = fresh
        return result

    strict = []
    for rule in rules:
        left, right = product(rule.lhs), product(rule.rhs)
        for i in range(dimension):
            for j in range(dimension):
                solver.add(left[i][j] >= right[i][j])
        falls = z3.And(
            [left[0][last] >= 0]
            + [
                z3.Or(left[i][j] > right[i][j], right[i][j] < 0)
                for i in range(dimension)
                for j in range(dimension)
            ]
        )
        if relative:
            strict.append(falls)
        else:
            solver.add(falls)
    if relative:
        solver.add(z3.Or(strict))

    outcome = solver.check()
    if outcome == z3.unsat:
        return (
            f"арктической интерпретации размерности {dimension} с элементами "
            f"до {max_entry} не существует — это доказано решателем. "
            f"Про большие размерности и элементы отсюда не следует ничего"
        )
    if outcome != z3.sat:
        return f"решатель не уложился в {timeout_ms} мс на размерности {dimension}"
    model = solver.model()

    def read(letter: str, i: int, j: int) -> float:
        raw = model.eval(cells[letter][i][j], model_completion=True).as_long()
        return NEG if raw < 0 else float(raw)

    return ArcticInterpretation(
        {
            letter: ArcticMatrix(
                tuple(
                    tuple(read(letter, i, j) for j in range(dimension))
                    for i in range(dimension)
                )
            )
            for letter in letters
        }
    )


def relative_step(
    rules, dimension: int = 2, max_entry: int = 3, timeout_ms: int = 20_000
):
    """Интерпретация, роняющая все правила нестрого и хотя бы одно строго.

    Возвращает интерпретацию либо `None`: проверяет её вызывающий.
    """
    if not have_solver():
        return None
    found = _search(tuple(rules), dimension, max_entry, timeout_ms, relative=True)
    return found if isinstance(found, ArcticInterpretation) else None


def find_arctic_interpretation(
    system,
    dimension: int = 2,
    max_entry: int = 3,
    timeout_ms: int = 20_000,
) -> Verdict:
    """Искать арктическую интерпретацию SMT-решателем.

    Отрицательный ответ решателя означает «нет интерпретации **с такой**
    размерностью и такими элементами», а не «нет вовсе».
    """
    if not have_solver():
        return unknown(
            "SMT-решателя Z3 в окружении нет, поиск арктической интерпретации "
            "не выполнялся"
        )
    found = _search(system.rules, dimension, max_entry, timeout_ms, relative=False)
    if isinstance(found, str):
        return unknown(found)
    # Ответ решателя перепроверяется своим арбитром: кодировка «−∞ = −1»
    # верна, но верить в это на слово в этом проекте не принято.
    verified = found.check(system)
    if verified.value is not True:
        return unknown(
            f"решатель выдал интерпретацию, не прошедшую проверку: {verified.reason}"
        )
    return proved(
        f"арктическая интерпретация размерности {dimension}: {found}", found
    )
