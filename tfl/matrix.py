"""Матричные интерпретации для строковых систем переписывания.

Каждой букве сопоставляется матрица над ℕ, слову — произведение матриц
в порядке букв. Пустому слову отвечает единичная матрица, так что
интерпретация согласована с конкатенацией по построению.

Правило `l → r` **убывает**, если

* `M(l) ⩾ M(r)` покоординатно и
* `M(l)[0][d−1] > M(r)[0][d−1]`.

Чтобы убывание сохранялось в контексте, каждая буквенная матрица обязана
иметь `A[0][0] ⩾ 1` и `A[d−1][d−1] ⩾ 1`. Тогда в разложении

    (U·L·V)[0][d−1] = Σᵢⱼ U[0][i]·L[i][j]·V[j][d−1]

слагаемое с `i = 0`, `j = d−1` строго больше соответствующего для `R`,
а остальные не меньше. Значение `M(w)[0][d−1]` лежит в ℕ и строго убывает
на каждом шаге — отсюда завершимость.

Зачем это здесь. Линейные интерпретации `x ↦ m·x + c` (`SRS.find_interpretation`)
закрывают на вариантах ЛР1 2025 ровно один случай из двадцати «не выяснено»,
армейский порядок бессилен против удлиняющих правил, а перебор матриц 2×2
с элементами `{0,1}` не добавил вообще ничего. Матрица — это та же линейная
интерпретация, но многомерная: она умеет считать не одну величину, а несколько
сразу, и потому берёт системы, где убывает не длина и не взвешенная сумма,
а более хитрая мера.

**Поиск и проверка разделены нарочно.** `MatrixInterpretation.check`
работает всегда и ни от чего не зависит: это арбитр. Поиск
(`find_matrix_interpretation`) обращается к SMT-решателю Z3, и если его
в окружении нет, честно возвращает «не выяснено» вместо ошибки. Модель
можно получить текстом (`smtlib_model`) и скормить любому решателю.
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl.verdict import Verdict, proved, unknown

__all__ = [
    "Matrix",
    "MatrixInterpretation",
    "find_matrix_interpretation",
    "smtlib_model",
    "have_solver",
]


@dataclass(frozen=True)
class Matrix:
    """Квадратная матрица над ℕ."""

    rows: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        return len(self.rows)

    @staticmethod
    def identity(dimension: int) -> Matrix:
        return Matrix(
            tuple(
                tuple(1 if i == j else 0 for j in range(dimension))
                for i in range(dimension)
            )
        )

    def __mul__(self, other: Matrix) -> Matrix:
        size = self.dimension
        return Matrix(
            tuple(
                tuple(
                    sum(self.rows[i][k] * other.rows[k][j] for k in range(size))
                    for j in range(size)
                )
                for i in range(size)
            )
        )

    def __ge__(self, other: Matrix) -> bool:
        return all(
            mine >= yours
            for row, other_row in zip(self.rows, other.rows)
            for mine, yours in zip(row, other_row)
        )

    @property
    def top_right(self) -> int:
        """Элемент `[0][d−1]` — по нему и сравниваются слова."""
        return self.rows[0][self.dimension - 1]

    def __str__(self) -> str:
        width = max((len(str(v)) for row in self.rows for v in row), default=1)
        body = "; ".join(" ".join(f"{v:>{width}}" for v in row) for row in self.rows)
        return f"[{body}]"


@dataclass(frozen=True)
class MatrixInterpretation:
    """Буква → матрица. Слово → произведение."""

    matrices: dict[str, Matrix]

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
                " & ".join(str(value) for value in row) for row in matrix.rows
            )
            lines.append(rf"$[{letter}] = \begin{{pmatrix}} {rows} \end{{pmatrix}}$")
        return "\n\n".join(lines)

    def value(self, word: str) -> Matrix:
        result = Matrix.identity(self.dimension)
        for letter in word:
            result = result * self.matrices[letter]
        return result

    def is_monotone(self) -> bool:
        """Углы `[0][0]` и `[d−1][d−1]` не меньше единицы у каждой буквы.

        Без этого убывание не переносится в контекст, и «интерпретация»
        ничего не доказывает.
        """
        last = self.dimension - 1
        return all(
            matrix.rows[0][0] >= 1 and matrix.rows[last][last] >= 1
            for matrix in self.matrices.values()
        )

    def decreases(self, left: str, right: str) -> bool:
        """Строго ли убывает правило `left → right`."""
        first, second = self.value(left), self.value(right)
        return first >= second and first.top_right > second.top_right

    def check(self, system) -> Verdict:
        """Доказывает ли интерпретация завершимость системы.

        Арбитр: ни от какого решателя не зависит, считает точно.
        """
        if not self.is_monotone():
            return unknown(
                "интерпретация не монотонна: у какой-то буквы угол [0][0] "
                "или [d−1][d−1] равен нулю, и убывание не переносится "
                "в контекст — вывод неправомерен"
            )
        missing = {
            letter
            for rule in system.rules
            for letter in rule.lhs + rule.rhs
            if letter not in self.matrices
        }
        if missing:
            return unknown(f"нет матриц для букв: {', '.join(sorted(missing))}")
        bad = [rule for rule in system.rules if not self.decreases(rule.lhs, rule.rhs)]
        if bad:
            rule = bad[0]
            left, right = self.value(rule.lhs), self.value(rule.rhs)
            return unknown(
                f"правило «{rule}» не убывает: {left} против {right} "
                f"(сравниваем углы {left.top_right} и {right.top_right})",
                rule,
            )
        return proved(
            f"все {len(system.rules)} правил убывают в матричной интерпретации "
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


def find_matrix_interpretation(
    system,
    dimension: int = 2,
    max_entry: int = 3,
    timeout_ms: int = 20_000,
) -> Verdict:
    """Искать матричную интерпретацию SMT-решателем.

    Элементы ограничены сверху нарочно: без границы задача уходит
    в нелинейную целочисленную арифметику без конца, а с границей
    становится конечной и решается быстро. Отрицательный ответ решателя
    поэтому означает «нет интерпретации **с такими** элементами
    и такой размерности», а не «нет вовсе».
    """
    if not have_solver():
        return unknown(
            "SMT-решателя Z3 в окружении нет, поиск матричной интерпретации "
            "не выполнялся. Обязательной зависимостью он не сделан; модель "
            "можно получить через smtlib_model и решить снаружи"
        )
    import z3

    letters = sorted({letter for rule in system.rules for letter in rule.lhs + rule.rhs})
    if not letters:
        return unknown("в системе нет букв")
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
                solver.add(cells[letter][i][j] >= 0, cells[letter][i][j] <= max_entry)
        solver.add(cells[letter][0][0] >= 1)
        solver.add(cells[letter][last][last] >= 1)

    def product(word: str):
        result = [
            [z3.IntVal(1 if i == j else 0) for j in range(dimension)]
            for i in range(dimension)
        ]
        for letter in word:
            result = [
                [
                    z3.Sum(
                        [result[i][k] * cells[letter][k][j] for k in range(dimension)]
                    )
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ]
        return result

    for rule in system.rules:
        left, right = product(rule.lhs), product(rule.rhs)
        for i in range(dimension):
            for j in range(dimension):
                solver.add(left[i][j] >= right[i][j])
        solver.add(left[0][last] > right[0][last])

    outcome = solver.check()
    if outcome == z3.unsat:
        return unknown(
            f"матричной интерпретации размерности {dimension} с элементами "
            f"до {max_entry} не существует — это доказано решателем. "
            f"Про большие размерности и элементы отсюда не следует ничего"
        )
    if outcome != z3.sat:
        return unknown(
            f"решатель не уложился в {timeout_ms} мс на размерности {dimension}"
        )
    model = solver.model()
    found = MatrixInterpretation(
        {
            letter: Matrix(
                tuple(
                    tuple(model.eval(cells[letter][i][j]).as_long() for j in range(dimension))
                    for i in range(dimension)
                )
            )
            for letter in letters
        }
    )
    # Ответ решателя перепроверяется своим арбитром: доверять чужому
    # «sat» на слово в этом проекте не принято.
    verified = found.check(system)
    if verified.value is not True:
        return unknown(
            f"решатель выдал интерпретацию, не прошедшую проверку: "
            f"{verified.reason}"
        )
    return proved(
        f"матричная интерпретация размерности {dimension}: {found}", found
    )


def smtlib_model(system, dimension: int = 2, max_entry: int = 3) -> str:
    """Та же модель текстом на smtlib — для решателя вне проекта.

    Нужна ровно затем же, зачем `tfl.pcp.PCP.smtlib_model`: чтобы работа
    не упиралась в наличие питоновской обвязки решателя.
    """
    letters = sorted({letter for rule in system.rules for letter in rule.lhs + rule.rhs})
    last = dimension - 1
    lines = ["(set-logic QF_NIA)"]

    def name(letter: str, i: int, j: int) -> str:
        return f"{letter}_{i}_{j}"

    for letter in letters:
        for i in range(dimension):
            for j in range(dimension):
                lines.append(f"(declare-const {name(letter, i, j)} Int)")
                lines.append(f"(assert (>= {name(letter, i, j)} 0))")
                lines.append(f"(assert (<= {name(letter, i, j)} {max_entry}))")
        lines.append(f"(assert (>= {name(letter, 0, 0)} 1))")
        lines.append(f"(assert (>= {name(letter, last, last)} 1))")

    def product(word: str) -> list[list[str]]:
        result = [
            [("1" if i == j else "0") for j in range(dimension)] for i in range(dimension)
        ]
        for letter in word:
            result = [
                [
                    "(+ "
                    + " ".join(
                        f"(* {result[i][k]} {name(letter, k, j)})"
                        for k in range(dimension)
                    )
                    + ")"
                    for j in range(dimension)
                ]
                for i in range(dimension)
            ]
        return result

    for number, rule in enumerate(system.rules, start=1):
        left, right = product(rule.lhs), product(rule.rhs)
        lines.append(f"; правило {number}: {rule}")
        for i in range(dimension):
            for j in range(dimension):
                lines.append(f"(assert (>= {left[i][j]} {right[i][j]}))")
        lines.append(f"(assert (> {left[0][last]} {right[0][last]}))")

    lines += ["(check-sat)", "(get-model)"]
    return "\n".join(lines)
