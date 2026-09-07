"""Проблема соответствия Поста: домино, поиск решения и счётные опровержения.

Экземпляр — список домино `⟨верх, низ⟩`. Решение — непустая
последовательность номеров, при которой склейка верхов совпадает
со склейкой низов. Задача неразрешима в общем виде, поэтому оракул
устроен так же, как везде в проекте: поиск даёт `True` со свидетелем,
счётные условия дают `False` с доказательством, а «не нашли» остаётся
«не выяснено».

**Метод преподавателя** (ЛР0 2023, лист «ЛР 1-2»): если решение `W`
существует, в нём определены числа `Mᵢ` — сколько раз использовано
домино `i`. Равенство числа вхождений каждой буквы сверху и снизу
даёт однородную линейную систему на `Mᵢ`. Её неразрешимость в целых
неотрицательных числах доказывает, что решения нет. Разобранный
в условии пример — `⟨ab,a⟩, ⟨aa,b⟩, ⟨bb,a⟩` с уравнениями
`2·M₂ = M₃` и `M₁ + 2·M₃ = M₂`, у которых решений нет.

Тонкость, которой в условии нет. Преподаватель пишет «в целых
**положительных** значениях `Mᵢ` решений не существует». Читать это
буквально нельзя: решение ПСП не обязано использовать все домино,
и система с `Mᵢ > 0` может не иметь решений, тогда как `Mᵢ ⩾ 0`
с ненулевой суммой — иметь. Опровергать надо по слабому условию,
иначе опровержение неверно. Оракул считает оба и различает их
в вердикте: «нет даже неотрицательного» — доказательство,
«положительного нет, а неотрицательное есть» — лишь сведение к тому,
что какие-то домино в решении не участвуют.

Усиление уравнениями на **пары** букв (переменные `Nᵢⱼ` — сколько раз
домино `j` стоит сразу за `i`) в лоб не считается: переменных
становится `n + n²`, и точное исключение Фурье–Моцкина взрывается.
Для него `smtlib_model` печатает SMT-модель, как и просит условие ЛР0.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = ["Domino", "PCP", "parse_pcp"]


@dataclass(frozen=True)
class Domino:
    """Одно домино: слово сверху и слово снизу."""

    top: str
    bottom: str

    def __str__(self) -> str:
        return f"⟨{self.top or 'ε'}, {self.bottom or 'ε'}⟩"


def parse_pcp(text: str) -> PCP:
    """Разобрать список домино. Пары пишутся как `(ab,a)` либо `⟨ab, a⟩`.

    >>> str(parse_pcp("(ab,a)\\n(aa,b)\\n(bb,a)"))
    '⟨ab, a⟩ ⟨aa, b⟩ ⟨bb, a⟩'
    """
    dominoes: list[Domino] = []
    for line in text.splitlines():
        line = line.strip().strip("(){}⟨⟩[]").strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            raise ValueError(f"домино пишется парой через запятую: {line!r}")
        top, bottom = line.split(",", 1)
        dominoes.append(Domino(top.strip().replace("ε", ""), bottom.strip().replace("ε", "")))
    return PCP(tuple(dominoes))


@dataclass(frozen=True)
class PCP:
    """Экземпляр проблемы соответствия Поста."""

    dominoes: tuple[Domino, ...]

    def __len__(self) -> int:
        return len(self.dominoes)

    def __str__(self) -> str:
        return " ".join(str(d) for d in self.dominoes)

    @property
    def alphabet(self) -> str:
        letters = set()
        for domino in self.dominoes:
            letters |= set(domino.top) | set(domino.bottom)
        return "".join(sorted(letters))

    @staticmethod
    def all_pairs(words) -> PCP:
        """Экземпляр из **всех** пар слов данного множества (вопрос 11 «Аптеки» 2022).

        Ответ на вопрос про общий метод виден сразу, как только экземпляр
        построен: среди всех пар есть диагональные `⟨wᵢ, wᵢ⟩`, а такое
        домино решает задачу в одиночку. Значит, для множеств такого вида
        ПСП разрешима всегда, и решение — одно домино.
        """
        words = list(words)
        return PCP(tuple(Domino(top, bottom) for top in words for bottom in words))

    # ------------------------------------------------------------- решение

    def word(self, indices) -> tuple[str, str]:
        """Склейка верхов и низов по последовательности номеров."""
        top = "".join(self.dominoes[i].top for i in indices)
        bottom = "".join(self.dominoes[i].bottom for i in indices)
        return top, bottom

    def is_solution(self, indices) -> bool:
        top, bottom = self.word(indices)
        return bool(indices) and top == bottom

    def solve(self, max_dominoes: int = 40, max_nodes: int = 500_000) -> Verdict:
        """Искать решение обходом в ширину по «хвосту» — непокрытому остатку.

        Состояние — пара «хвост, чья сторона забежала вперёд». Больше помнить
        нечего: продолжение зависит только от хвоста, поэтому повторные
        состояния отсекаются. Обход идёт по уровням, уровень — число
        использованных домино, так что первое встреченное решение кратчайшее,
        а отсев повторов запоминает наименьший уровень и границу не ломает.

        Исходов два. `True` со свидетелем — решение найдено. `None` —
        не найдено, и это **не** значит, что его нет: задача неразрешима.
        Но в вердикте остаётся точное утверждение, годное в отчёт: решения
        не более чем из стольких-то домино не существует.
        """
        if any(d.top == d.bottom for d in self.dominoes):
            index = next(i for i, d in enumerate(self.dominoes) if d.top == d.bottom)
            return proved(
                f"домино {index + 1} = {self.dominoes[index]} решает задачу в одиночку",
                (index,),
            )

        level: list[tuple[str, int, tuple[int, ...]]] = [("", 0, ())]
        seen: set[tuple[str, int]] = {("", 0)}
        nodes = 0
        for depth in range(max_dominoes):
            following: list[tuple[str, int, tuple[int, ...]]] = []
            for surplus, side, path in level:
                for index, domino in enumerate(self.dominoes):
                    top = (surplus if side > 0 else "") + domino.top
                    bottom = (surplus if side < 0 else "") + domino.bottom
                    if top.startswith(bottom):
                        rest, ahead = top[len(bottom) :], 1
                    elif bottom.startswith(top):
                        rest, ahead = bottom[len(top) :], -1
                    else:
                        continue
                    found = (*path, index)
                    if not rest:
                        return proved(
                            "решение: домино "
                            + " ".join(str(i + 1) for i in found)
                            + f"; слово «{self.word(found)[0]}»",
                            found,
                        )
                    state = (rest, ahead)
                    if state in seen:
                        continue
                    nodes += 1
                    if nodes >= max_nodes:
                        return unknown(
                            f"поиск прерван на {max_nodes} состояниях, дойдя "
                            f"до последовательностей из {depth + 1} домино"
                        )
                    seen.add(state)
                    following.append((rest, ahead, found))
            level = following
            if not level:
                return unknown(
                    f"продолжений не осталось: ни одна незавершённая цепочка "
                    f"не продолжается дальше {depth} домино, решения нет. "
                    f"Это доказательство для данного экземпляра"
                )
        return unknown(
            f"решения из не более чем {max_dominoes} домино не существует "
            f"(обход по уровням пройден целиком, {len(seen)} состояний). "
            f"Про более длинные решения отсюда не следует ничего"
        )

    # --------------------------------------------- счётные опровержения

    def letter_equations(self) -> list[tuple[str, list[int]]]:
        """Уравнения на числа `Mᵢ`: вхождений каждой буквы сверху и снизу поровну.

        Возвращает пары «буква, коэффициенты», где коэффициент при `Mᵢ` —
        это `|верхᵢ|_c − |низᵢ|_c`.
        """
        rows = []
        for letter in self.alphabet:
            row = [d.top.count(letter) - d.bottom.count(letter) for d in self.dominoes]
            rows.append((letter, row))
        return rows

    def equations_markdown(self) -> str:
        """Уравнения в том виде, в каком их пишет преподаватель."""
        lines = []
        for letter, row in self.letter_equations():
            left = _side(row, +1)
            right = _side(row, -1)
            lines.append(f"- буква `{letter}`: ${left or '0'} = {right or '0'}$")
        return "\n".join(lines)

    def refute_by_counting(self) -> Verdict:
        """Опровергнуть существование решения счётными уравнениями.

        Необходимое условие: у системы «вхождений каждой буквы сверху
        и снизу поровну» есть **неотрицательное ненулевое** решение.
        Если нет — решения ПСП нет, и это доказательство.
        """
        rows = [row for _, row in self.letter_equations()]
        weak = _nonnegative_nonzero(rows, len(self))
        if weak is None:
            return refuted(
                "система уравнений на числа домино не имеет неотрицательных "
                "ненулевых решений, значит решения ПСП не существует:\n"
                + self.equations_markdown(),
                self.letter_equations(),
            )
        strong = _positive(rows, len(self))
        shown = ", ".join(f"M{i + 1}={v}" for i, v in enumerate(weak))
        if strong is None:
            return unknown(
                f"счётные уравнения решение не запрещают, но **положительных** "
                f"решений у них нет: любое решение ПСП обязано обойтись без "
                f"каких-то домино. Неотрицательное решение: {shown}",
                weak,
            )
        return unknown(
            "счётные уравнения решения не запрещают: "
            + ", ".join(f"M{i + 1}={v}" for i, v in enumerate(strong)),
            strong,
        )

    def adjacency_system(self, first: int, last: int):
        """Уравнения метода преподавателя целиком, при заданных крайних домино.

        Переменные: `M₁…Mₙ` — сколько раз использовано домино, и `Nᵢⱼ` —
        сколько раз домино `j` стоит непосредственно справа от `i`.
        Уравнений три вида:

        * буквы — вхождений каждой буквы сверху и снизу поровну;
        * баланс соседств — правый сосед есть у каждого вхождения, кроме
          последнего, левый — у каждого, кроме первого; отсюда и берётся
          оговорка условия «крайние домино соседствуют только с одним»;
        * пары букв — биграмма склейки либо лежит внутри слова одного
          домино, либо приходится на стык, а стыков ровно `Nᵢⱼ`.

        Номер переменной: `Mᵢ` — это `i`, `Nᵢⱼ` — это `n + i·n + j`.
        """
        n = len(self)
        size = n + n * n

        def index_n(i: int, j: int) -> int:
            return n + i * n + j

        system: list[tuple[list[int], int]] = []
        for _, row in self.letter_equations():
            system.append((row + [0] * (n * n), 0))

        for i in range(n):
            right = [0] * size
            right[i] = -1
            for j in range(n):
                right[index_n(i, j)] = 1
            system.append((right, -1 if i == last else 0))

            left = [0] * size
            left[i] = -1
            for j in range(n):
                left[index_n(j, i)] = 1
            system.append((left, -1 if i == first else 0))

        letters = self.alphabet
        for head in letters:
            for tail in letters:
                pair = head + tail
                row = [0] * size
                for i, domino in enumerate(self.dominoes):
                    row[i] += _bigrams(domino.top, pair) - _bigrams(domino.bottom, pair)
                for i, left_domino in enumerate(self.dominoes):
                    for j, right_domino in enumerate(self.dominoes):
                        row[index_n(i, j)] += _joint(
                            left_domino.top, right_domino.top, pair
                        ) - _joint(left_domino.bottom, right_domino.bottom, pair)
                if any(row):
                    system.append((row, 0))
        return system, size

    def refute_by_adjacency(self) -> Verdict:
        """Опровергнуть решение усиленной системой — с уравнениями на пары букв.

        Крайние домино перебираются: каждая пара «первое, последнее» даёт
        свою систему, и если ни одна не разрешима в неотрицательных числах,
        решения нет. Домино с пустым словом ломают счёт стыков, поэтому
        на них метод не применяется — условие ЛР0 их и запрещает.

        Побочный, но полезный результат: список **выживших** пар. Он сам
        по себе необходимое условие и годится в отчёт.
        """
        if any(not d.top or not d.bottom for d in self.dominoes):
            return unknown(
                "среди домино есть пустое слово: стыковые биграммы тогда "
                "не определены, и метод неприменим"
            )
        survivors = []
        for first in range(len(self)):
            for last in range(len(self)):
                system, size = self.adjacency_system(first, last)
                lower = [Fraction(0)] * size
                lower[first] = Fraction(1)
                lower[last] = Fraction(1)
                if _feasible(system, size, lower) is not None:
                    survivors.append((first + 1, last + 1))
        if not survivors:
            return refuted(
                "усиленная система (буквы, соседства и пары букв) неразрешима "
                "ни при каком выборе первого и последнего домино, "
                "значит решения ПСП не существует",
                (),
            )
        shown = ", ".join(f"{a}…{b}" for a, b in survivors)
        return unknown(
            f"усиленная система решение не запрещает. Отсеяны все пары "
            f"«первое…последнее домино», кроме {len(survivors)} из "
            f"{len(self) ** 2}: {shown}",
            survivors,
        )

    def smtlib_model(self) -> str:
        """SMT-модель по условию ЛР0 2023: уравнения на `Mᵢ` **и** на `Nᵢⱼ`.

        `Nᵢⱼ` — сколько раз домино `j` стоит непосредственно справа от `i`.
        Связь с `Mᵢ` — обычный баланс: у каждого вхождения есть правый сосед,
        кроме последнего, и левый, кроме первого. Отсюда «крайние домино
        соседствуют только с одним», о чём и говорит условие: номера первого
        и последнего домино вводятся булевыми флагами с ровно одной единицей.

        Уравнения на пары букв: биграммы склейки складываются из биграмм
        внутри слов и из стыков, а стыков ровно `Nᵢⱼ`.

        Печатается текст, а не вердикт: Z3 в окружении нет, и делать его
        обязательной зависимостью проекта я не стал.
        """
        n = len(self)
        lines = ["(set-logic QF_LIA)"]
        for i in range(n):
            lines.append(f"(declare-const M{i + 1} Int)")
            lines.append(f"(assert (>= M{i + 1} 0))")
        for i in range(n):
            for j in range(n):
                lines.append(f"(declare-const N{i + 1}_{j + 1} Int)")
                lines.append(f"(assert (>= N{i + 1}_{j + 1} 0))")
        for i in range(n):
            lines.append(f"(declare-const first{i + 1} Int)")
            lines.append(f"(declare-const last{i + 1} Int)")
            lines.append(f"(assert (and (>= first{i + 1} 0) (<= first{i + 1} 1)))")
            lines.append(f"(assert (and (>= last{i + 1} 0) (<= last{i + 1} 1)))")
        lines.append(f"(assert (= 1 {_sum(f'first{i + 1}' for i in range(n))}))")
        lines.append(f"(assert (= 1 {_sum(f'last{i + 1}' for i in range(n))}))")
        lines.append(f"(assert (>= {_sum(f'M{i + 1}' for i in range(n))} 1))")

        lines.append("; баланс соседств: правый сосед есть у всех, кроме последнего")
        for i in range(n):
            right = _sum(f"N{i + 1}_{j + 1}" for j in range(n))
            lines.append(f"(assert (= {right} (- M{i + 1} last{i + 1})))")
            left = _sum(f"N{j + 1}_{i + 1}" for j in range(n))
            lines.append(f"(assert (= {left} (- M{i + 1} first{i + 1})))")
            lines.append(f"(assert (<= first{i + 1} M{i + 1}))")
            lines.append(f"(assert (<= last{i + 1} M{i + 1}))")

        lines.append("; буквы: вхождений сверху и снизу поровну")
        for letter, row in self.letter_equations():
            terms = _sum(f"(* {c} M{i + 1})" for i, c in enumerate(row) if c)
            lines.append(f"(assert (= 0 {terms or '0'})) ; буква {letter}")

        lines.append("; пары букв: биграммы внутри слов плюс стыки")
        letters = self.alphabet
        for first in letters:
            for second in letters:
                pair = first + second
                terms = []
                for i, domino in enumerate(self.dominoes):
                    inside = _bigrams(domino.top, pair) - _bigrams(domino.bottom, pair)
                    if inside:
                        terms.append(f"(* {inside} M{i + 1})")
                for i, left in enumerate(self.dominoes):
                    for j, right in enumerate(self.dominoes):
                        joint = _joint(left.top, right.top, pair) - _joint(
                            left.bottom, right.bottom, pair
                        )
                        if joint:
                            terms.append(f"(* {joint} N{i + 1}_{j + 1})")
                if terms:
                    lines.append(f"(assert (= 0 {_sum(terms)})) ; пара {pair}")

        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------


def _sum(terms) -> str:
    terms = list(terms)
    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "(+ " + " ".join(terms) + ")"


def _bigrams(word: str, pair: str) -> int:
    return sum(1 for k in range(len(word) - 1) if word[k : k + 2] == pair)


def _joint(left: str, right: str, pair: str) -> int:
    """Стык двух слов даёт биграмму, если оба непусты."""
    if not left or not right:
        return 0
    return 1 if left[-1] + right[0] == pair else 0


def _side(row: list[int], sign: int) -> str:
    parts = []
    for index, value in enumerate(row):
        if value * sign <= 0:
            continue
        weight = abs(value)
        parts.append(f"M_{index + 1}" if weight == 1 else f"{weight} M_{index + 1}")
    return " + ".join(parts)


# --------------------------------------------------------------------------
# Точное решение однородной системы в неотрицательных числах
# --------------------------------------------------------------------------


def _positive(rows: list[list[int]], size: int) -> list[Fraction] | None:
    """Решение `A·M = 0` со **всеми** `Mᵢ ⩾ 1`, если оно есть."""
    return _feasible([(row, 0) for row in rows], size, [Fraction(1)] * size)


def _nonnegative_nonzero(rows: list[list[int]], size: int) -> list[Fraction] | None:
    """Решение `A·M = 0` с `M ⩾ 0` и хотя бы одной положительной координатой.

    Проверяется по одной координате: если у какого-то `i` есть решение
    с `Mᵢ = 1`, оно и годится. Обратно тоже верно, поэтому проверка точна.
    """
    for index in range(size):
        bounds = [Fraction(0)] * size
        bounds[index] = Fraction(1)
        unit = [0] * size
        unit[index] = 1
        system = [(row, 0) for row in rows] + [(unit, 1)]
        found = _feasible(system, size, bounds)
        if found is not None:
            return found
    return None


def _feasible(
    system: list[tuple[list, object]], size: int, lower: list[Fraction]
) -> list[Fraction] | None:
    """Точка `x ⩾ lower`, удовлетворяющая всем уравнениям `Σ aₖ·xₖ = b`.

    Равенства снимаются методом Гаусса, остаток — неравенства на свободные
    переменные, и они исключаются по Фурье–Моцкину. Всё в дробях, поэтому
    ответ точный: «решений нет» здесь означает именно это. Ответ даётся
    в рациональных числах — для **опровержения** этого достаточно
    (нет рационального решения ⇒ нет и целого), а вот найденная точка
    целой быть не обязана, поэтому она годится только как «не опровергнуто».
    """
    equations = [
        [Fraction(c) for c in row] + [Fraction(rhs)] for row, rhs in system
    ]

    # Гаусс. Строка `e` с ведущей позицией `c` хранится в приведённом виде:
    # `e[c] = 1`, коэффициенты при остальных ведущих позициях занулены,
    # `e[size]` — правая часть. Читается она как
    # `M[c] = e[size] − Σ_{k свободна} e[k]·M[k]`.
    pivots: dict[int, list[Fraction]] = {}
    for row in equations:
        row = row[:]
        for column, expression in pivots.items():
            if row[column]:
                factor = row[column]
                for k in range(size + 1):
                    row[k] -= factor * expression[k]
        column = next((k for k in range(size) if row[k]), None)
        if column is None:
            if row[size]:
                return None  # 0 = ненулевая константа
            continue
        factor = row[column]
        normalized = [value / factor for value in row]
        for expression in pivots.values():
            if expression[column]:
                weight = expression[column]
                for k in range(size + 1):
                    expression[k] -= weight * normalized[k]
        pivots[column] = normalized

    free = [k for k in range(size) if k not in pivots]
    # M[k] как функция свободных переменных: коэффициенты при них плюс константа.
    formulas: list[tuple[list[Fraction], Fraction]] = []
    for k in range(size):
        if k in pivots:
            expression = pivots[k]
            formulas.append(([-expression[j] for j in free], expression[size]))
        else:
            coefficients = [Fraction(1) if j == k else Fraction(0) for j in free]
            formulas.append((coefficients, Fraction(0)))

    inequalities = [
        (coefficients, constant - lower[k]) for k, (coefficients, constant) in enumerate(formulas)
    ]
    point = _fourier_motzkin(inequalities, len(free))
    if point is None:
        return None
    return [sum((c * t for c, t in zip(coefficients, point)), constant)
            for coefficients, constant in formulas]


def _fourier_motzkin(rows, dimension: int) -> list[Fraction] | None:
    """Точка, удовлетворяющая всем `Σ cₖ·tₖ + b ⩾ 0`, либо `None`."""
    if dimension == 0:
        return [] if all(constant >= 0 for _, constant in rows) else None

    lower, upper, rest = [], [], []
    for coefficients, constant in rows:
        head, weight = coefficients[:-1], coefficients[-1]
        if weight == 0:
            rest.append((head, constant))
        elif weight > 0:
            lower.append((head, constant, weight))
        else:
            upper.append((head, constant, -weight))

    combined = list(rest)
    for head_low, constant_low, weight_low in lower:
        for head_high, constant_high, weight_high in upper:
            merged = [
                weight_low * a + weight_high * b for a, b in zip(head_high, head_low)
            ]
            combined.append(
                (merged, weight_low * constant_high + weight_high * constant_low)
            )

    partial = _fourier_motzkin(combined, dimension - 1)
    if partial is None:
        return None

    def value_of(head, constant, weight):
        return (-constant - sum(c * t for c, t in zip(head, partial))) / weight

    bounds_low = [value_of(*item) for item in lower]
    bounds_high = [
        (sum(c * t for c, t in zip(head, partial)) + constant) / weight
        for head, constant, weight in upper
    ]
    if bounds_low:
        chosen = max(bounds_low)
    elif bounds_high:
        chosen = min(bounds_high)
    else:
        chosen = Fraction(0)
    return [*partial, chosen]
