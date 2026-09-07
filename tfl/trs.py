"""Системы переписывания термов: переменные, унификация, интерпретации.

`tfl/srs.py` работает со строками. Здесь — термы с переменными, то есть
настоящие TRS: раздел «F. КЗ-языки» «Аптеки», задачи вида

    variables = [x, y]
    f(f(a, x), y) -> f(f(x, f(a, y)), a)

и вопросы из банка 2024 года, где к системе сразу прилагается
интерпретация (`f(x) := x+1; g(x,y) := x + y*2 + 1`).

**Переменные объявляются явно, и это не педантизм.** В корпусе две
противоположные договорённости: в issue #2 переменные строчные,
а конструкторы заглавные (`F(q,q,q)`), в «Аптеке» ровно наоборот —
`variables = [X]`, а `E`, `Q`, `W` это функциональные символы. Угадывать
по регистру нельзя: `w(w(w(s,X),Y),Z)` развалится в любую сторону.
Поэтому `parse_trs` читает заголовок `variables = [...]`, а `parse_term`
требует множество переменных аргументом.

Что здесь механизировано: разбор, подстановка, сопоставление, унификация
с проверкой вхождения, переписывание, поиск петли, полиномиальные
интерпретации и мост в `tfl/srs.py` для унарных систем. Чего нет:
критических пар и пополнения (см. `docs/OPEN-GAPS.md`).

Про границы вывода. Завершимость TRS неразрешима, поэтому `terminates`
возвращает `Verdict` с тремя исходами, как и его строковый двойник.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "Term",
    "var",
    "app",
    "Rule",
    "TRS",
    "Poly",
    "Interpretation",
    "parse_term",
    "parse_trs",
    "parse_interpretation",
    "substitute",
    "match",
    "unify",
]


# --------------------------------------------------------------------------
# Термы
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """Терм: переменная либо применение функционального символа."""

    head: str
    args: tuple[Term, ...] = ()
    variable: bool = False

    def __str__(self) -> str:
        if self.variable or not self.args:
            return self.head
        return f"{self.head}({', '.join(str(a) for a in self.args)})"

    @property
    def arity(self) -> int:
        return len(self.args)

    def variables(self) -> frozenset[str]:
        if self.variable:
            return frozenset({self.head})
        return frozenset().union(*(a.variables() for a in self.args)) if self.args else frozenset()

    def symbols(self) -> dict[str, int]:
        """Функциональные символы и их арности."""
        if self.variable:
            return {}
        found = {self.head: len(self.args)}
        for arg in self.args:
            found.update(arg.symbols())
        return found

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def positions(self):
        """Пары «позиция, подтерм» — позиция это путь из индексов аргументов."""
        yield (), self
        for index, arg in enumerate(self.args):
            for path, sub in arg.positions():
                yield (index, *path), sub

    def replace(self, path: tuple[int, ...], replacement: Term) -> Term:
        if not path:
            return replacement
        head, *rest = path
        args = list(self.args)
        args[head] = args[head].replace(tuple(rest), replacement)
        return Term(self.head, tuple(args), self.variable)

    def contains(self, other: Term) -> bool:
        return any(sub == other for _, sub in self.positions())


def var(name: str) -> Term:
    return Term(name, (), True)


def app(head: str, *args: Term) -> Term:
    return Term(head, tuple(args), False)


# --------------------------------------------------------------------------
# Подстановка, сопоставление, унификация
# --------------------------------------------------------------------------


def substitute(term: Term, binding: dict[str, Term]) -> Term:
    if term.variable:
        return binding.get(term.head, term)
    return Term(term.head, tuple(substitute(a, binding) for a in term.args), False)


def match(pattern: Term, term: Term) -> dict[str, Term] | None:
    """Одностороннее сопоставление: подставляем **только** в образец.

    Именно это нужно для переписывания: левая часть правила подгоняется
    под подтерм, а подтерм не трогается. Унификация здесь была бы ошибкой —
    она позволила бы «доопределить» переписываемый терм.
    """
    binding: dict[str, Term] = {}
    stack = [(pattern, term)]
    while stack:
        left, right = stack.pop()
        if left.variable:
            if left.head in binding:
                if binding[left.head] != right:
                    return None
            else:
                binding[left.head] = right
        elif right.variable or left.head != right.head or left.arity != right.arity:
            return None
        else:
            stack.extend(zip(left.args, right.args))
    return binding


def unify(left: Term, right: Term) -> dict[str, Term] | None:
    """Наибольший общий унификатор или `None`, если термы не унифицируются.

    Проверка вхождения обязательна: без неё `x` и `F(x)` «унифицируются»
    подстановкой в бесконечный терм.

    Это результат алгоритма Мартелли–Монтанари, но не его ход.
    Преподаватель в issue #2 требует именно **алгоритм 3** с представлением
    через мультиуравнения; здесь считается тот же наибольший общий
    унификатор обычным способом. Ответ сверяется с разобранным примером
    преподавателя, промежуточные шаги — нет.
    """
    binding: dict[str, Term] = {}
    stack = [(left, right)]
    while stack:
        a, b = stack.pop()
        a, b = _walk(a, binding), _walk(b, binding)
        if a == b:
            continue
        if a.variable or b.variable:
            if not a.variable:
                a, b = b, a
            if substitute(b, binding).contains(a):
                return None  # проверка вхождения
            binding = {name: substitute(t, {a.head: b}) for name, t in binding.items()}
            binding[a.head] = b
        elif a.head != b.head or a.arity != b.arity:
            return None
        else:
            stack.extend(zip(a.args, b.args))
    return {name: _resolve(t, binding) for name, t in binding.items()}


def _walk(term: Term, binding: dict[str, Term]) -> Term:
    while term.variable and term.head in binding:
        term = binding[term.head]
    return term


def _resolve(term: Term, binding: dict[str, Term]) -> Term:
    resolved = substitute(term, binding)
    while resolved != term:
        term, resolved = resolved, substitute(resolved, binding)
    return resolved


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------

# Числовые константы (`0`, `1`) в условиях встречаются наравне с именами:
# `plus(0, y) → y` — обычная запись сложения по Пеано.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9']*|\d+|[(),]")


def parse_term(text: str, variables: frozenset[str] | set[str] | str) -> Term:
    """Разобрать терм. `variables` — имена, считающиеся переменными.

    >>> str(parse_term("f(f(a, x), y)", {"x", "y"}))
    'f(f(a, x), y)'
    """
    names = frozenset(variables.split()) if isinstance(variables, str) else frozenset(variables)
    tokens = TOKEN.findall(text)
    position = 0

    def parse() -> Term:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"неожиданный конец терма: {text!r}")
        head = tokens[position]
        position += 1
        if head in "(),":
            raise ValueError(f"ожидалось имя, а не «{head}» в {text!r}")
        if position < len(tokens) and tokens[position] == "(":
            position += 1
            args = [parse()]
            while tokens[position] == ",":
                position += 1
                args.append(parse())
            if tokens[position] != ")":
                raise ValueError(f"не закрыта скобка в {text!r}")
            position += 1
            if head in names:
                raise ValueError(f"переменная «{head}» не может иметь аргументов")
            return Term(head, tuple(args), False)
        return Term(head, (), head in names)

    result = parse()
    if position != len(tokens):
        raise ValueError(f"лишние символы после терма в {text!r}")
    return result


@dataclass(frozen=True)
class Rule:
    """Правило `l → r`."""

    lhs: Term
    rhs: Term

    def __post_init__(self) -> None:
        if self.lhs.variable:
            raise ValueError(f"левая часть правила не может быть переменной: {self}")
        extra = self.rhs.variables() - self.lhs.variables()
        if extra:
            raise ValueError(
                f"в правой части правила {self} появляются свободные переменные "
                f"{sorted(extra)}: переписывание было бы недетерминированным"
            )

    def __str__(self) -> str:
        return f"{self.lhs} → {self.rhs}"


VARIABLES_HEADER = re.compile(r"^\s*(?:variables\s*=\s*)?\[([^\]]*)\]\s*$")
ARROWS = ("-->", "->", "→", "=>")


def parse_trs(text: str, variables: str = "") -> TRS:
    """Разобрать систему. Заголовок `variables = [x, y]` (или просто `[x, y]`).

    >>> system = parse_trs("variables = [x, y]\\nf(f(a, x), y) -> f(f(x, f(a, y)), a)")
    >>> len(system)
    1
    >>> sorted(system.variables)
    ['x', 'y']
    """
    names = set(variables.replace(",", " ").split())
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        header = VARIABLES_HEADER.match(line)
        if header:
            names |= {n.strip() for n in header.group(1).split(",") if n.strip()}
            continue
        lines.append(line)

    rules = []
    for line in lines:
        for arrow in ARROWS:
            if arrow in line:
                left, right = line.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {line!r}")
        rules.append(Rule(parse_term(left, names), parse_term(right, names)))
    return TRS(tuple(rules), frozenset(names))


# --------------------------------------------------------------------------
# Система
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TRS:
    """Система переписывания термов."""

    rules: tuple[Rule, ...]
    variables: frozenset[str] = field(default=frozenset())

    def __len__(self) -> int:
        return len(self.rules)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.rules)

    def signature(self) -> dict[str, int]:
        found: dict[str, int] = {}
        for rule in self.rules:
            found.update(rule.lhs.symbols())
            found.update(rule.rhs.symbols())
        return found

    def step(self, term: Term) -> set[Term]:
        """Все термы, получаемые одним применением правила в любой позиции."""
        results = set()
        for path, sub in term.positions():
            for rule in self.rules:
                binding = match(rule.lhs, sub)
                if binding is not None:
                    results.add(term.replace(path, substitute(rule.rhs, binding)))
        return results

    def normal_form(self, term: Term, budget: int = 10_000) -> Term | None:
        """Одна нормальная форма, если её удалось достичь за бюджет шагов."""
        current = term
        for _ in range(budget):
            following = self.step(current)
            if not following:
                return current
            current = min(following, key=lambda t: (t.size(), str(t)))
        return None

    # ---------------------------------------------------------- завершимость

    def start_terms(self, depth: int = 1) -> list[Term]:
        """Термы для запуска поиска: левые части и их мелкие обобщения."""
        seeds = {rule.lhs for rule in self.rules}
        constants = [Term(s, (), False) for s, n in self.signature().items() if n == 0]
        if not constants:
            constants = [var("z")]
        for rule in self.rules:
            names = sorted(rule.lhs.variables())
            for values in itertools.product(constants, repeat=len(names)):
                seeds.add(substitute(rule.lhs, dict(zip(names, values))))
        return sorted(seeds, key=lambda t: (t.size(), str(t)))[: 40 * depth]

    def find_loop(self, max_size: int = 30, max_nodes: int = 20_000) -> list[Term] | None:
        """Найти петлю `t →⁺ C[t]` — терм, содержащий исходный как подтерм.

        Цикл `t →⁺ t` — частный случай. Как и в `tfl/srs.py`, петля ловит
        неограниченный рост, который циклом не ловится: терм растёт,
        ни разу не повторившись.
        """
        budget = max_nodes
        for start in self.start_terms():
            if start.size() > max_size:
                continue
            frontier: dict[Term, tuple[Term, ...]] = {start: (start,)}
            seen = {start}
            while frontier and budget > 0:
                following: dict[Term, tuple[Term, ...]] = {}
                for term, path in frontier.items():
                    for nxt in sorted(self.step(term), key=str):
                        if nxt.size() > max_size:
                            continue
                        if nxt.contains(start):
                            return [*path, nxt]
                        if nxt in seen:
                            continue
                        budget -= 1
                        if budget <= 0:
                            break
                        seen.add(nxt)
                        following[nxt] = (*path, nxt)
                frontier = following
        return None

    def terminates(self, max_size: int = 30) -> Verdict:
        """Вердикт о завершимости: петля, интерпретация или «не выяснено»."""
        loop = self.find_loop(max_size=max_size)
        if loop is not None:
            kind = "цикл" if loop[0] == loop[-1] else "петля"
            return refuted(
                f"найден{'' if kind == 'цикл' else 'а'} {kind}: "
                + " → ".join(str(t) for t in loop),
                loop,
            )
        found = self.find_interpretation()
        if found.value is True:
            return found
        if self.is_unary():
            bridged = self.as_srs().terminates()
            if bridged.value is not None:
                return Verdict(
                    bridged.value,
                    f"через строковую систему: {bridged.reason}",
                    bridged.witness,
                )
        return unknown(
            "ни петли, ни полиномиальной интерпретации с малыми "
            "коэффициентами не нашлось; нужен более сильный порядок "
            "(рекурсивный по путям, матричные интерпретации)"
        )

    # ------------------------------------------------------- мост в строки

    def is_unary(self) -> bool:
        """Все ли функциональные символы одноместные.

        Такая система — это система переписывания **строк**: цепочка
        `f(g(h(X)))` читается как слово `fgh`, а подтерм с переменной внизу
        это в точности подстрока. Значит, к ней применимо всё из `tfl/srs.py`.
        """
        signature = self.signature()
        return bool(signature) and all(n == 1 for n in signature.values())

    def as_srs(self):
        """Перевести унарную систему в строковую (`tfl.srs.SRS`).

        Требуется, чтобы каждая часть правила была цепочкой, кончающейся
        одной и той же переменной: `l(X) → r(X)`.
        """
        from tfl.srs import SRS, Rule as StringRule

        if not self.is_unary():
            raise ValueError("перевод в строки возможен только для унарных систем")

        def chain(term: Term) -> str:
            letters = []
            while not term.variable:
                letters.append(term.head)
                term = term.args[0]
            return "".join(letters)

        return SRS(tuple(StringRule(chain(r.lhs), chain(r.rhs)) for r in self.rules))

    # -------------------------------------------------- интерпретации

    def check_interpretation(self, interpretation: Interpretation) -> Verdict:
        """Убывает ли каждое правило при данной интерпретации."""
        for rule in self.rules:
            if not interpretation.decreases(rule):
                return refuted(
                    f"правило {rule} не убывает: "
                    f"[{rule.lhs}] = {interpretation.value(rule.lhs)}, "
                    f"[{rule.rhs}] = {interpretation.value(rule.rhs)}",
                    rule,
                )
        return proved(
            f"все правила убывают при интерпретации {interpretation}", interpretation
        )

    def find_interpretation(
        self, max_coefficient: int = 3, max_shift: int = 3, budget: int = 400_000
    ) -> Verdict:
        """Перебрать простые полиномиальные интерпретации.

        Для символа арности `n` берутся полиномы вида
        `c₀ + Σ cᵢ·xᵢ` и, при `n = 2`, дополнительно `+ d·x₁·x₂`.
        Коэффициенты при переменных не меньше единицы — иначе
        интерпретация не монотонна и вывод неправомерен.
        """
        signature = sorted(self.signature().items())
        if not signature:
            return unknown("в системе нет функциональных символов")

        options: list[list[Poly]] = []
        space = 1
        for symbol, arity in signature:
            variants = _templates(arity, max_coefficient, max_shift)
            options.append(variants)
            space *= len(variants)
            if space > budget:
                return unknown(
                    f"перебор {space} интерпретаций превышает бюджет {budget}"
                )

        for combination in itertools.product(*options):
            candidate = Interpretation(
                {symbol: poly for (symbol, _), poly in zip(signature, combination)}
            )
            if all(candidate.decreases(rule) for rule in self.rules):
                return proved(
                    "полиномиальная интерпретация убывает на всех правилах, "
                    f"значит система завершима: {candidate}",
                    candidate,
                )
        return unknown(
            f"полиномиальной интерпретации с коэффициентами до {max_coefficient} "
            f"и сдвигами до {max_shift} не нашлось; про завершимость это "
            "ничего не говорит"
        )


# --------------------------------------------------------------------------
# Полиномы
# --------------------------------------------------------------------------

#: Обозначения аргументов в шаблоне интерпретации символа.
ARGUMENT = "#{}".format


@dataclass(frozen=True)
class Poly:
    """Многочлен с целыми коэффициентами: одночлен → коэффициент.

    Одночлен — отсортированный кортеж пар «переменная, степень».

    >>> x, y = Poly.of("x"), Poly.of("y")
    >>> str(x * y + x + Poly.number(2))
    'x*y + x + 2'
    """

    monomials: tuple[tuple[tuple[tuple[str, int], ...], int], ...] = ()

    @staticmethod
    def number(value: int) -> Poly:
        return Poly((((), value),) if value else ())

    @staticmethod
    def of(name: str) -> Poly:
        return Poly(((((name, 1),), 1),))

    def as_dict(self) -> dict[tuple[tuple[str, int], ...], int]:
        return dict(self.monomials)

    @staticmethod
    def _pack(table: dict) -> Poly:
        return Poly(tuple(sorted((m, c) for m, c in table.items() if c)))

    def __add__(self, other: Poly) -> Poly:
        table = self.as_dict()
        for monomial, coefficient in other.monomials:
            table[monomial] = table.get(monomial, 0) + coefficient
        return Poly._pack(table)

    def __sub__(self, other: Poly) -> Poly:
        table = self.as_dict()
        for monomial, coefficient in other.monomials:
            table[monomial] = table.get(monomial, 0) - coefficient
        return Poly._pack(table)

    def __mul__(self, other: Poly) -> Poly:
        table: dict[tuple[tuple[str, int], ...], int] = {}
        for left, a in self.monomials:
            for right, b in other.monomials:
                degrees: dict[str, int] = dict(left)
                for name, power in right:
                    degrees[name] = degrees.get(name, 0) + power
                key = tuple(sorted(degrees.items()))
                table[key] = table.get(key, 0) + a * b
        return Poly._pack(table)

    def substitute(self, binding: dict[str, Poly]) -> Poly:
        result = Poly()
        for monomial, coefficient in self.monomials:
            piece = Poly.number(coefficient)
            for name, power in monomial:
                factor = binding.get(name, Poly.of(name))
                for _ in range(power):
                    piece = piece * factor
            result = result + piece
        return result

    def constant(self) -> int:
        return self.as_dict().get((), 0)

    def dominates(self, other: Poly) -> bool:
        """Строго больше на всех неотрицательных значениях переменных.

        Достаточное условие: у разности все коэффициенты неотрицательны,
        а свободный член строго положителен. Это не критерий, а именно
        достаточное условие — оно и используется в отчётах.
        """
        difference = self - other
        return difference.constant() > 0 and all(c >= 0 for _, c in difference.monomials)

    def __str__(self) -> str:
        if not self.monomials:
            return "0"
        parts = []
        for monomial, coefficient in sorted(
            self.monomials, key=lambda item: (-sum(p for _, p in item[0]), item[0])
        ):
            body = "*".join(n if p == 1 else f"{n}^{p}" for n, p in monomial)
            if not body:
                parts.append(str(coefficient))
            elif coefficient == 1:
                parts.append(body)
            else:
                parts.append(f"{coefficient}*{body}")
        return " + ".join(parts)


def _templates(arity: int, max_coefficient: int, max_shift: int) -> list[Poly]:
    """Шаблоны интерпретаций символа данной арности."""
    arguments = [Poly.of(ARGUMENT(i)) for i in range(arity)]
    out: list[Poly] = []
    for shift in range(max_shift + 1):
        for factors in itertools.product(range(1, max_coefficient + 1), repeat=arity):
            base = Poly.number(shift)
            for coefficient, argument in zip(factors, arguments):
                base = base + Poly.number(coefficient) * argument
            out.append(base)
            if arity == 2:
                for product in range(1, max_coefficient + 1):
                    out.append(
                        base + Poly.number(product) * arguments[0] * arguments[1]
                    )
    return out


@dataclass(frozen=True)
class Interpretation:
    """Каждому функциональному символу — многочлен от его аргументов."""

    symbols: dict[str, Poly]

    def __str__(self) -> str:
        parts = []
        for symbol, poly in sorted(self.symbols.items()):
            names = sorted({n for m, _ in poly.monomials for n, _ in m})
            arguments = ", ".join(names) if names else ""
            parts.append(f"[{symbol}]({arguments}) = {poly}")
        return "; ".join(parts)

    def value(self, term: Term) -> Poly:
        if term.variable:
            return Poly.of(term.head)
        template = self.symbols.get(term.head)
        if template is None:
            raise KeyError(f"нет интерпретации для символа «{term.head}»")
        binding = {ARGUMENT(i): self.value(arg) for i, arg in enumerate(term.args)}
        return template.substitute(binding)

    def decreases(self, rule: Rule) -> bool:
        return self.value(rule.lhs).dominates(self.value(rule.rhs))


# Скобки у константы необязательны: и `0() = 1`, и `0 = 1` встречаются.
ASSIGNMENT = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9']*|\d+)\s*(?:\(([^)]*)\))?\s*(?::=|=)\s*(.+)$"
)


def parse_interpretation(text: str) -> Interpretation:
    """Разобрать запись вида `f(x) := x+1; g(x,y) = x + y*2 + 1`.

    Понимает `+`, `*`, степень через `^`, числа и имена аргументов.

    >>> str(parse_interpretation("f(q) = q*q*q"))
    '[f](#0) = #0^3'
    """
    symbols: dict[str, Poly] = {}
    for chunk in re.split(r"[;\n]", text):
        if not chunk.strip():
            continue
        matched = ASSIGNMENT.match(chunk)
        if not matched:
            raise ValueError(f"не разобрать интерпретацию: {chunk!r}")
        name, arguments, body = matched.groups()
        names = [a.strip() for a in (arguments or "").split(",") if a.strip()]
        binding = {n: Poly.of(ARGUMENT(i)) for i, n in enumerate(names)}
        symbols[name] = _parse_expression(body, binding)
    return Interpretation(symbols)


def _parse_expression(text: str, binding: dict[str, Poly]) -> Poly:
    tokens = re.findall(r"\d+|[A-Za-z_][A-Za-z_0-9']*|[+*^()]", text)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def sum_() -> Poly:
        nonlocal position
        value = product()
        while peek() == "+":
            position += 1
            value = value + product()
        return value

    def product() -> Poly:
        nonlocal position
        value = power()
        while peek() == "*":
            position += 1
            value = value * power()
        return value

    def power() -> Poly:
        nonlocal position
        value = atom()
        if peek() == "^":
            position += 1
            exponent = int(tokens[position])
            position += 1
            result = Poly.number(1)
            for _ in range(exponent):
                result = result * value
            return result
        return value

    def atom() -> Poly:
        nonlocal position
        token = peek()
        if token is None:
            raise ValueError(f"неожиданный конец выражения: {text!r}")
        position += 1
        if token == "(":
            value = sum_()
            if peek() != ")":
                raise ValueError(f"не закрыта скобка в {text!r}")
            position += 1
            return value
        if token.isdigit():
            return Poly.number(int(token))
        return binding.get(token, Poly.of(token))

    result = sum_()
    if position != len(tokens):
        raise ValueError(f"лишние символы в выражении {text!r}")
    return result
