"""µ-выражения Клини: регулярные операции плюс оператор неподвижной точки.

Определение курса — лекция 8 (2022), слайд 7:

> Определим оператор минимальной неподвижной точки: скажем, что `L(µX.r(X))`
> — это объединение языков `r(X)`, `r(r(X))`, …, `rⁿ(X)`.
> µ-оператор по переменным + регулярные операции определяют множество
> всех КС-языков.

Отсюда способ решать задачи вида «описать язык µ-выражения»: перевести
выражение в КС-грамматику и дальше работать готовым аппаратом —
перечислять язык, проверять гипотезу предикатом, чистить, строить
автоматы. Своей теории µ-выражениям не нужно, они и есть грамматики
в другой записи.

Контрольные точки — обе из курса: `µX.aXb | ε` задаёт `{aⁿbⁿ}`
(лекция 13, 2021), а `µy.a(µx.axb + y + a)` — грамматику
`Y → aX`, `X → aXb | Y | a` (лекция 8, 2022).

**Что считать переменной, решает связывание, а не регистр.** В лекции 8
переменные строчные (`µy.a(µx.…)`), а буквы — тоже строчные; в билете
2022 переменные заглавные (`µX.(a(µY.…)bX|ε)`). Угадывать по регистру
нельзя и не нужно: имя — переменная тогда и только тогда, когда его
связывает объемлющий `µ`. Всё остальное — буквы алфавита.
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, proved, refuted

__all__ = [
    "Expr",
    "Epsilon",
    "Letter",
    "Var",
    "Alt",
    "Cat",
    "Star",
    "Mu",
    "MuExpression",
    "parse_mu",
]

MU = "µμ"
ALTERNATION = "|+"
EPSILON_TOKENS = "εεϵ@"


# --------------------------------------------------------------------------
# Синтаксическое дерево
# --------------------------------------------------------------------------


class Expr:
    """Базовый класс узла µ-выражения."""


@dataclass(frozen=True)
class Epsilon(Expr):
    def __str__(self) -> str:
        return "ε"


@dataclass(frozen=True)
class Letter(Expr):
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Alt(Expr):
    parts: tuple[Expr, ...]

    def __str__(self) -> str:
        return " | ".join(str(p) for p in self.parts)


@dataclass(frozen=True)
class Cat(Expr):
    parts: tuple[Expr, ...]

    def __str__(self) -> str:
        # Скобки нужны и вокруг альтернативы, и вокруг `µ`: тело `µ` тянется
        # вправо до упора, поэтому без скобок `aµX.b` прочтётся иначе.
        return "".join(
            f"({p})" if isinstance(p, (Alt, Mu)) else str(p) for p in self.parts
        )


@dataclass(frozen=True)
class Star(Expr):
    body: Expr

    def __str__(self) -> str:
        inner = str(self.body)
        return f"({inner})*" if len(inner) > 1 else f"{inner}*"


@dataclass(frozen=True)
class Mu(Expr):
    name: str
    body: Expr

    def __str__(self) -> str:
        return f"µ{self.name}.{self.body}"


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def parse_mu(text: str) -> MuExpression:
    """Разобрать µ-выражение.

    Приоритеты обычные: альтернатива слабее конкатенации, конкатенация
    слабее звёздочки. Тело `µX.` — **вся** альтернатива справа, до
    закрывающей скобки или до конца: иначе `µX.aXb | ε` разобралось бы
    как `(µX.aXb) | ε` и задавало бы не тот язык.

    >>> str(parse_mu("µX.aXb | ε").expression)
    'µX.aXb | ε'
    """
    position = 0
    bound: list[str] = []

    def peek() -> str | None:
        while position < len(text) and text[position].isspace():
            skip()
        return text[position] if position < len(text) else None

    def skip() -> None:
        nonlocal position
        position += 1

    def parse_alt() -> Expr:
        parts = [parse_cat()]
        while peek() in tuple(ALTERNATION):
            skip()
            parts.append(parse_cat())
        return parts[0] if len(parts) == 1 else Alt(tuple(parts))

    def parse_cat() -> Expr:
        parts: list[Expr] = []
        while True:
            char = peek()
            if char is None or char in ALTERNATION or char == ")":
                break
            parts.append(parse_postfix())
        if not parts:
            return Epsilon()
        return parts[0] if len(parts) == 1 else Cat(tuple(parts))

    def parse_postfix() -> Expr:
        node = parse_atom()
        while peek() == "*":
            skip()
            node = Star(node)
        return node

    def parse_atom() -> Expr:
        char = peek()
        if char is None:
            raise ValueError(f"неожиданный конец выражения: {text!r}")
        if char == "(":
            skip()
            inner = parse_alt()
            if peek() != ")":
                raise ValueError(f"не закрыта скобка в {text!r}")
            skip()
            return inner
        if char in MU:
            skip()
            name = peek()
            if name is None or not name.isalpha():
                raise ValueError(f"после µ ожидается имя переменной: {text!r}")
            skip()
            if peek() != ".":
                raise ValueError(f"после «µ{name}» ожидается точка: {text!r}")
            skip()
            bound.append(name)
            body = parse_alt()
            bound.pop()
            return Mu(name, body)
        if char in EPSILON_TOKENS:
            skip()
            return Epsilon()
        if char in ".*)|+":
            raise ValueError(f"неожиданный символ «{char}» в {text!r}")
        skip()
        # Переменная или буква — решает связывание, а не регистр.
        return Var(char) if char in bound else Letter(char)

    node = parse_alt()
    if position != len(text.rstrip()):
        raise ValueError(f"лишние символы в {text!r} с позиции {position}")
    return MuExpression(node)


# --------------------------------------------------------------------------
# Выражение целиком
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MuExpression:
    """µ-выражение вместе с переводом в грамматику."""

    expression: Expr

    def __str__(self) -> str:
        return str(self.expression)

    def letters(self) -> frozenset[str]:
        def walk(node: Expr) -> frozenset[str]:
            if isinstance(node, Letter):
                return frozenset({node.value})
            if isinstance(node, (Alt, Cat)):
                return frozenset().union(*(walk(p) for p in node.parts))
            if isinstance(node, Star):
                return walk(node.body)
            if isinstance(node, Mu):
                return walk(node.body)
            return frozenset()

        return walk(self.expression)

    def free_variables(self) -> frozenset[str]:
        """Переменные, не связанные ни одним `µ`.

        Разбор их и не порождает — незнакомое имя становится буквой, —
        так что непустой ответ означает ошибку в дереве, собранном руками.
        """

        def walk(node: Expr, bound: frozenset[str]) -> frozenset[str]:
            if isinstance(node, Var):
                return frozenset() if node.name in bound else frozenset({node.name})
            if isinstance(node, (Alt, Cat)):
                return frozenset().union(*(walk(p, bound) for p in node.parts))
            if isinstance(node, Star):
                return walk(node.body, bound)
            if isinstance(node, Mu):
                return walk(node.body, bound | {node.name})
            return frozenset()

        return walk(self.expression, frozenset())

    def to_cfg(self) -> CFG:
        """Перевести в КС-грамматику: каждый `µ` — нетерминал, регулярные
        операции — вспомогательные нетерминалы.

        `µX.r` даёт правила `X → …` по альтернативам `r`; звёздочка
        разворачивается в правую рекурсию `A → r A | ε`; альтернатива
        внутри конкатенации получает собственный нетерминал.
        """
        terminals = self.letters()
        productions: list[Production] = []
        names: dict[str, str] = {}
        counter = [0]

        def fresh(hint: str = "") -> str:
            candidate = hint.upper()
            if (
                candidate
                and candidate.isalpha()
                and candidate not in terminals
                and candidate not in names.values()
            ):
                return candidate
            counter[0] += 1
            return f"N{counter[0]}"

        def symbols(node: Expr) -> tuple[str, ...]:
            if isinstance(node, Epsilon):
                return ()
            if isinstance(node, Letter):
                return (node.value,)
            if isinstance(node, Var):
                return (names[node.name],)
            if isinstance(node, Cat):
                return tuple(s for part in node.parts for s in symbols(part))
            if isinstance(node, Alt):
                head = fresh()
                for part in node.parts:
                    productions.append(Production(head, symbols(part)))
                return (head,)
            if isinstance(node, Star):
                head = fresh()
                productions.append(Production(head, (*symbols(node.body), head)))
                productions.append(Production(head, ()))
                return (head,)
            if isinstance(node, Mu):
                head = fresh(node.name)
                names[node.name] = head
                body = node.body
                alternatives = body.parts if isinstance(body, Alt) else (body,)
                for part in alternatives:
                    productions.append(Production(head, symbols(part)))
                return (head,)
            raise TypeError(f"неизвестный узел: {node!r}")

        root = symbols(self.expression)
        if len(root) == 1 and root[0] not in terminals:
            start = root[0]
        else:
            start = fresh()
            productions.append(Production(start, root))
        heads = frozenset(p.lhs for p in productions)
        return CFG(start, tuple(productions), heads, terminals)

    # ------------------------------------------------------------- оракул

    def words(self, max_len: int) -> list[str]:
        """Слова языка длины не больше данной, по возрастанию."""
        from tfl.parse import language

        return sorted(language(self.to_cfg(), max_len), key=lambda w: (len(w), w))

    def accepts(self, word: str) -> bool:
        from tfl.parse import recognize

        return recognize(self.to_cfg(), word)

    def agrees_with(self, predicate, max_len: int = 8, alphabet: str | None = None) -> Verdict:
        """Сверить язык выражения с гипотезой-предикатом на всех коротких словах.

        Именно так и решается билетный вопрос «описать язык»: описание
        придумывает человек, а совпадение проверяет оракул. Расхождение
        возвращается свидетелем — конкретным словом и тем, кто из двоих
        его принимает.
        """
        letters = "".join(sorted(alphabet or self.letters()))
        mine = set(self.words(max_len))
        checked = 0
        for length in range(max_len + 1):
            for word in _words_of_length(letters, length):
                checked += 1
                here, there = word in mine, bool(predicate(word))
                if here != there:
                    who = "выражение" if here else "предикат"
                    return refuted(
                        f"расхождение на слове «{word or 'ε'}»: его принимает "
                        f"только {who}",
                        word,
                    )
        return proved(f"совпадение на всех {checked} словах длины ≤ {max_len}")


def _words_of_length(letters: str, length: int):
    if length == 0:
        yield ""
        return
    for prefix in _words_of_length(letters, length - 1):
        for letter in letters:
            yield prefix + letter
