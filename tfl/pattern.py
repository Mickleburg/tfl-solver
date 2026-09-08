"""Переписывание образцов: правила с переменной, `aXb → bXa`.

Семинар 05.09.2026, группа 52-Б, задача 1: система `P = {aXb → bXa,
Xb → aaX}` — завершима ли, конфлюэнтна ли. `tfl/srs.py` работает
с плоскими строками, и такие правила через него не выражаются:
переменная покрывает **произвольный кусок** слова, а не один символ.

Переменные объявляются явно — по той же причине, что в `tfl/trs.py`:
в корпусе встречаются обе договорённости о регистре, и угадывать нельзя.

Что здесь механизировано и что нет. Шаг, достижимость, нормальные формы
и поиск петли — считаются. Завершимость и конфлюэнтность в общем виде
не решаются, поэтому вместо них два арбитра:

* `check_measure` — проверить **предложенную** фундированную меру;
* `check_invariant` — проверить **предложенный** инвариант.

Меру и инвариант придумывает человек, оракул их проверяет на всех
коротких словах. Ровно так семинар и разбирал задачу: мера
`(|w|_b, Σ позиций b)` в лексикографическом порядке и инвариант
`μ(w) = |w|_a + 2|w|_b`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "PatternRule",
    "PatternSystem",
    "parse_patterns",
    "matches",
]

ARROWS = ("->", "→", "-->", "=>")


@dataclass(frozen=True)
class PatternRule:
    """Правило `aXb → bXa`: слева и справа слова над Σ ∪ переменные."""

    lhs: str
    rhs: str

    def __str__(self) -> str:
        return f"{self.lhs or 'ε'} → {self.rhs or 'ε'}"


def matches(
    pattern: str, word: str, start: int, variables: frozenset[str], binding: dict[str, str]
) -> Iterator[tuple[int, dict[str, str]]]:
    """Все способы сопоставить образец с куском слова, начиная с позиции.

    Возвращает пары «конец куска, подстановка». Повторная переменная
    обязана получить то же значение — поэтому `XaX` сопоставляется
    только с `uau`.
    """
    if not pattern:
        yield start, binding
        return
    head, rest = pattern[0], pattern[1:]
    if head in variables:
        if head in binding:
            value = binding[head]
            if word.startswith(value, start):
                yield from matches(rest, word, start + len(value), variables, binding)
            return
        for end in range(start, len(word) + 1):
            yield from matches(
                rest, word, end, variables, {**binding, head: word[start:end]}
            )
        return
    if start < len(word) and word[start] == head:
        yield from matches(rest, word, start + 1, variables, binding)


@dataclass(frozen=True)
class PatternSystem:
    """Система переписывания образцов."""

    rules: tuple[PatternRule, ...]
    variables: frozenset[str] = field(default=frozenset())

    def __len__(self) -> int:
        return len(self.rules)

    def __str__(self) -> str:
        return "\n".join(str(rule) for rule in self.rules)

    @property
    def alphabet(self) -> frozenset[str]:
        """Буквы системы — всё, что в правилах не переменная."""
        return frozenset(
            symbol
            for rule in self.rules
            for symbol in rule.lhs + rule.rhs
            if symbol not in self.variables
        )

    def substitute(self, pattern: str, binding: dict[str, str]) -> str:
        return "".join(
            binding.get(symbol, symbol) if symbol in self.variables else symbol
            for symbol in pattern
        )

    def step(self, word: str) -> set[str]:
        """Все слова, получаемые одним применением правила в любом месте."""
        results: set[str] = set()
        for rule in self.rules:
            for start in range(len(word) + 1):
                for end, binding in matches(rule.lhs, word, start, self.variables, {}):
                    results.add(word[:start] + self.substitute(rule.rhs, binding) + word[end:])
        results.discard(word)
        return results

    def reachable(self, word: str, max_len: int = 12, max_nodes: int = 20_000):
        """Достижимые слова и признак того, что обход был полным."""
        seen = {word}
        frontier = {word}
        truncated = False
        while frontier:
            following: set[str] = set()
            for current in frontier:
                for nxt in self.step(current):
                    if len(nxt) > max_len:
                        truncated = True
                        continue
                    if nxt not in seen:
                        if len(seen) >= max_nodes:
                            truncated = True
                            break
                        seen.add(nxt)
                        following.add(nxt)
            frontier = following
        return seen, truncated

    def normal_forms(self, word: str, max_len: int = 12) -> set[str]:
        """Слова без применимых правил среди достижимых."""
        found, _ = self.reachable(word, max_len)
        return {candidate for candidate in found if not self.step(candidate)}

    def find_loop(self, words, max_len: int = 12, max_nodes: int = 20_000):
        """Петля `w →⁺ u·w·v`. Цикл — её частный случай.

        Тот же приём, что в `tfl/srs.py`: правило с левой частью, вложенной
        в правую, слово не повторяет, а наращивает, и циклом такое
        не ловится в принципе.
        """
        for start in words:
            budget = max_nodes
            frontier = {start: (start,)}
            seen = {start}
            while frontier and budget > 0:
                following: dict[str, tuple[str, ...]] = {}
                for word, path in frontier.items():
                    for nxt in sorted(self.step(word)):
                        if len(nxt) > max_len:
                            continue
                        if start in nxt:
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

    # ------------------------------------------------------------ арбитры

    def check_measure(
        self, measure: Callable[[str], object], words, max_len: int = 12
    ) -> Verdict:
        """Убывает ли предложенная мера на каждом шаге.

        Мера возвращает что угодно сравнимое — число либо кортеж
        (лексикографический порядок). Убывание на **всех** шагах вместе
        с фундированностью меры доказывает завершимость; фундированность
        остаётся на совести автора меры, и вердикт об этом говорит.
        """
        checked = 0
        for word in words:
            found, _ = self.reachable(word, max_len)
            for current in found:
                for nxt in self.step(current):
                    checked += 1
                    if not measure(nxt) < measure(current):
                        return refuted(
                            f"мера не убывает: «{current}» → «{nxt}», "
                            f"{measure(current)} ⩾ {measure(nxt)}",
                            (current, nxt),
                        )
        return proved(
            f"мера убывает на всех {checked} шагах из данных слов. "
            f"Завершимость отсюда следует, **если** мера фундирована — "
            f"это проверяет автор меры, а не оракул"
        )

    def check_invariant(
        self, invariant: Callable[[str], object], words, max_len: int = 12
    ) -> Verdict:
        """Сохраняется ли предложенная величина на каждом шаге."""
        checked = 0
        for word in words:
            found, _ = self.reachable(word, max_len)
            for current in found:
                for nxt in self.step(current):
                    checked += 1
                    if invariant(nxt) != invariant(current):
                        return refuted(
                            f"величина не сохраняется: «{current}» → «{nxt}», "
                            f"{invariant(current)} ≠ {invariant(nxt)}",
                            (current, nxt),
                        )
        return proved(f"величина сохраняется на всех {checked} шагах из данных слов")

    def confluent_on(self, words, max_len: int = 12) -> Verdict:
        """У каждого слова ровно одна нормальная форма — на данном срезе.

        Это не конфлюэнтность: срез конечен. Но расхождение, если оно есть,
        предъявляется словом с двумя нормальными формами, и такой ответ
        уже доказателен.
        """
        for word in words:
            forms = self.normal_forms(word, max_len)
            if len(forms) > 1:
                return refuted(
                    f"у «{word}» больше одной нормальной формы: "
                    + ", ".join(f"«{form}»" for form in sorted(forms)),
                    (word, sorted(forms)),
                )
        return unknown(
            f"на всех проверенных словах нормальная форма единственна; "
            f"конфлюэнтностью это не является — проверен конечный срез"
        )

    def agrees_with_srs(self, system, words, wrap=None, max_len: int = 12) -> Verdict:
        """Совпадают ли достижимые множества с моделирующей строковой системой.

        Приём семинара: образец с переменной моделируется маркером-долгом,
        который ходит по слову. Сравнивать надо **достижимые множества**,
        причём у строковой системы — только слова без маркеров.

        `wrap` оборачивает слово в краевые символы (`^w$`), `system` —
        `tfl.srs.SRS`. Расхождение возвращается со свидетелем.

        В сравнение идут только **чистые** строки: те, где остались одни
        буквы образцовой системы. Строка с недогашенным маркером вида
        `^Ma$` промежуточная, и считать её достижимым словом нельзя —
        иначе моделирование «добавляет» всё подряд.

        Потолок длины у обеих сторон обязан быть одинаковым. Строковой
        системе нужен запас на маркеры, поэтому её обход идёт дальше,
        но в сравнение всё равно берутся слова не длиннее `max_len` —
        иначе «расхождением» окажется разница границ, а не языков.
        """
        wrap = wrap or (lambda word: word)
        letters = self.alphabet
        for word in words:
            mine, _ = self.reachable(word, max_len)
            search = system.reachable(wrap(word), max_len + 4)
            theirs = {
                candidate.strip("^$")
                for candidate in search.words
                if candidate.startswith("^")
                and candidate.endswith("$")
                and set(candidate.strip("^$")) <= letters
                and len(candidate.strip("^$")) <= max_len
            }
            if mine != theirs:
                missing = sorted(mine - theirs) or sorted(theirs - mine)
                where = "теряет" if mine - theirs else "добавляет"
                return refuted(
                    f"на слове «{word}» строковая система {where} «{missing[0]}»",
                    (word, missing[0]),
                )
        return proved(
            f"достижимые множества совпали на всех {len(list(words))} словах"
        )


def parse_patterns(text: str, variables: str = "") -> PatternSystem:
    """Разобрать систему. Переменные перечисляются явно.

    >>> str(parse_patterns("aXb -> bXa", variables="X"))
    'aXb → bXa'
    """
    names = frozenset(variables.replace(",", " ").split())
    rules: list[PatternRule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("variables"):
            _, _, listed = line.partition("=")
            names = frozenset(listed.strip(" []").replace(",", " ").split())
            continue
        for arrow in ARROWS:
            if arrow in line:
                left, right = line.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {raw!r}")
        rules.append(PatternRule(left.strip(), right.strip().replace("ε", "")))
    if not rules:
        raise ValueError("система пуста")
    free = {
        symbol
        for rule in rules
        for symbol in rule.rhs
        if symbol in names and symbol not in rule.lhs
    }
    if free:
        raise ValueError(
            f"переменные {sorted(free)} есть справа, но не слева: "
            f"подставлять в них нечего"
        )
    return PatternSystem(tuple(rules), names)
