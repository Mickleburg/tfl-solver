"""Разбор по КС-грамматике: алгоритм Эрли и перечисление языка.

Это оракул принадлежности для всего, что связано с КС-грамматиками:
проверить гипотезу «грамматика порождает вот такой язык» можно только
сравнив её с независимым предикатом на всех коротких словах.

Выбран Эрли, а не Кока–Янгера–Касами: он работает с произвольной
грамматикой, без приведения к нормальной форме Хомского. Приведение
меняет грамматику, а сравнивать надо ту, что дана в условии.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, refuted, unknown
from tfl.words import iter_words

__all__ = [
    "recognize",
    "language",
    "derivation",
    "disagreements",
    "cyk",
    "prefix_free",
    "equivalent_up_to",
]


@dataclass(frozen=True)
class _State:
    """Ситуация Эрли: правило с точкой и позиция начала разбора."""

    production: Production
    dot: int
    origin: int

    @property
    def at_end(self) -> bool:
        return self.dot >= len(self.production.rhs)

    @property
    def next_symbol(self) -> str | None:
        return None if self.at_end else self.production.rhs[self.dot]

    def advance(self) -> _State:
        return _State(self.production, self.dot + 1, self.origin)


def recognize(grammar: CFG, word: str, tokens: Iterable[str] | None = None) -> bool:
    """Принадлежит ли слово языку грамматики (алгоритм Эрли).

    `tokens` позволяет задать разбиение на символы явно; по умолчанию
    слово делится посимвольно — так записаны все задания курса.
    """
    symbols = list(tokens) if tokens is not None else list(word)
    n = len(symbols)
    nullable = grammar.nullable()
    chart: list[set[_State]] = [set() for _ in range(n + 1)]
    for p in grammar.rules_for(grammar.start):
        chart[0].add(_State(p, 0, 0))

    for i in range(n + 1):
        queue = list(chart[i])
        seen = set(queue)
        while queue:
            state = queue.pop()
            symbol = state.next_symbol
            if symbol is None:
                # Завершение: продвигаем всех, кто ждал этот нетерминал
                for candidate in list(chart[state.origin]):
                    if candidate.next_symbol == state.production.lhs:
                        moved = candidate.advance()
                        if moved not in seen:
                            seen.add(moved)
                            chart[i].add(moved)
                            queue.append(moved)
            elif symbol in grammar.nonterminals:
                # Предсказание
                for p in grammar.rules_for(symbol):
                    fresh = _State(p, 0, i)
                    if fresh not in seen:
                        seen.add(fresh)
                        chart[i].add(fresh)
                        queue.append(fresh)
                # Приём Айкока–Хорспула для аннулируемых нетерминалов.
                # Без него ε-правила теряются: завершение `T → •` происходит
                # в том же столбце, и если ситуация, ожидающая `T`, добавлена
                # позже, продвинуть её уже некому. Поэтому продвигаем сразу.
                if symbol in nullable:
                    moved = state.advance()
                    if moved not in seen:
                        seen.add(moved)
                        chart[i].add(moved)
                        queue.append(moved)
            elif i < n and symbol == symbols[i]:
                # Сканирование — попадает в следующий столбец
                chart[i + 1].add(state.advance())

    return any(
        state.at_end
        and state.origin == 0
        and state.production.lhs == grammar.start
        for state in chart[n]
    )


def language(grammar: CFG, max_len: int) -> set[str]:
    """Все слова языка длины не больше `max_len`.

    Считается неподвижной точкой: для каждого нетерминала копится множество
    выводимых из него терминальных слов подходящей длины. Способ точный —
    в отличие от обхода сентенциальных форм, где легко потерять слова,
    выводимые через длинные промежуточные формы.
    """
    derivable: dict[str, set[str]] = {n: set() for n in grammar.nonterminals}

    def expand(symbols: tuple[str, ...]) -> set[str]:
        results = {""}
        for symbol in symbols:
            pieces = (
                derivable[symbol] if symbol in grammar.nonterminals else {symbol}
            )
            if not pieces:
                return set()
            results = {
                a + b for a in results for b in pieces if len(a) + len(b) <= max_len
            }
            if not results:
                return set()
        return results

    changed = True
    while changed:
        changed = False
        for p in grammar.productions:
            before = len(derivable[p.lhs])
            derivable[p.lhs] |= expand(p.rhs)
            if len(derivable[p.lhs]) != before:
                changed = True

    return derivable.get(grammar.start, set())


def derivation(grammar: CFG, word: str, max_steps: int = 200_000) -> list[str] | None:
    """Левосторонний вывод слова — цепочка сентенциальных форм.

    Нужен для отчёта: «слово принадлежит языку» убедительнее с предъявленным
    выводом. Поиск в ширину, поэтому вывод получается кратчайшим.
    """
    if not recognize(grammar, word):
        return None
    start = (grammar.start,)
    queue: list[tuple[tuple[str, ...], list[str]]] = [(start, ["".join(start)])]
    seen = {start}
    steps = 0
    while queue and steps < max_steps:
        form, history = queue.pop(0)
        steps += 1
        if not any(s in grammar.nonterminals for s in form):
            if "".join(form) == word:
                return history
            continue
        index = next(i for i, s in enumerate(form) if s in grammar.nonterminals)
        prefix_terminals = "".join(form[:index])
        if not word.startswith(prefix_terminals):
            continue
        for p in grammar.rules_for(form[index]):
            fresh = form[:index] + p.rhs + form[index + 1 :]
            terminal_count = sum(1 for s in fresh if s not in grammar.nonterminals)
            if terminal_count > len(word) or fresh in seen:
                continue
            seen.add(fresh)
            queue.append((fresh, history + ["".join(fresh)]))
    return None


def disagreements(
    grammar: CFG,
    predicate: Callable[[str], bool],
    alphabet: Iterable[str],
    max_len: int = 10,
    limit: int = 10,
) -> list[str]:
    """Слова, на которых грамматика расходится с эталонным предикатом.

    Основной способ проверить гипотезу «эта грамматика задаёт тот самый
    язык из условия»: предикат пишется прямо по словесному описанию,
    а грамматика — независимо, и они обязаны совпасть.
    """
    bad: list[str] = []
    for word in iter_words(alphabet, max_len):
        if recognize(grammar, word) != predicate(word):
            bad.append(word)
            if len(bad) >= limit:
                break
    return bad


def cyk(grammar: CFG, word: str, cnf: CFG | None = None) -> bool:
    """Принадлежность по алгоритму Кока–Янгера–Касами.

    Третий независимый способ ответить на тот же вопрос, что `recognize`
    (Эрли) и `language` (неподвижная точка). Именно ради независимости он
    и нужен: Эрли работает сверху вниз по исходной грамматике, CYK — снизу
    вверх по нормальной форме Хомского, и совпасть «по построению» они
    не могут.

    `cnf` позволяет передать уже приведённую грамматику, чтобы не
    пересчитывать её на каждом слове.
    """
    normal = cnf if cnf is not None else grammar.chomsky_normal_form()
    if not word:
        return any(p.lhs == normal.start and not p.rhs for p in normal.productions)

    by_body: dict[tuple[str, ...], set[str]] = {}
    for p in normal.productions:
        if p.rhs:
            by_body.setdefault(p.rhs, set()).add(p.lhs)

    n = len(word)
    # table[length][start] — нетерминалы, выводящие word[start : start + length]
    table = [[set() for _ in range(n + 1)] for _ in range(n + 1)]
    for i, char in enumerate(word):
        table[1][i] = set(by_body.get((char,), ()))
    for length in range(2, n + 1):
        for start in range(n - length + 1):
            cell = table[length][start]
            for split in range(1, length):
                for left in table[split][start]:
                    for right in table[length - split][start + split]:
                        cell |= by_body.get((left, right), set())
    return normal.start in table[n][0]


def prefix_free(grammar: CFG, max_len: int = 12) -> Verdict:
    """Беспрефиксен ли язык: нет ли в нём слова, являющегося началом другого.

    Свойство важно в ЛР3 потому, что по пустому стеку DPDA распознаёт
    **ровно** беспрефиксные детерминированные языки: если язык не
    беспрефиксен, детерминированный распознаватель обязан принимать
    по финальному состоянию.

    Исходов два, и «да» среди них нет: беспрефиксность КС-языка
    неразрешима (это пустота пересечения `L ∩ L·Σ⁺`), поэтому отсутствие
    контрпримера до длины `max_len` остаётся «не выяснено».
    """
    known = language(grammar, max_len)
    for word in sorted(known, key=lambda w: (len(w), w)):
        for cut in range(len(word)):
            if word[:cut] in known:
                return refuted(
                    f"«{word[:cut] or 'ε'}» — собственное начало «{word}», оба в языке",
                    (word[:cut], word),
                )
    return unknown(f"контрпример не найден среди слов длины ≤ {max_len}")


def equivalent_up_to(
    left: CFG, right: CFG, max_len: int = 8
) -> tuple[list[str], list[str]]:
    """Слова, различающие две грамматики: `только в левой`, `только в правой`.

    Это и есть «автоматическое тестирование предполагаемой эквивалентности»
    из задания ЛР3: грамматику сравнивают с ней же после пересечения
    с регулярной аппроксимацией. Аппроксимация строится **сверху**, поэтому
    непустой второй список означает ошибку построения, а непустой первый —
    что аппроксимация действительно сузила язык.
    """
    ours = language(left, max_len)
    theirs = language(right, max_len)
    order = lambda w: (len(w), w)  # noqa: E731
    return sorted(ours - theirs, key=order), sorted(theirs - ours, key=order)
