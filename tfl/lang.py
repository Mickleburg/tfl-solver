"""Язык как исполняемый предикат: пересечения и таблицы различимости.

Обслуживает `RK1-*`, `RK2-B`, `EXAM-1` — всё, где спрашивают «регулярен ли
язык» или «контекстно-свободен ли», а язык задан словами.

Модуль построен вокруг одного приёма, который в этом курсе является
основным. Он повторяется и в ответах преподавателя, и в разобранных
экзаменационных задачах, и во всех просмотренных проверенных работах РК2
(см. `corpus/chat/FINDINGS.md`, §3):

> **Пересечь язык с подходящим регулярным**, получить язык попроще,
> и на нём доказать нерегулярность таблицей Майхилла–Нероуда или
> не-КС-свойство накачкой. Оба класса замкнуты относительно пересечения
> с регулярным, поэтому вывод переносится на исходный язык.

Что здесь механизируется: предикат, пересечение, построение таблицы,
подсчёт числа попарно различимых префиксов. Что **не** механизируется:
выбор регулярного языка для пересечения и выбор семейств префиксов
и суффиксов. Это содержательная догадка, и она же — самая интересная
часть задачи.

Про границы вывода. Различив `n` префиксов, мы **доказали**, что классов
не меньше `n` — это честное доказательство для конкретного `n`. Переход
от «для всех n ≤ N» к «для всех n» доказательством не является и делается
рассуждением о параметрическом семействе, которое пишет человек. Функции
здесь эту границу не размывают.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from tfl.automata import DFA, dfa_of
from tfl.verdict import Verdict, proved, refuted, unknown
from tfl.words import iter_words

__all__ = [
    "Language",
    "Table",
    "from_predicate",
    "from_regex",
    "from_dfa",
    "from_cfg",
    "table",
    "at_least_classes",
    "nonregular_evidence",
    "one_letter_noncf",
    "distinguishing_suffix",
    "family",
]

Predicate = Callable[[str], bool]


@dataclass
class Language:
    """Язык, заданный предикатом принадлежности.

    Предикат пишется прямо по формулировке условия и потому независим
    от любой построенной конструкции — автомата, грамматики, регулярки.
    На этой независимости держится вся проверка: совпадение независимо
    написанных предиката и конструкции на всех коротких словах —
    настоящий довод, а совпадение «по построению» — нет.
    """

    predicate: Predicate
    alphabet: str
    name: str = ""

    def __contains__(self, word: str) -> bool:
        return bool(self.predicate(word))

    def __str__(self) -> str:
        return self.name or "язык"

    def words(self, max_len: int, limit: int | None = None) -> list[str]:
        """Слова языка длины не больше `max_len`, в шортлекс-порядке."""
        found = []
        for word in iter_words(self.alphabet, max_len):
            if word in self:
                found.append(word)
                if limit is not None and len(found) >= limit:
                    break
        return found

    def __and__(self, other: Language) -> Language:
        return Language(
            lambda w: w in self and w in other,
            self.alphabet,
            f"{self} ∩ {other}",
        )

    def __or__(self, other: Language) -> Language:
        return Language(
            lambda w: w in self or w in other,
            self.alphabet,
            f"{self} ∪ {other}",
        )

    def __invert__(self) -> Language:
        return Language(lambda w: w not in self, self.alphabet, f"дополнение {self}")

    def restrict(self, pattern: str) -> Language:
        """Пересечь с регулярным языком, заданным академической регуляркой.

        Основной рабочий приём курса: пересечение сужает язык до простого
        параметрического семейства, на котором таблица классов строится
        руками. Регулярные и КС-языки замкнуты относительно пересечения
        с регулярным, поэтому нерегулярность (не-КС-свойство) пересечения
        доказывает то же и для исходного языка.
        """
        automaton = dfa_of(pattern, alphabet=self.alphabet)
        return Language(
            lambda w: w in self and automaton.accepts(w),
            self.alphabet,
            f"{self} ∩ {pattern}",
        )

    def disagreements(
        self, other: Language | DFA, max_len: int = 10, limit: int = 10
    ) -> list[str]:
        """Слова, на которых язык расходится с конструкцией или другим языком."""
        member = (
            (lambda w: other.accepts(w)) if isinstance(other, DFA) else (lambda w: w in other)
        )
        bad = []
        for word in iter_words(self.alphabet, max_len):
            if (word in self) != member(word):
                bad.append(word)
                if len(bad) >= limit:
                    break
        return bad


# --------------------------------------------------------------------------
# Способы задать язык
# --------------------------------------------------------------------------


def from_predicate(fn: Predicate, alphabet: str, name: str = "") -> Language:
    return Language(fn, alphabet, name)


def from_regex(pattern: str, alphabet: str, name: str = "") -> Language:
    automaton = dfa_of(pattern, alphabet=alphabet)
    return Language(automaton.accepts, alphabet, name or pattern)


def from_dfa(automaton: DFA, alphabet: str, name: str = "") -> Language:
    return Language(automaton.accepts, alphabet, name or "ДКА")


def from_cfg(grammar, name: str = "") -> Language:
    """Язык грамматики через алгоритм Эрли."""
    from tfl.parse import recognize

    return Language(
        lambda w: recognize(grammar, w),
        "".join(sorted(grammar.terminals)),
        name or f"L({grammar.start})",
    )


def family(template: Callable[[int], str], count: int, start: int = 0) -> list[str]:
    """Параметрическое семейство слов: `family(lambda i: "a" * i, 5)`.

    Именно в таком виде префиксы и суффиксы выписывают в решениях:
    `pᵢ = ab^i a`, `s_j = b^{j+1} aa`.
    """
    return [template(i) for i in range(start, start + count)]


# --------------------------------------------------------------------------
# Таблица различимости
# --------------------------------------------------------------------------


@dataclass
class Table:
    """Таблица Майхилла–Нероуда: строки — префиксы, столбцы — суффиксы.

    В ячейке — принадлежит ли `pᵢsⱼ` языку. Два префикса различимы, если
    нашёлся суффикс, на котором ответы разошлись.
    """

    language: Language
    prefixes: tuple[str, ...]
    suffixes: tuple[str, ...]
    cells: tuple[tuple[bool, ...], ...]

    def row(self, index: int) -> tuple[bool, ...]:
        return self.cells[index]

    def separated(self) -> list[tuple[int, int]]:
        """Пары индексов префиксов, которые какой-то суффикс различил."""
        return [
            (i, j)
            for i in range(len(self.prefixes))
            for j in range(i + 1, len(self.prefixes))
            if self.cells[i] != self.cells[j]
        ]

    def merged(self) -> list[tuple[int, int]]:
        """Пары префиксов, которые **не** различил ни один суффикс.

        Непустой список означает, что семейство выбрано неудачно:
        различить их могут другие суффиксы, но предъявленная таблица
        доказательством уже не является.
        """
        return [
            (i, j)
            for i in range(len(self.prefixes))
            for j in range(i + 1, len(self.prefixes))
            if self.cells[i] == self.cells[j]
        ]

    @property
    def is_diagonal(self) -> bool:
        """Единицы ровно на диагонали — самый частый вид таблицы в решениях."""
        n = min(len(self.prefixes), len(self.suffixes))
        return all(
            self.cells[i][j] == (i == j)
            for i in range(n)
            for j in range(n)
        )

    def markdown(self) -> str:
        head = "| | " + " | ".join(f"`{s or 'ε'}`" for s in self.suffixes) + " |"
        rule = "|---" * (len(self.suffixes) + 1) + "|"
        rows = [
            f"| `{p or 'ε'}` | "
            + " | ".join("+" if cell else "−" for cell in self.cells[i])
            + " |"
            for i, p in enumerate(self.prefixes)
        ]
        return "\n".join([head, rule, *rows])


def table(language: Language, prefixes: Sequence[str], suffixes: Sequence[str]) -> Table:
    """Построить таблицу различимости."""
    cells = tuple(
        tuple((prefix + suffix) in language for suffix in suffixes)
        for prefix in prefixes
    )
    return Table(language, tuple(prefixes), tuple(suffixes), cells)


def at_least_classes(
    language: Language, prefixes: Sequence[str], suffixes: Sequence[str]
) -> Verdict:
    """Доказать, что классов Майхилла–Нероуда не меньше, чем различимых префиксов.

    Это **доказательство** для конкретного числа: если все пары префиксов
    попарно различены предъявленными суффиксами, классов ровно столько же
    или больше. Вывод о бесконечности отсюда не следует — см.
    `nonregular_evidence`.
    """
    built = table(language, prefixes, suffixes)
    merged = built.merged()
    if merged:
        i, j = merged[0]
        return refuted(
            f"префиксы «{built.prefixes[i] or 'ε'}» и «{built.prefixes[j] or 'ε'}» "
            "не различены ни одним из предъявленных суффиксов",
            built,
        )
    return proved(f"классов не меньше {len(prefixes)}", built)


def nonregular_evidence(
    language: Language,
    prefix: Callable[[int], str],
    suffix: Callable[[int], str],
    upto: int = 8,
) -> Verdict:
    """Проверить параметрические семейства на роль различающих.

    Возвращает **два** исхода из трёх, и «доказано» среди них нет.

    * `False` — семейство не работает: нашлись `i ≠ j`, которых ни один
      суффикс из `s₀…s_{upto}` не различает. Это опровержение предложенного
      семейства (не регулярности языка), и оно точное.
    * `None` — все пары до `upto` различены. Это **не** доказательство
      нерегулярности: нужно показать, что различение сохраняется при
      произвольном `n`, а такое рассуждение пишет человек. Таблица
      прикладывается как свидетель — её и предъявляют в отчёте.

    Обратите внимание, что второй исход — обычный рабочий результат,
    а не неудача: оракул подтвердил, что семейство выбрано правильно,
    и осталось написать индукционный шаг.
    """
    prefixes = family(prefix, upto)
    suffixes = family(suffix, upto)
    verdict = at_least_classes(language, prefixes, suffixes)
    if verdict.value is False:
        return verdict
    built = verdict.witness
    return unknown(
        f"все {upto} префиксов попарно различены; для нерегулярности нужно "
        "то же рассуждение при произвольном n",
        built,
    )


def one_letter_noncf(
    language: Language,
    prefixes: Sequence[str],
    suffixes: Sequence[str],
) -> Verdict:
    """Оценка классов над однобуквенным алфавитом — она же оценка про КС.

    Следствие теоремы Париха (лекция 7 курса, стр. 4):

    > Множества регулярных и КС-языков над однобуквенным алфавитом
    > совпадают.

    Поэтому над одной буквой различающая таблица Майхилла–Нероуда бьёт
    не только по регулярности, но и по контекстной свободе — накачка
    для КС-языков не нужна.

    Так устроено доказательство из работы на 5 баллов (photo_176):
    язык пересекается с регулярным, стирающий гомоморфизм оставляет
    один символ, и дальше работает разрыв между соседними квадратами.
    КС-языки замкнуты относительно пересечения с регулярным и
    относительно гомоморфизма, поэтому вывод переносится на исходный язык.

    Границы те же, что у `at_least_classes`: различили `n` префиксов —
    доказано, что классов не меньше `n`. Бесконечность классов
    (а с ней и не-КС-свойство) — это обобщение при `n → ∞`,
    и пишет его человек.
    """
    letters = set(language.alphabet)
    if len(letters) != 1:
        return refuted(
            f"алфавит {sorted(letters)} не однобуквенный — следствие Париха "
            "здесь не применимо",
            sorted(letters),
        )
    verdict = at_least_classes(language, prefixes, suffixes)
    if verdict.value is not True:
        return verdict
    return proved(
        f"{verdict.reason}; над однобуквенным алфавитом регулярные и КС-языки "
        "совпадают, поэтому та же оценка запрещает и КС-свойство",
        verdict.witness,
    )


def distinguishing_suffix(
    language: Language, left: str, right: str, max_len: int = 8
) -> str | None:
    """Суффикс, различающий два слова, либо `None`, если не нашёлся до `max_len`.

    `None` не означает, что слова эквивалентны: различающий суффикс может
    быть длиннее границы.
    """
    for word in iter_words(language.alphabet, max_len):
        if ((left + word) in language) != ((right + word) in language):
            return word
    return None
