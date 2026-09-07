"""Автоматы с памятью (MFA) и ref-слова — лекция 12.

Формализм для регулярок с **обратными ссылками**: ячейка памяти хранит
подслово, прочитанное с ленты, и его потом можно потребовать снова.
Это тот же предмет, что ЛР4 (`tfl/extre.py`), но с другой стороны:
там разбор выражения, здесь распознаватель.

Определение курса
(`corpus/txt/FormalLanguageTheory_2023_lect_tfl_12.txt`, слайд 7):

> `k−MFA A` имеет функцию перехода из `Q × Σ ∪ {ε} ∪ {1,…,k}`
> в подмножество `Q × ⟨o, c, ⋄⟩ᵏ`, где флаги управления памятью означают:
> `c` — «закрыть» ячейку памяти; `o` — «открыть»; `⋄` — не менять.
>
> Начальная конфигурация памяти: `⟨⟨ε,c⟩,…,⟨ε,c⟩⟩`.

**Главная тонкость, и лекция выделяет её отдельно:**

> Запись в ячейку слова, считанного с ленты по переходу, осуществляется
> не исходя из флагов памяти в предыдущем состоянии, а исходя из флагов
> памяти в состоянии, куда осуществляется переход. То есть мы сначала
> открываем (или закрываем) память, а уже потом читаем с ленты и пишем
> в открытые ячейки.

Отсюда три следствия, которые легко перепутать и которые здесь
закреплены тестами:

* открытие закрытой ячейки её **обнуляет**: `u' = v`, а не `u·v`;
* дописывание идёт только когда ячейка была открыта и осталась открытой;
* закрытая ячейка содержимое **сохраняет**, а не теряет.

Зачем это нужно. По разбалловке РК2 2023 построение MFA или ref-слова
общего вида стоит **3 балла**, столько же — обоснование, что
детерминированного (или ациклического) MFA не существует.

Про границы вывода. Симуляция ведётся с бюджетом: ε-переходы и чтения
пустой памяти дают циклы по конфигурациям, поэтому исчерпанный бюджет —
это «не выяснено», а не «слово не принимается».
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tfl.verdict import Verdict, proved, refuted, unknown
from tfl.words import iter_words

__all__ = [
    "OPEN",
    "CLOSE",
    "KEEP",
    "Cell",
    "MFA",
    "parse_mfa",
    "RefWord",
    "parse_ref_word",
]

OPEN = "o"
CLOSE = "c"
KEEP = "⋄"

#: Обозначения «не менять», встречающиеся в записи условий.
KEEP_TOKENS = {KEEP, ".", "-", "_", "d"}


@dataclass(frozen=True)
class Cell:
    """Ячейка памяти: содержимое и открыта ли она на запись."""

    content: str = ""
    status: str = CLOSE

    def __str__(self) -> str:
        return f"⟨{self.content or 'ε'}, {self.status}⟩"


@dataclass(frozen=True)
class MFA:
    """`k`-MFA: конечный автомат с `k` ячейками памяти.

    `transitions` — отображение `(состояние, что читаем)` в множество
    пар `(новое состояние, флаги)`. «Что читаем» — буква алфавита,
    `ε`, либо номер ячейки `1…k` (как целое число).
    """

    states: frozenset[str]
    alphabet: frozenset[str]
    cells: int
    start: str
    accepting: frozenset[str]
    transitions: dict[tuple[str, str | int], frozenset[tuple[str, tuple[str, ...]]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for (state, read), targets in self.transitions.items():
            if isinstance(read, int) and not 1 <= read <= self.cells:
                raise ValueError(f"ячейка {read} вне диапазона 1…{self.cells}")
            for target, flags in targets:
                if len(flags) != self.cells:
                    raise ValueError(
                        f"переход {state} → {target}: флагов {len(flags)}, "
                        f"а ячеек {self.cells}"
                    )

    def __str__(self) -> str:
        lines = []
        for (state, read), targets in sorted(self.transitions.items(), key=str):
            label = f"память {read}" if isinstance(read, int) else (read or "ε")
            for target, flags in sorted(targets):
                lines.append(f"{state} --{label}, {' '.join(flags)}--> {target}")
        return "\n".join(lines)

    # ------------------------------------------------------------- шаг

    def step(
        self, state: str, position: int, memory: tuple[Cell, ...], word: str
    ) -> list[tuple[str, int, tuple[Cell, ...]]]:
        """Все конфигурации, достижимые из данной за один переход.

        Порядок действий ровно тот, что в лекции: сначала применяются
        флаги, потом читается лента, потом идёт запись в открытые ячейки.
        """
        results: list[tuple[str, int, tuple[Cell, ...]]] = []
        for read in {word[position:position + 1], "", *range(1, self.cells + 1)}:
            for target, flags in self.transitions.get((state, read), ()):
                statuses = tuple(
                    cell.status if flag in KEEP_TOKENS else flag
                    for cell, flag in zip(memory, flags)
                )
                if isinstance(read, int):
                    if statuses[read - 1] != CLOSE:
                        continue  # читать можно только из закрытой ячейки
                    value = memory[read - 1].content
                else:
                    value = read
                if word[position:position + len(value)] != value:
                    continue
                updated = []
                for cell, status in zip(memory, statuses):
                    if status == CLOSE:
                        updated.append(Cell(cell.content, CLOSE))
                    elif cell.status == OPEN:
                        updated.append(Cell(cell.content + value, OPEN))
                    else:
                        updated.append(Cell(value, OPEN))
                results.append((target, position + len(value), tuple(updated)))
        return results

    # -------------------------------------------------------- симуляция

    def run(self, word: str, budget: int = 20_000) -> Verdict:
        """Принимается ли слово. Свидетель — трасса принимающего прогона.

        Обход в ширину по конфигурациям `(состояние, позиция, память)`.
        Циклы по ε-переходам и чтениям пустой памяти обход не зацикливают:
        повторные конфигурации отсекаются, а само пространство конечно —
        в ячейку попадает только то, что прочитано с ленты, поэтому длина
        содержимого ограничена длиной слова.

        Бюджет здесь не от зацикливания, а от размера: пространство
        конфигураций растёт как `|Q| · |w| · (число содержимых ячеек)ᵏ`,
        и на недетерминированном автомате с несколькими ячейками это
        много. Исчерпанный бюджет означает «не выяснено», а не «отказ».
        """
        start = (self.start, 0, tuple(Cell() for _ in range(self.cells)))
        seen = {start}
        frontier = [(start, [start])]
        visited = 0
        while frontier:
            following = []
            for (state, position, memory), path in frontier:
                if position == len(word) and state in self.accepting:
                    return proved(f"слово «{word or 'ε'}» принимается", path)
                for nxt in self.step(state, position, memory, word):
                    visited += 1
                    if visited > budget:
                        return unknown(
                            f"обход превысил бюджет {budget} конфигураций; "
                            f"про слово «{word or 'ε'}» ничего не известно"
                        )
                    if nxt not in seen:
                        seen.add(nxt)
                        following.append((nxt, [*path, nxt]))
            frontier = following
        return refuted(f"слово «{word or 'ε'}» не принимается", None)

    def accepts(self, word: str, budget: int = 20_000) -> bool:
        """Удобная обёртка: «не выяснено» приравнивается к отказу.

        Пользоваться ею можно только там, где бюджета заведомо хватает;
        когда это важно, берите `run` и смотрите на вердикт.
        """
        return self.run(word, budget).value is True

    def words(self, max_len: int, budget: int = 20_000) -> list[str]:
        letters = "".join(sorted(self.alphabet))
        return [w for w in iter_words(letters, max_len) if self.accepts(w, budget)]

    def memory_after(self, word: str, budget: int = 20_000) -> tuple[Cell, ...] | None:
        """Память в конце принимающего прогона — для сверки с лекцией."""
        verdict = self.run(word, budget)
        if verdict.value is not True:
            return None
        return verdict.witness[-1][2]

    # ---------------------------------------------------- детерминизм

    def is_deterministic(self) -> bool:
        """Условие лекции: `∀q ∈ Q, b ∈ Σ (|⋃ᵢ δ(q,i)| + |δ(q,b)| ⩽ 1)`.

        Читается так: из одного состояния нельзя одновременно иметь
        два разных перехода по одной букве, две разные ссылки на память,
        и даже одну ссылку вместе с одним чтением буквы. Язык, для
        которого такой автомат существует, называется DMFL.

        Заметьте: условие сформулировано через `Σ` и ссылки, ε-переходы
        в него не входят. Здесь оно проверяется буквально.
        """
        for state in self.states:
            from_memory = sum(
                len(self.transitions.get((state, i), ()))
                for i in range(1, self.cells + 1)
            )
            for letter in self.alphabet:
                if from_memory + len(self.transitions.get((state, letter), ())) > 1:
                    return False
        return True

    def agrees_with(self, predicate, max_len: int = 8, budget: int = 20_000) -> Verdict:
        """Сверить автомат с предикатом, написанным по условию задачи."""
        letters = "".join(sorted(self.alphabet))
        for word in iter_words(letters, max_len):
            verdict = self.run(word, budget)
            if verdict.value is None:
                return verdict
            if (verdict.value is True) != bool(predicate(word)):
                side = "автомат лишнее" if verdict.value else "автомат не берёт"
                return refuted(f"расхождение на слове «{word or 'ε'}»: {side}", word)
        return proved(
            f"автомат и условие совпали на всех словах до длины {max_len}", letters
        )


def parse_mfa(
    text: str,
    start: str,
    accepting: str,
    cells: int,
    alphabet: str,
) -> MFA:
    """Разобрать переходы: по одному на строку, `состояние читаем цель флаги`.

    Пример — автомат лекции для `{aⁿ²}`::

        q0 2 q1 o ⋄
        q1 1 q2 c o
        q2 a q3 ⋄ ⋄
        q3 2 q1 o c

    Цифра означает **номер ячейки**, если такой буквы нет в алфавите;
    иначе она читается как буква. `ε` (или `eps`) — пустое чтение.
    Флаг «не менять» пишется как `⋄` или точка.
    """
    letters = frozenset(alphabet)
    transitions: dict[tuple[str, str | int], set[tuple[str, tuple[str, ...]]]] = {}
    states = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 + cells:
            raise ValueError(
                f"в строке «{line}» {len(parts)} полей, а нужно {3 + cells}: "
                "состояние, чтение, цель и по флагу на ячейку"
            )
        state, read, target = parts[0], parts[1], parts[2]
        flags = tuple(KEEP if f in KEEP_TOKENS else f for f in parts[3:])
        states |= {state, target}
        key: str | int
        if read in {"ε", "eps", "_"}:
            key = ""
        elif read.isdigit() and read not in letters:
            key = int(read)
        else:
            key = read
        transitions.setdefault((state, key), set()).add((target, flags))
    return MFA(
        frozenset(states | {start} | set(accepting.split())),
        letters,
        cells,
        start,
        frozenset(accepting.split()),
        {k: frozenset(v) for k, v in transitions.items()},
    )


# --------------------------------------------------------------------------
# Ref-слова (Schmid, 2014)
# --------------------------------------------------------------------------

TOKEN = re.compile(r"\[(\d)|\](\d)|x(\d)|(.)")


@dataclass(frozen=True)
class RefWord:
    """Ref-слово: запись со скобками памяти `[ᵢ … ]ᵢ` и переменными `xᵢ`.

    > Значение в скобках `[k…]k` сохраняется в ячейку памяти с номером `k`
    > и затем может быть прочитано из входной строки при чтении
    > в backref-REGEX переменной `xk`.

    Разные скобочные блоки разрешается путать между собой — именно этим
    ref-слова отличаются от обычной скобочной записи:

    >>> ref = parse_ref_word("[1a[2b]1x1]2x2")
    >>> ref.expand()
    'ababbab'
    >>> ref.values()
    {1: 'ab', 2: 'bab'}
    """

    tokens: tuple[tuple[str, str | int], ...]

    def __str__(self) -> str:
        out = []
        for kind, value in self.tokens:
            out.append(
                f"[{value}" if kind == "open"
                else f"]{value}" if kind == "close"
                else f"x{value}" if kind == "read"
                else str(value)
            )
        return "".join(out)

    def _evaluate(self) -> tuple[str, dict[int, str]]:
        produced: list[str] = []
        content: dict[int, list[str]] = {}
        openness: set[int] = set()
        closed: dict[int, str] = {}

        def emit(piece: str) -> None:
            produced.append(piece)
            for index in openness:
                content[index].append(piece)

        for kind, value in self.tokens:
            if kind == "open":
                openness.add(int(value))
                content[int(value)] = []
            elif kind == "close":
                index = int(value)
                if index not in openness:
                    raise ValueError(f"ячейка {index} закрывается, не будучи открытой")
                openness.discard(index)
                closed[index] = "".join(content[index])
            elif kind == "read":
                index = int(value)
                if index not in closed:
                    raise ValueError(
                        f"переменная x{index} читается до закрытия ячейки {index}"
                    )
                emit(closed[index])
            else:
                emit(str(value))
        return "".join(produced), closed

    def expand(self) -> str:
        """Слово, которое задаёт ref-слово."""
        return self._evaluate()[0]

    def values(self) -> dict[int, str]:
        """Значения ячеек памяти после разбора."""
        return self._evaluate()[1]


def parse_ref_word(text: str) -> RefWord:
    """Разобрать ref-слово из записи вида `[1a[2b]1x1]2x2`."""
    tokens: list[tuple[str, str | int]] = []
    for match in TOKEN.finditer(text):
        opened, closed, read, letter = match.groups()
        if opened is not None:
            tokens.append(("open", int(opened)))
        elif closed is not None:
            tokens.append(("close", int(closed)))
        elif read is not None:
            tokens.append(("read", int(read)))
        elif not letter.isspace():
            tokens.append(("letter", letter))
    return RefWord(tuple(tokens))
