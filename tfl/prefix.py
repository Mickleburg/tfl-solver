"""Префиксные грамматики: общая модель и алфавитный частный случай.

Текущий курс определяет общую грамматику ``G=(W,R)``, где базис ``W`` —
множество слов, а правило ``u -> v`` применяется только к началу:
``uz => vz``. Ей соответствуют :class:`GeneralPrefixGrammar`, конструкция
по правому графу :func:`dfa_to_prefix_grammar` и точная проверка лексических
меток :func:`check_labelled_rule`.

Ниже сохранён и более узкий алфавитный вариант прежнего курса.

Определение — лекция 4 курса 2022 года
(`corpus/txt/FormalLanguageTheory_2022_lect_tfl_4.txt`, слайд 31), дословно:

> Дана SRS `S` с правилами переписывания двух видов: `aᵢ → b₁…bₙ` и `aᵢ → ε`.
> Разрешим применять правила только к первым буквам слова. Пусть дана пара
> `⟨S, w₀⟩`, где `w₀` — слово в алфавите `Σ`. Эта пара определяет алфавитную
> префиксную грамматику.
>
> **Утверждение.** Язык `L⟨S, w₀⟩` регулярен.

Левая часть правила — **одна буква**, и переписывается только первая буква
слова. Отсюда и регулярность: хвост слова, до которого переписывание уже
не доберётся, дальше не меняется никогда, а «фронт» переписывания движется
только вправо.

Зачем нужен частный случай. Префиксная грамматика — второе представление
языка на задаче 2 РК1 (+2 балла), и она же стоит в банке «Аптеки» задачами
на 2 и на 4 балла. Следствие теоремы Турчина из лекции 6 опирается ровно
на эту конструкцию.

Для алфавитного варианта механизированы: коллапсирование букв, перевод в линейную
грамматику по лекции и прямое переписывание для сверки. Перевод и прямое
переписывание написаны **независимо друг от друга**, и `agrees_with_rewriting`
их сверяет — это и есть проверка конструкции, а не веры в неё.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable, Iterable

from tfl.automata import DFA
from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "GeneralPrefixGrammar",
    "PrefixGrammar",
    "check_labelled_rule",
    "dfa_to_prefix_grammar",
    "parse_prefix_grammar",
]


@dataclass(frozen=True)
class GeneralPrefixGrammar:
    """Общая префиксная грамматика ``G = (W, R)``.

    В отличие от алфавитного частного случая ниже, обе стороны правила
    могут быть словами произвольной длины. Правило ``u -> v`` применяется
    только в начале: ``uz => vz``.
    """

    bases: frozenset[str]
    rules: tuple[tuple[str, str], ...]
    alphabet: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.alphabet:
            letters = {letter for word in self.bases for letter in word}
            for left, right in self.rules:
                letters.update(left)
                letters.update(right)
            object.__setattr__(self, "alphabet", frozenset(letters))

    def step(self, word: str) -> set[str]:
        """Все результаты одного переписывания префикса."""
        results = {
            right + word[len(left) :]
            for left, right in self.rules
            if word.startswith(left)
        }
        results.discard(word)
        return results

    def _reachable(
        self, max_len: int, max_nodes: int
    ) -> tuple[set[str], bool, bool]:
        seen = {word for word in self.bases if len(word) <= max_len}
        queue = deque(sorted(seen))
        cut_by_length = any(len(word) > max_len for word in self.bases)
        while queue:
            word = queue.popleft()
            for following in sorted(self.step(word)):
                if len(following) > max_len:
                    cut_by_length = True
                    continue
                if following in seen:
                    continue
                if len(seen) >= max_nodes:
                    return seen, cut_by_length, True
                seen.add(following)
                queue.append(following)
        return seen, cut_by_length, False

    def reachable(self, max_len: int = 12, max_nodes: int = 100_000) -> set[str]:
        """Слова ``L(G)`` до заданной длины.

        Это ограниченный обход. Для грамматики, построенной функцией
        :func:`dfa_to_prefix_grammar`, правила не уменьшают длину, поэтому
        сравнение всех слов до ``max_len`` не теряет коротких выводов.
        """
        return self._reachable(max_len, max_nodes)[0]

    def agrees_with_dfa(
        self, machine: DFA, max_len: int = 8, max_nodes: int = 100_000
    ) -> Verdict:
        """Сверить грамматику с ДКА на всех словах до ``max_len``."""
        from tfl.words import iter_words

        generated, cut_by_length, cut_by_nodes = self._reachable(max_len, max_nodes)
        contracting = any(len(right) < len(left) for left, right in self.rules)
        if cut_by_nodes or (cut_by_length and contracting):
            return unknown(
                "обход префиксной грамматики обрезан; совпадение на полном "
                "множестве коротких слов не проверено"
            )
        alphabet = "".join(sorted(machine.alphabet))
        for word in iter_words(alphabet, max_len):
            if (word in generated) != machine.accepts(word):
                return refuted(
                    f"грамматика и ДКА расходятся на слове «{word or 'ε'}»",
                    word,
                )
        return proved(
            f"грамматика и ДКА совпали на всех словах до длины {max_len}"
        )


def dfa_to_prefix_grammar(machine: DFA) -> GeneralPrefixGrammar:
    """Построить префиксную грамматику по правому графу ДКА.

    Для каждого достижимого состояния ``q`` выбирается кратчайший
    представитель ``r_q``. Финальные представители образуют базис. Ребро
    ``q -a-> p`` обращается в правило ``r_p -> r_q a``. Обе части приводят
    начальное состояние в ``p``, поэтому правило сохраняет принадлежность
    языка при любом общем суффиксе.
    """
    representatives: dict[Hashable, str] = {machine.start: ""}
    queue = deque([machine.start])
    while queue:
        state = queue.popleft()
        for letter in sorted(machine.alphabet):
            following = machine.delta.get((state, letter))
            if following is None or following in representatives:
                continue
            representatives[following] = representatives[state] + letter
            queue.append(following)

    bases = frozenset(
        representatives[state]
        for state in machine.finals
        if state in representatives
    )
    rules: list[tuple[str, str]] = []
    states = sorted(
        representatives,
        key=lambda state: (
            len(representatives[state]),
            representatives[state],
            repr(state),
        ),
    )
    for state in states:
        for letter in sorted(machine.alphabet):
            following = machine.delta.get((state, letter))
            if following not in representatives:
                continue
            rule = (
                representatives[following],
                representatives[state] + letter,
            )
            if rule[0] != rule[1] and rule not in rules:
                rules.append(rule)
    return GeneralPrefixGrammar(bases, tuple(rules), machine.alphabet)


def check_labelled_rule(
    machine: DFA,
    left: str,
    right: str,
    source_finals: Iterable[Hashable],
    target_finals: Iterable[Hashable],
) -> Verdict:
    """Проверить правило ``left -> right`` с меткой ``(T1, T2)``.

    Цвета ``source_finals`` и ``target_finals`` задают два лексических
    домена на одном правом графе. Проверяется точное условие
    ``left*z in T1 => right*z in T2`` для **всех** суффиксов ``z``.
    При нарушении возвращается кратчайший суффикс-свидетель.
    """
    first_finals = frozenset(source_finals)
    second_finals = frozenset(target_finals)
    start = (machine.run(left), machine.run(right))
    queue = deque([start])
    suffixes: dict[tuple[Hashable | None, Hashable | None], str] = {start: ""}

    def advance(state: Hashable | None, letter: str) -> Hashable | None:
        return None if state is None else machine.delta.get((state, letter))

    while queue:
        first, second = queue.popleft()
        suffix = suffixes[(first, second)]
        if first in first_finals and second not in second_finals:
            return refuted(
                f"метка неверна: для суффикса «{suffix or 'ε'}» слово "
                f"«{left + suffix or 'ε'}» лежит в T1, а "
                f"«{right + suffix or 'ε'}» не лежит в T2",
                suffix,
            )
        for letter in sorted(machine.alphabet):
            following = (advance(first, letter), advance(second, letter))
            if following not in suffixes:
                suffixes[following] = suffix + letter
                queue.append(following)
    return proved(
        "для всех суффиксов z выполнено: left·z ∈ T1 влечёт right·z ∈ T2"
    )


@dataclass(frozen=True)
class PrefixGrammar:
    """Пара `⟨S, w₀⟩`: правила вида `a → β` и стартовое слово.

    >>> apg = parse_prefix_grammar("a -> b c\\nb -> ε\\nc -> ε", start="ab")
    >>> sorted(apg.collapsing())
    ['a', 'b', 'c']
    >>> sorted(apg.reachable(max_len=4))
    ['', 'ab', 'b', 'bcb', 'cb']
    """

    rules: tuple[tuple[str, str], ...]
    start: str
    alphabet: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        if not self.alphabet:
            letters = set(self.start)
            for left, right in self.rules:
                letters.add(left)
                letters |= set(right)
            object.__setattr__(self, "alphabet", frozenset(letters))
        for left, _ in self.rules:
            if len(left) != 1:
                raise ValueError(
                    f"левая часть правила «{left}» не буква: в APG переписывается "
                    "ровно один символ"
                )

    def __str__(self) -> str:
        body = "\n".join(f"{left} → {right or 'ε'}" for left, right in self.rules)
        return f"w₀ = {self.start or 'ε'}\n{body}"

    def rules_for(self, letter: str) -> list[str]:
        return [right for left, right in self.rules if left == letter]

    # ------------------------------------------------------------ коллапс

    def collapsing(self) -> frozenset[str]:
        """Буквы, для которых `a ↠ ε`, — наименьшая неподвижная точка.

        > Скажем, что `a ↠ ε` (a коллапсирует), если либо `a → ε ∈ S`,
        > либо `∃b₁…bₙ(∀bᵢ(bᵢ ↠ ε) & a → b₁…bₙ ∈ S)`.
        """
        collapse: set[str] = set()
        changed = True
        while changed:
            changed = False
            for left, right in self.rules:
                if left in collapse:
                    continue
                if all(letter in collapse for letter in right):
                    collapse.add(left)
                    changed = True
        return frozenset(collapse)

    # ------------------------------------------- прямое переписывание

    def step(self, word: str) -> set[str]:
        """Один шаг: переписать **первую** букву слова."""
        if not word:
            return set()
        return {right + word[1:] for right in self.rules_for(word[0])}

    def reachable(self, max_len: int = 12, max_nodes: int = 100_000) -> set[str]:
        """Слова, достижимые из `w₀`, длиной не больше `max_len`.

        Это независимое от конструкции определение языка: по нему
        конструкция и проверяется.
        """
        seen = {self.start} if len(self.start) <= max_len else set()
        frontier = set(seen)
        while frontier and len(seen) < max_nodes:
            following = set()
            for word in frontier:
                for nxt in self.step(word):
                    if len(nxt) <= max_len and nxt not in seen:
                        seen.add(nxt)
                        following.add(nxt)
            frontier = following
        return seen

    # --------------------------------------------- перевод в грамматику

    def _nonterminal(self, letter: str) -> str:
        return f"N{letter}"

    def _expansion(self, body: str) -> list[tuple[str, ...]]:
        """Правые части для развёртки слова `body` по правилам лекции.

        Фронт переписывания доходит до первой буквы, которая не коллапсирует,
        и дальше не идёт. Если не коллапсирует буква `bᵢ`, добавляются
        правые части `B₁b₂…bₙ`, …, `Bᵢbᵢ₊₁…bₙ`; если коллапсируют все —
        то же самое до конца, плюс отдельная `Bₙ`.
        """
        collapse = self.collapsing()
        options: list[tuple[str, ...]] = []
        for i, letter in enumerate(body):
            options.append((self._nonterminal(letter), *body[i + 1 :]))
            if letter not in collapse:
                break
        return options

    def to_cfg(self) -> CFG:
        """Линейная грамматика того же языка — конструкция из лекции 4.

        Каждой букве `aᵢ` сопоставляется нетерминал; правило `a → b₁…bₙ`
        даёт правые части, в которых нетерминал стоит на позиции фронта
        переписывания, а всё, до чего фронт не дойдёт, остаётся терминалами.
        Правило `A → a` — это «ничего не переписывали».
        """
        collapse = self.collapsing()
        productions: list[Production] = []

        for letter in sorted(self.alphabet):
            name = self._nonterminal(letter)
            productions.append(Production(name, (letter,)))
            if letter in {left for left, _ in self.rules} and "" in self.rules_for(letter):
                productions.append(Production(name, ()))
            for right in self.rules_for(letter):
                for option in self._expansion(right):
                    productions.append(Production(name, option))

        start = "S"
        for option in self._expansion(self.start):
            productions.append(Production(start, option))
        productions.append(Production(start, tuple(self.start)))
        if all(letter in collapse for letter in self.start):
            productions.append(Production(start, ()))

        nonterminals = frozenset({start} | {self._nonterminal(a) for a in self.alphabet})
        return CFG(
            start=start,
            productions=tuple(dict.fromkeys(productions)),
            nonterminals=nonterminals,
            terminals=frozenset(self.alphabet),
        )

    def agrees_with_rewriting(self, max_len: int = 8) -> Verdict:
        """Сверить построенную грамматику с прямым переписыванием.

        Конструкция из лекции и перебор достижимых слов написаны независимо,
        поэтому их совпадение на всех словах до `max_len` — настоящий довод.
        Расхождение возвращается свидетелем: это либо ошибка в конструкции,
        либо неверно прочитанное правило.
        """
        from tfl.parse import recognize
        from tfl.words import iter_words

        grammar = self.to_cfg()
        direct = self.reachable(max_len=max_len)
        letters = "".join(sorted(self.alphabet))
        for word in iter_words(letters, max_len):
            if recognize(grammar, word) != (word in direct):
                return refuted(
                    f"грамматика и переписывание расходятся на слове «{word or 'ε'}»",
                    word,
                )
        return proved(
            f"грамматика и прямое переписывание совпали на всех словах "
            f"до длины {max_len}",
            grammar,
        )


def parse_prefix_grammar(text: str, start: str) -> PrefixGrammar:
    """Разобрать правила `a -> b c` по строке на правило.

    Пустая правая часть записывается как `ε`, `eps` или пустая строка.
    """
    from tfl.srs import ARROWS, EPSILON_TOKENS

    rules: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for arrow in ARROWS:
            if arrow in line:
                left, _, right = line.partition(arrow)
                break
        else:
            raise ValueError(f"в строке «{line}» нет стрелки")
        left = left.strip().replace(" ", "")
        right = right.strip()
        right = "" if right in EPSILON_TOKENS else right.replace(" ", "")
        rules.append((left, right))
    return PrefixGrammar(tuple(rules), start)
