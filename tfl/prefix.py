"""Алфавитные префиксные грамматики: переписывание только начала слова.

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

Зачем это нужно. Префиксная грамматика — второе независимое представление
языка на задаче 2 РК1 (+2 балла), и она же стоит в банке «Аптеки» задачами
на 2 и на 4 балла. Следствие теоремы Турчина из лекции 6 опирается ровно
на эту конструкцию.

Что здесь механизировано: коллапсирование букв, перевод в линейную
грамматику по лекции и прямое переписывание для сверки. Перевод и прямое
переписывание написаны **независимо друг от друга**, и `agrees_with_rewriting`
их сверяет — это и есть проверка конструкции, а не веры в неё.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, proved, refuted

__all__ = [
    "PrefixGrammar",
    "parse_prefix_grammar",
]


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
