"""Конъюнктивные грамматики и автоматы Треллиса — лекция 11.

Определение курса (`corpus/txt/FormalLanguageTheory_2023_lect_tfl_11.txt`,
слайд 10), дословно:

> Конъюнктивная грамматика `G` — грамматика, правила которой имеют вид
> `Aᵢ → Φ₁ & ⋯ & Φₙ`, где `Aᵢ` — нетерминал; `Φⱼ` — строки в смешанном
> алфавите терминалов и нетерминалов.

Смысл конъюнкции: `A` выводит слово `w`, если **каждый** из конъюнктов
`Φⱼ` выводит **то же самое** слово `w`. Это не разные части слова,
а разные разборы одного и того же куска — отсюда и сила класса.

Зачем это нужно. По разбалловке РК2 2023
(`corpus/issues/rk-2023-issues21-25.md`) построение конъюнктивной
грамматики общего вида стоит **3 балла**, линейная конъюнктивная
грамматика либо автомат Треллиса — ещё **3 балла**. На экзамене 2024
конъюнктивную грамматику просят построить в трёх билетах.

Что здесь механизировано: распознавание (обобщённый CYK по неподвижной
точке), перечисление языка, проверка линейности, автомат Треллиса
и его перевод в грамматику по конструкции из лекции. Что **не**
механизировано: построение грамматики по языку — это содержательная
догадка, и оракул нужен, чтобы её проверить, а не выдумать.

Про границы вывода. Совпадение грамматики с предикатом на всех словах
до длины `N` — сильный довод, но не доказательство: конъюнктивные
грамматики не замкнуты ни по каким известным леммам о накачке,
и общего способа доказать равенство языков у нас нет.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tfl.cfg import ARROWS, tokenize_body
from tfl.verdict import Verdict, proved, refuted
from tfl.words import iter_words

__all__ = [
    "ConjunctiveRule",
    "ConjunctiveGrammar",
    "Trellis",
    "parse_conjunctive",
]


@dataclass(frozen=True)
class ConjunctiveRule:
    """Правило `A → Φ₁ & ⋯ & Φₙ`. Один конъюнкт — обычное КС-правило."""

    lhs: str
    conjuncts: tuple[tuple[str, ...], ...]

    def __str__(self) -> str:
        parts = [" ".join(c) if c else "ε" for c in self.conjuncts]
        return f"{self.lhs} → {' & '.join(parts)}"

    @property
    def is_plain(self) -> bool:
        """Правило без конъюнкции — то же самое, что правило КС-грамматики."""
        return len(self.conjuncts) == 1


@dataclass(frozen=True)
class ConjunctiveGrammar:
    """Конъюнктивная грамматика.

    Грамматика лекции для `{(aⁿb)ᵏ | n, k ⩾ 1}` — блоки обязаны быть равными:

    >>> g = parse_conjunctive("S -> S A & C b | A\\nA -> a A | a b\\nC -> a C a | B\\nB -> B A | b")
    >>> g.words(6)
    ['ab', 'aab', 'aaab', 'abab', 'aaaab', 'aaaaab', 'aabaab', 'ababab']
    >>> g.recognize("abaab")
    False
    """

    start: str
    rules: tuple[ConjunctiveRule, ...]
    nonterminals: frozenset[str] = field(default=frozenset())
    terminals: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        if not self.nonterminals:
            object.__setattr__(self, "nonterminals", frozenset(r.lhs for r in self.rules))
        if not self.terminals:
            symbols = {s for r in self.rules for c in r.conjuncts for s in c}
            object.__setattr__(self, "terminals", frozenset(symbols - self.nonterminals))

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def is_linear(self) -> bool:
        """Линейна ли грамматика: в каждом конъюнкте не больше одного нетерминала.

        Линейные конъюнктивные языки — это в точности языки автоматов
        Треллиса (лекция 11, слайд 14), и на РК2 за них платят отдельно.
        """
        return all(
            sum(1 for s in conjunct if s in self.nonterminals) <= 1
            for rule in self.rules
            for conjunct in rule.conjuncts
        )

    # ------------------------------------------------------------- разбор

    def table(self, word: str) -> dict[tuple[int, int], frozenset[str]]:
        """Какие нетерминалы выводят каждое подслово — обобщённый CYK.

        Считается наименьшая неподвижная точка: множества только растут,
        поэтому обход заканчивается. Нормальная форма не нужна —
        правая часть любой длины разбирается протяжкой по позициям.
        """
        n = len(word)
        found: dict[tuple[int, int], set[str]] = {
            (i, j): set() for i in range(n + 1) for j in range(i, n + 1)
        }

        def derives(conjunct: tuple[str, ...], left: int, right: int) -> bool:
            """Разрезается ли `word[left:right]` по символам конъюнкта."""
            reach = {left}
            for symbol in conjunct:
                step: set[int] = set()
                for position in reach:
                    if symbol in self.nonterminals:
                        for end in range(position, right + 1):
                            if symbol in found[(position, end)]:
                                step.add(end)
                    elif word[position:position + 1] == symbol:
                        step.add(position + 1)
                reach = step
                if not reach:
                    return False
            return right in reach

        # Отрезки обходятся по возрастанию длины, и неподвижная точка ищется
        # **внутри одного отрезка**: правило для `word[i:j]` смотрит только
        # на подотрезки и на сам этот отрезок (случай `S → D & T`, где оба
        # конъюнкта покрывают слово целиком). Больший отрезок на меньший
        # повлиять не может, поэтому глобальный пересчёт не нужен.
        for length in range(0, n + 1):
            for i in range(0, n - length + 1):
                j = i + length
                changed = True
                while changed:
                    changed = False
                    for rule in self.rules:
                        if rule.lhs in found[(i, j)]:
                            continue
                        if all(derives(c, i, j) for c in rule.conjuncts):
                            found[(i, j)].add(rule.lhs)
                            changed = True
        return {span: frozenset(names) for span, names in found.items()}

    def recognize(self, word: str) -> bool:
        return self.start in self.table(word)[(0, len(word))]

    def words(self, max_len: int, alphabet: str = "") -> list[str]:
        letters = alphabet or "".join(sorted(self.terminals))
        return [w for w in iter_words(letters, max_len) if self.recognize(w)]

    def agrees_with(self, predicate, max_len: int = 8, alphabet: str = "") -> Verdict:
        """Сверить грамматику с предикатом из условия на всех коротких словах.

        Предикат пишется по тексту задачи и не зависит от грамматики,
        поэтому расхождение — настоящая улика, а совпадение — сильный,
        но не окончательный довод.
        """
        letters = alphabet or "".join(sorted(self.terminals))
        for word in iter_words(letters, max_len):
            if self.recognize(word) != bool(predicate(word)):
                side = "грамматика лишнее" if self.recognize(word) else "грамматика не берёт"
                return refuted(
                    f"расхождение на слове «{word or 'ε'}»: {side}", word
                )
        return proved(
            f"грамматика и условие совпали на всех словах до длины {max_len}",
            letters,
        )


def parse_conjunctive(text: str, start: str | None = None) -> ConjunctiveGrammar:
    """Разобрать грамматику: альтернативы через `|`, конъюнкты через `&`.

    Нотация правой части — та же, что у `parse_cfg`: разбор посимвольный,
    заглавная буква со штрихами и цифрами это один нетерминал.

    >>> str(parse_conjunctive("S -> a S & S b | ε").rules[0])
    'S → a S & S b'
    """
    rules: list[ConjunctiveRule] = []
    heads: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for arrow in ARROWS:
            if arrow in line:
                left, right = line.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {raw!r}")
        head = left.strip()
        if not head:
            raise ValueError(f"пустая левая часть: {raw!r}")
        heads.append(head)
        for alternative in right.split("|"):
            conjuncts = tuple(
                tokenize_body(part) for part in alternative.split("&")
            )
            rules.append(ConjunctiveRule(head, conjuncts))

    known = set(heads) | {
        s for r in rules for c in r.conjuncts for s in c if s[0].isupper()
    }
    terminals = {s for r in rules for c in r.conjuncts for s in c} - known
    return ConjunctiveGrammar(
        start or heads[0], tuple(rules), frozenset(known), frozenset(terminals)
    )


# --------------------------------------------------------------------------
# Автоматы Треллиса
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trellis:
    """Real-time клеточный автомат: треугольник, растущий от букв к вершине.

    > Если переписывание стартует с «листьев», но структурой является сеть,
    > то получается т.н. «автомат Треллиса». Языки, распознаваемые такими
    > автоматами, — это в точности линейные конъюнктивные языки.

    Нижний ряд — состояния отдельных букв (`init`), каждая следующая ячейка
    считается из двух соседних снизу (`delta`). Слово принимается, если
    состояние на вершине допускающее. Пустое слово не принимается никогда:
    у него нет нижнего ряда.
    """

    alphabet: frozenset[str]
    init: dict[str, str]
    delta: dict[tuple[str, str], str]
    accepting: frozenset[str]

    def states(self) -> frozenset[str]:
        return frozenset(self.init.values()) | frozenset(self.delta.values()) | frozenset(
            q for pair in self.delta for q in pair
        )

    def row(self, word: str) -> list[str] | None:
        """Состояния всех ячеек верхнего достижимого ряда, или `None`."""
        if not word or any(letter not in self.init for letter in word):
            return None
        current = [self.init[letter] for letter in word]
        while len(current) > 1:
            following = []
            for left, right in zip(current, current[1:]):
                if (left, right) not in self.delta:
                    return None
                following.append(self.delta[(left, right)])
            current = following
        return current

    def accepts(self, word: str) -> bool:
        top = self.row(word)
        return top is not None and top[0] in self.accepting

    def to_conjunctive(self) -> ConjunctiveGrammar:
        """Перевод в линейную конъюнктивную грамматику — конструкция лекции.

        > `S → A_final`;  `A_init(a) → a, a ∈ Σ`;
        > `A_δ(q1,q2) → A_q1 c & b A_q2, ∀b, c ∈ Σ`

        Читается так: слово с вершиной `δ(q1,q2)` — это слово, у которого
        начало без последней буквы имеет вершину `q1`, а конец без первой
        буквы имеет вершину `q2`. Обе половины описывают **одно и то же**
        слово, поэтому и нужна конъюнкция.

        Число правил растёт как `|Σ|²` на каждую пару состояний — то самое
        «неудобство, связанное с линейным разрастанием количества правил»,
        о котором предупреждает лекция.
        """
        letters = sorted(self.alphabet)
        rules: list[ConjunctiveRule] = []
        name = {q: f"A{index}" for index, q in enumerate(sorted(self.states()))}

        for q in sorted(self.accepting):
            rules.append(ConjunctiveRule("S", ((name[q],),)))
        for letter in letters:
            if letter in self.init:
                rules.append(ConjunctiveRule(name[self.init[letter]], ((letter,),)))
        for (left, right), target in sorted(self.delta.items()):
            for before in letters:
                for after in letters:
                    rules.append(
                        ConjunctiveRule(
                            name[target],
                            ((name[left], after), (before, name[right])),
                        )
                    )

        nonterminals = frozenset({"S"} | set(name.values()))
        return ConjunctiveGrammar(
            "S", tuple(dict.fromkeys(rules)), nonterminals, frozenset(letters)
        )

    def agrees_with_grammar(self, max_len: int = 6) -> Verdict:
        """Сверить автомат с построенной по нему грамматикой.

        Автомат считается треугольником, грамматика — обобщённым CYK;
        общего кода у них нет, поэтому совпадение проверяет конструкцию.
        """
        grammar = self.to_conjunctive()
        letters = "".join(sorted(self.alphabet))
        for word in iter_words(letters, max_len):
            if not word:
                continue
            if grammar.recognize(word) != self.accepts(word):
                return refuted(
                    f"автомат и грамматика расходятся на слове «{word}»", word
                )
        return proved(
            f"автомат и построенная грамматика совпали на всех словах "
            f"до длины {max_len}",
            grammar,
        )
