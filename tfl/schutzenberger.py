"""Скобочное представление по Хомскому–Шютценберже — лекция 8 (2021).

> **Теорема.** Любой CF-язык получается гомоморфизмом из языка
> `L′ = PARENₙ ∩ R`, где `R` — регулярный.

`PARENₙ` — язык Дика над `4n` скобками `[₁,]₁,…,[ₙ,]ₙ, (₁,)₁,…,(ₙ,)ₙ`.
Построение по грамматике в нормальной форме Хомского, правила
пронумерованы:

> 1. Если правило `n` имеет вид `A → BC`, тогда порождаем правило
>    `A → [ₙB]ₙ(ₙC)ₙ`.
> 2. Если правило `n` имеет вид `A → a`, тогда порождаем правило
>    `A → [ₙ]ₙ(ₙ)ₙ`.

Регулярное условие:

> `R = {x ∈ {[ⱼ,]ⱼ,(ⱼ,)ⱼ}* | x начинается с [ₙ для некоторого правила
> `n : A → ⋯` & все `]ₙ` предшествуют `(ₙ`}`

Зачем это на РК1: разбалловка задачи 2 (issues #6 и #21) платит
за **число независимых представлений** языка, и скобочное — одно из них,
стоит +2 балла.

**Прочтение гомоморфизма.** Лекция пишет: «Если `n` — нефинальное
правило, то `h([ₙ) = h(]ₙ) = h((ₙ) = h()ₙ) = ε. Иначе `h([ₙ) = a,
для остальных скобок так же». Буквальное «так же» (все четыре скобки
в `a`) даёт `h([ₙ]ₙ(ₙ)ₙ) = aaaa` вместо `a` и теорему ломает. Значит,
читать надо иначе: в букву переходит **одна** скобка, остальные — в ε.
Проверено исполнением: при этом прочтении `h(L(G′)) = L(G)`,
при буквальном — нет.
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, proved, refuted

__all__ = ["Bracketed", "bracketed", "dyck_words", "is_dyck"]

Word = tuple[str, ...]


def _names(number: int) -> tuple[str, str, str, str]:
    """Четыре скобки правила: `[ₙ`, `]ₙ`, `(ₙ`, `)ₙ`."""
    index = str(number).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return f"[{index}", f"]{index}", f"({index}", f"){index}"


@dataclass(frozen=True)
class Bracketed:
    """Скобочное представление: `G′`, гомоморфизм и регулярное условие."""

    grammar: CFG
    homomorphism: dict[str, str]
    source: CFG
    numbering: tuple[Production, ...]

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Пары скобок в порядке нумерации правил: `([ₙ, ]ₙ)` и `((ₙ, )ₙ)`."""
        found = []
        for number in range(1, len(self.numbering) + 1):
            opening, closing, round_open, round_close = _names(number)
            found.append((opening, closing))
            found.append((round_open, round_close))
        return tuple(found)

    def markdown(self) -> str:
        """Представление в том виде, в каком его сдают."""
        lines = ["Нумерация правил исходной грамматики:", ""]
        for number, production in enumerate(self.numbering, start=1):
            lines.append(f"{number}. `{production}`")
        lines += ["", "Скобочная грамматика `G′`:", ""]
        for production in self.grammar.productions:
            lines.append(f"- `{production}`")
        images = [
            f"`h({symbol}) = {image}`"
            for symbol, image in sorted(self.homomorphism.items())
            if image
        ]
        lines += ["", "Гомоморфизм (остальные скобки — в ε):", "", "- " + ", ".join(images)]
        return "\n".join(lines)

    # ------------------------------------------------------------- язык G′

    def words(self, max_symbols: int = 16) -> set[Word]:
        """Слова `L(G′)` длиной не больше данного числа **скобок**.

        Длина считается в скобках, а не в символах: скобка `[₁₂` пишется
        тремя знаками, и мерить её посимвольно было бы бессмысленно.
        """
        derivable: dict[str, set[Word]] = {n: set() for n in self.grammar.nonterminals}

        def expand(symbols: tuple[str, ...]) -> set[Word]:
            results: set[Word] = {()}
            for symbol in symbols:
                pieces = (
                    derivable[symbol]
                    if symbol in self.grammar.nonterminals
                    else {(symbol,)}
                )
                if not pieces:
                    return set()
                results = {
                    a + b
                    for a in results
                    for b in pieces
                    if len(a) + len(b) <= max_symbols
                }
                if not results:
                    return set()
            return results

        changed = True
        while changed:
            changed = False
            for production in self.grammar.productions:
                before = len(derivable[production.lhs])
                derivable[production.lhs] |= expand(production.rhs)
                changed = changed or len(derivable[production.lhs]) != before
        return derivable.get(self.grammar.start, set())

    def image(self, word: Word) -> str:
        """Гомоморфный образ скобочного слова."""
        return "".join(self.homomorphism.get(symbol, "") for symbol in word)

    # ------------------------------------------------- регулярное условие

    #: Уровни строгости `R`. `"слайд-9"` — дословно с девятого слайда;
    #: `"свойства"` — плюс два свойства `L(G′)` со слайдов 7-8;
    #: `"строгое"` — плюс согласование по нетерминалам.
    LEVELS = ("слайд-9", "свойства", "строгое")

    def allowed_next(self, level: str = "строгое") -> dict[str, frozenset[str] | None]:
        """Какая скобка может стоять сразу за какой. `None` — годится любая.

        Условие локальное, поэтому язык остаётся регулярным при любом
        уровне: смотрится окно в две скобки.
        """
        count = len(self.numbering)
        square_open = {_names(n)[0] for n in range(1, count + 1)}
        round_open = {_names(n)[2] for n in range(1, count + 1)}
        closing = {_names(n)[1] for n in range(1, count + 1)} | {
            _names(n)[3] for n in range(1, count + 1)
        }
        rules_of: dict[str, set[str]] = {}
        for number, production in enumerate(self.numbering, start=1):
            rules_of.setdefault(production.lhs, set()).add(_names(number)[0])

        allowed: dict[str, frozenset[str] | None] = {}
        for number, production in enumerate(self.numbering, start=1):
            opening, closer, opening_round, closer_round = _names(number)
            if level == "слайд-9":
                allowed.update(dict.fromkeys(_names(number), None))
                continue
            if level == "свойства":
                # «Ни одна `)ₙ` не предшествует непосредственно левой скобке»
                allowed[closer_round] = frozenset(closing)
                # «`[ₙ` непосредственно предшествует некоторой `[ₚ`,
                # так же как и `(ₙ`» — но только для правил `A → BC`
                branching = len(production.rhs) == 2
                allowed[opening] = frozenset(square_open) if branching else None
                allowed[opening_round] = frozenset(square_open) if branching else None
                allowed[closer] = None
                continue
            # Строгое: те же условия, но с учётом того, **какой** нетерминал
            # раскрывается. В лекции сказано «некоторой `[ₚ`», и этого мало:
            # без согласования по нетерминалам в `R ∩ PARENₙ` попадают слова,
            # склеенные из кусков разных правил.
            allowed[closer_round] = frozenset(closing)
            allowed[closer] = frozenset({opening_round})
            if len(production.rhs) == 2:
                left, right = production.rhs
                allowed[opening] = frozenset(rules_of.get(left, set()))
                allowed[opening_round] = frozenset(rules_of.get(right, set()))
            else:
                allowed[opening] = frozenset({closer})
                allowed[opening_round] = frozenset({closer_round})
        return allowed

    def starters(self) -> frozenset[str]:
        """Скобки `[ₙ`, с которых слово может начинаться: правила старта."""
        return frozenset(
            _names(number)[0]
            for number, production in enumerate(self.numbering, start=1)
            if production.lhs == self.source.start
        )

    def in_regular(self, word: Word, level: str = "строгое") -> bool:
        """Условие `R` из формулировки теоремы.

        Со слайда 9 лекции их два: слово начинается с `[ₙ` для правила
        **стартового** нетерминала, и для каждого `n` все `]ₙ` предшествуют
        всем `(ₙ`. Этого не хватает (см. `check_theorem`), поэтому уровней
        три — от дословного до достаточного.
        """
        if level not in self.LEVELS:
            raise ValueError(f"уровень должен быть одним из {self.LEVELS}")
        if not word or word[0] not in self.starters():
            return False
        for number in range(1, len(self.numbering) + 1):
            _, closer, opening_round, _ = _names(number)
            last_close = max((i for i, s in enumerate(word) if s == closer), default=-1)
            first_open = min(
                (i for i, s in enumerate(word) if s == opening_round), default=len(word)
            )
            if last_close > first_open:
                return False
        allowed = self.allowed_next(level)
        for position, symbol in enumerate(word[:-1]):
            permitted = allowed.get(symbol)
            if permitted is not None and word[position + 1] not in permitted:
                return False
        # Каждое правило `G′` кончается на `)ₙ`, поэтому и слово тоже.
        # Условия на конец слова в лекции нет вовсе, а без него в `R` попадают
        # обрубки вроде `[₁]₁`: у них просто нет следующей скобки, и локальные
        # запреты к ним не применяются.
        if level == "строгое" and word[-1] not in self.closing_round():
            return False
        return True

    def closing_round(self) -> frozenset[str]:
        """Скобки `)ₙ` — единственное, чем может кончаться слово `L(G′)`."""
        return frozenset(_names(n)[3] for n in range(1, len(self.numbering) + 1))

    def paren_and_regular(
        self, max_symbols: int, level: str = "строгое", max_nodes: int = 2_000_000
    ) -> set[Word] | None:
        """Слова `PARENₙ ∩ R` длиной не больше данного числа скобок.

        Порождаются обходом с отсечением, а не фильтрацией всех слов Дика:
        уже на трёх правилах слов Дика длины 12 больше шести миллионов.
        Все требования `R` локальны либо монотонны по префиксу, поэтому
        отсекать можно на каждом шаге.

        `None` означает, что обход не уложился в бюджет: на слабых уровнях
        отсечение почти не работает, и перебор действительно велик.
        """
        count = len(self.numbering)
        alphabet = [name for number in range(1, count + 1) for name in _names(number)]
        closer_of = dict(self.pairs)
        closers = set(closer_of.values())
        square_close = {_names(n)[1]: n for n in range(1, count + 1)}
        round_open = {_names(n)[2]: n for n in range(1, count + 1)}
        starters = self.starters()
        allowed = self.allowed_next(level)

        closing_round = self.closing_round()
        found: set[Word] = set()
        budget = [max_nodes]

        def walk(word: list[str], stack: list[str], opened: frozenset[int]) -> bool:
            if word and not stack:
                if level != "строгое" or word[-1] in closing_round:
                    found.add(tuple(word))
            if len(word) >= max_symbols:
                return True
            previous = word[-1] if word else None
            for symbol in alphabet:
                if previous is None:
                    if symbol not in starters:
                        continue
                else:
                    permitted = allowed.get(previous)
                    if permitted is not None and symbol not in permitted:
                        continue
                if symbol in square_close and square_close[symbol] in opened:
                    continue  # все `]ₙ` обязаны идти раньше всех `(ₙ`
                budget[0] -= 1
                if budget[0] <= 0:
                    return False
                if symbol in closers:
                    if not stack or stack[-1] != symbol:
                        continue
                    if not walk([*word, symbol], stack[:-1], opened):
                        return False
                else:
                    grown = (
                        opened | {round_open[symbol]} if symbol in round_open else opened
                    )
                    if not walk([*word, symbol], [*stack, closer_of[symbol]], grown):
                        return False
            return True

        return found if walk([], [], frozenset()) else None


    # ------------------------------------------------------------- проверки

    def agrees_with_source(self, max_symbols: int = 16) -> Verdict:
        """Совпадает ли `h(L(G′))` с `L(G)` на коротких словах.

        Сравнение честное в обе стороны: длина скобочного слова ровно
        вчетверо больше числа применённых правил, а слово исходного языка
        длины `k` выводится в НФ Хомского за `2k − 1` правил. Поэтому
        сверять можно все слова длины до `(max_symbols/4 + 1) / 2`.
        """
        from tfl.parse import language

        bound = (max_symbols // 4 + 1) // 2
        mine = {self.image(word) for word in self.words(max_symbols)}
        theirs = language(self.source, bound)
        mine = {word for word in mine if len(word) <= bound}
        if mine == theirs:
            return proved(
                f"h(L(G′)) совпадает с L(G) на всех {len(theirs)} словах "
                f"длины ≤ {bound}"
            )
        missing = sorted(theirs - mine)
        extra = sorted(mine - theirs)
        witness = (missing or extra)[0]
        where = "теряет" if missing else "добавляет"
        return refuted(
            f"расхождение: скобочное представление {where} слово "
            f"«{witness or 'ε'}»",
            witness,
        )

    def check_theorem(
        self, max_symbols: int = 12, level: str = "строгое", max_nodes: int = 2_000_000
    ) -> Verdict:
        """Проверить `L(G′) = PARENₙ ∩ R` на всех скобочных словах до длины.

        Лекция пишет «Можно убедиться, что `L′ = R ∩ PARENₙ`», не разбирая.
        Проверка перебором показывает, что **в напечатанной формулировке
        это неверно**, и уровней в `R` поэтому три.

        Уже на грамматике из одного правила `S → a` слово `[₁]₁` —
        правильная скобочная последовательность, начинается с `[₁`
        (правило стартового нетерминала), а требование «все `]₁` раньше
        `(₁`» выполнено впустую. В `L(G′) = {[₁]₁(₁)₁}` его нет.

        Свойства со слайдов 7-8, в определение `R` не вошедшие, дыру
        не закрывают: они ничего не говорят про то, что за `]ₙ` обязана
        идти `(ₙ`, и про согласование по нетерминалам. Уровень
        `"строгое"` добавляет и то и другое — оба условия локальные,
        так что `R` остаётся регулярным, — и на нём равенство держится.
        """
        mine = {word for word in self.words(max_symbols) if len(word) <= max_symbols}
        theirs = self.paren_and_regular(max_symbols, level=level, max_nodes=max_nodes)
        if theirs is None:
            return Verdict(
                None,
                f"перебор `PARENₙ ∩ R` на уровне «{level}» не уложился "
                f"в {max_nodes} узлов: на слабых условиях отсечение почти "
                f"не работает. Возьмите меньше скобок либо строгий уровень",
            )
        if mine == theirs:
            return proved(
                f"L(G′) и PARENₙ ∩ R совпадают на всех словах из ≤ "
                f"{max_symbols} скобок ({len(mine)} слов), уровень «{level}»"
            )
        missing = sorted(theirs - mine)
        extra = sorted(mine - theirs)
        witness = (missing or extra)[0]
        where = "есть в PARENₙ ∩ R, но не в L(G′)" if missing else "есть только в L(G′)"
        return refuted(f"слово «{' '.join(witness)}» {where}", witness)


def bracketed(grammar: CFG) -> Bracketed:
    """Построить скобочное представление по конструкции теоремы.

    Грамматика приводится к нормальной форме Хомского: конструкция
    определена только для правил вида `A → BC` и `A → a`.
    """
    normal = grammar.chomsky_normal_form()
    numbering = tuple(sorted(normal.productions))
    productions: list[Production] = []
    homomorphism: dict[str, str] = {}
    terminals: set[str] = set()

    for number, production in enumerate(numbering, start=1):
        opening, closing, round_open, round_close = _names(number)
        terminals |= {opening, closing, round_open, round_close}
        if len(production.rhs) == 2:
            left, right = production.rhs
            body = (opening, left, closing, round_open, right, round_close)
            for symbol in (opening, closing, round_open, round_close):
                homomorphism[symbol] = ""
        elif len(production.rhs) == 1:
            body = (opening, closing, round_open, round_close)
            # В букву переходит одна скобка, остальные — в ε: иначе
            # `h([ₙ]ₙ(ₙ)ₙ)` дало бы `aaaa`, и теорема бы не работала.
            homomorphism[opening] = production.rhs[0]
            for symbol in (closing, round_open, round_close):
                homomorphism[symbol] = ""
        else:
            raise ValueError(
                f"правило «{production}» не в нормальной форме Хомского: "
                f"конструкция определена только для `A → BC` и `A → a`. "
                f"Пустое слово скобками не порождается вовсе — если ε "
                f"в языке есть, разбирайте его отдельно: L = L(G′-образ) ∪ {{ε}}"
            )
        productions.append(Production(production.lhs, body))

    return Bracketed(
        CFG(normal.start, tuple(productions), normal.nonterminals, frozenset(terminals)),
        homomorphism,
        normal,
        numbering,
    )


# --------------------------------------------------------------------------
# Язык Дика
# --------------------------------------------------------------------------


def is_dyck(word: Word, pairs) -> bool:
    """Правильная ли скобочная последовательность с учётом типов скобок."""
    opens = {opening: closing for opening, closing in pairs}
    closes = {closing for _, closing in pairs}
    stack: list[str] = []
    for symbol in word:
        if symbol in opens:
            stack.append(opens[symbol])
        elif symbol in closes:
            if not stack or stack.pop() != symbol:
                return False
        else:
            return False
    return not stack


def dyck_words(pairs, max_symbols: int) -> set[Word]:
    """Все правильные скобочные последовательности длины не больше данной.

    Порождается по разложению `D → ε | (ᵢ D )ᵢ D`, а не перебором всех слов
    алфавита с фильтром: слов Дика на порядки меньше, и только так проверка
    теоремы вообще считается.
    """
    if max_symbols < 0:
        return set()
    by_length: dict[int, set[Word]] = {0: {()}}
    for length in range(1, max_symbols + 1):
        found: set[Word] = set()
        if length >= 2:
            for inner in range(0, length - 1):
                rest = length - 2 - inner
                for opening, closing in pairs:
                    for middle in by_length.get(inner, ()):
                        for tail in by_length.get(rest, ()):
                            found.add((opening, *middle, closing, *tail))
        by_length[length] = found
    return {word for words in by_length.values() for word in words}
