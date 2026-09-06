"""Контекстно-свободные грамматики: чистка, нормальные формы, First/Follow,
таблицы LL(1) и LR(0)/SLR(1).

Обслуживает ЛР3 (детерминизм, PDA, регулярные аппроксимации сверху),
второй вопрос экзамена («задаёт ли грамматика LL-язык», «проанализировать
на детерминизм») и задачу 1 РК2 в форме «язык слов грамматики».

Одно различение проходит через весь модуль и является главным источником
ошибок в отчётах: **свойство грамматики — не свойство языка**. Грамматика
может не быть LL(1), а её язык при этом остаётся LL(1) — достаточно
переписать грамматику. Поэтому здесь всё называется честно:
`is_ll1()` — про грамматику, и никакой функции `language_is_ll1()` нет,
потому что этот вопрос в общем случае неразрешим.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

from tfl.verdict import Verdict, refuted, unknown

__all__ = [
    "Production",
    "CFG",
    "parse_cfg",
    "Conflict",
    "LRItem",
    "EPSILON",
    "END",
]

EPSILON = "ε"
END = "$"
ARROWS = ("-->", "->", "→", "::=", "=>")
EPSILON_TOKENS = {"ε", "eps", "", "λ", "\\varepsilon"}


@dataclass(frozen=True, order=True)
class Production:
    """Правило `A → α`. Пустая правая часть означает ε-правило."""

    lhs: str
    rhs: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.lhs} → {' '.join(self.rhs) if self.rhs else EPSILON}"

    @property
    def is_epsilon(self) -> bool:
        return not self.rhs


@dataclass(frozen=True)
class Conflict:
    """Конфликт в таблице разбора."""

    kind: str  # "LL(1)", "shift/reduce", "reduce/reduce"
    where: str  # состояние или пара (нетерминал, символ)
    options: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.kind} в {self.where}: " + " ⨯ ".join(self.options)


@dataclass(frozen=True)
class LRItem:
    """Пункт LR(0): правило с точкой в позиции `dot`."""

    production: Production
    dot: int

    def __str__(self) -> str:
        rhs = list(self.production.rhs)
        rhs.insert(self.dot, "•")
        return f"{self.production.lhs} → {' '.join(rhs)}"

    @property
    def at_end(self) -> bool:
        return self.dot >= len(self.production.rhs)

    @property
    def next_symbol(self) -> str | None:
        return None if self.at_end else self.production.rhs[self.dot]

    def advance(self) -> LRItem:
        return LRItem(self.production, self.dot + 1)


@dataclass
class CFG:
    """Контекстно-свободная грамматика."""

    start: str
    productions: tuple[Production, ...]
    nonterminals: frozenset[str] = field(default=frozenset())
    terminals: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        if not self.nonterminals:
            object.__setattr__(
                self, "nonterminals", frozenset(p.lhs for p in self.productions)
            )
        if not self.terminals:
            symbols = {s for p in self.productions for s in p.rhs}
            object.__setattr__(self, "terminals", frozenset(symbols - self.nonterminals))

    def __str__(self) -> str:
        return "\n".join(str(p) for p in self.productions)

    def __len__(self) -> int:
        return len(self.productions)

    def rules_for(self, nonterminal: str) -> list[Production]:
        return [p for p in self.productions if p.lhs == nonterminal]

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals

    # ------------------------------------------------------------- чистка

    def productive(self) -> set[str]:
        """Нетерминалы, из которых выводится хотя бы одно терминальное слово."""
        good: set[str] = set()
        changed = True
        while changed:
            changed = False
            for p in self.productions:
                if p.lhs in good:
                    continue
                if all(s in self.terminals or s in good for s in p.rhs):
                    good.add(p.lhs)
                    changed = True
        return good

    def reachable(self) -> set[str]:
        """Нетерминалы, достижимые из стартового."""
        seen = {self.start}
        stack = [self.start]
        while stack:
            for p in self.rules_for(stack.pop()):
                for s in p.rhs:
                    if s in self.nonterminals and s not in seen:
                        seen.add(s)
                        stack.append(s)
        return seen

    def clean(self) -> CFG:
        """Убрать непорождающие и недостижимые нетерминалы.

        Порядок важен: сначала непорождающие, потом недостижимые. В обратном
        порядке можно оставить недостижимый мусор, появившийся после
        первой чистки.
        """
        good = self.productive()
        if self.start not in good:
            return CFG(self.start, (), frozenset({self.start}), frozenset())
        step = CFG(
            self.start,
            tuple(
                p
                for p in self.productions
                if p.lhs in good and all(s in self.terminals or s in good for s in p.rhs)
            ),
        )
        alive = step.reachable()
        return CFG(
            self.start,
            tuple(
                p
                for p in step.productions
                if p.lhs in alive and all(s not in step.nonterminals or s in alive for s in p.rhs)
            ),
        )

    def nullable(self) -> set[str]:
        """Нетерминалы, из которых выводится ε."""
        result: set[str] = set()
        changed = True
        while changed:
            changed = False
            for p in self.productions:
                if p.lhs not in result and all(s in result for s in p.rhs):
                    result.add(p.lhs)
                    changed = True
        return result

    # ------------------------------------------------- First / Follow

    def first_k(self, k: int = 1) -> dict[str, set[tuple[str, ...]]]:
        """Множества First_k для всех нетерминалов.

        Элементы — кортежи длины не больше k. Кортеж короче k означает, что
        из нетерминала выводится слово именно такой длины (в частности,
        пустой кортеж — это ε).
        """
        first: dict[str, set[tuple[str, ...]]] = {n: set() for n in self.nonterminals}
        for t in self.terminals:
            first[t] = {(t,)}

        changed = True
        while changed:
            changed = False
            for p in self.productions:
                before = len(first[p.lhs])
                first[p.lhs] |= self._first_of_sequence(p.rhs, first, k)
                if len(first[p.lhs]) != before:
                    changed = True
        return {n: first[n] for n in self.nonterminals}

    def _first_of_sequence(
        self, symbols: Iterable[str], first: dict[str, set[tuple[str, ...]]], k: int
    ) -> set[tuple[str, ...]]:
        """First_k конкатенации: усечённое до k произведение множеств."""
        result: set[tuple[str, ...]] = {()}
        for symbol in symbols:
            pieces = first.get(symbol, {(symbol,)})
            result = {(a + b)[:k] for a in result for b in pieces}
            # Дальше идти незачем, если все префиксы уже набрали длину k
            if all(len(x) == k for x in result):
                break
        return result

    def first_of(self, symbols: Iterable[str], k: int = 1) -> set[tuple[str, ...]]:
        """First_k произвольной цепочки символов."""
        first = self.first_k(k)
        table = dict(first)
        for t in self.terminals:
            table[t] = {(t,)}
        return self._first_of_sequence(symbols, table, k)

    def follow_k(self, k: int = 1) -> dict[str, set[tuple[str, ...]]]:
        """Множества Follow_k. Конец строки обозначается `$`."""
        first = self.first_k(k)
        table = dict(first)
        for t in self.terminals:
            table[t] = {(t,)}

        follow: dict[str, set[tuple[str, ...]]] = {n: set() for n in self.nonterminals}
        follow[self.start] = {(END,) * min(1, k)} if k else {()}

        changed = True
        while changed:
            changed = False
            for p in self.productions:
                for i, symbol in enumerate(p.rhs):
                    if symbol not in self.nonterminals:
                        continue
                    tail = self._first_of_sequence(p.rhs[i + 1 :], table, k)
                    addition = {
                        (a + b)[:k] for a in tail for b in follow[p.lhs]
                    }
                    if not follow[p.lhs]:
                        addition = tail
                    before = len(follow[symbol])
                    follow[symbol] |= addition
                    if len(follow[symbol]) != before:
                        changed = True
        return follow

    # ------------------------------------------------------------ LL(1)

    def ll1_table(self) -> tuple[dict[tuple[str, str], list[Production]], list[Conflict]]:
        """Таблица LL(1)-разбора и список конфликтов.

        Ячейка может содержать несколько правил — это и есть конфликт.
        Возвращаются оба: таблица нужна для отчёта, конфликты — для вердикта.
        """
        first = self.first_k(1)
        follow = self.follow_k(1)
        table: dict[tuple[str, str], list[Production]] = {}
        for p in self.productions:
            heads = self._first_of_sequence(
                p.rhs, {**first, **{t: {(t,)} for t in self.terminals}}, 1
            )
            lookaheads = {h[0] for h in heads if h}
            if () in heads:  # правая часть аннулируема
                lookaheads |= {f[0] for f in follow[p.lhs] if f}
            for token in lookaheads:
                table.setdefault((p.lhs, token), []).append(p)

        conflicts = [
            Conflict("LL(1)", f"({nt}, {token})", tuple(str(p) for p in rules))
            for (nt, token), rules in sorted(table.items())
            if len(rules) > 1
        ]
        return table, conflicts

    def is_ll1(self) -> bool:
        """LL(1) ли **грамматика**.

        Про язык это не говорит ничего: язык может быть LL(1), даже если
        конкретная грамматика — нет. Обратное тоже бывает: устранение левой
        рекурсии часто делает LL(1) грамматику неLL(1)-языка.
        """
        return not self.ll1_table()[1]

    def left_recursive(self) -> set[str]:
        """Нетерминалы с левой рекурсией (в том числе косвенной).

        Левая рекурсия — верный признак того, что грамматика не LL(k)
        ни при каком k, и первое, что стоит проверять.
        """
        nullable = self.nullable()
        begins: dict[str, set[str]] = {n: set() for n in self.nonterminals}
        for p in self.productions:
            for symbol in p.rhs:
                if symbol in self.nonterminals:
                    begins[p.lhs].add(symbol)
                    if symbol not in nullable:
                        break
                else:
                    break
        # транзитивное замыкание
        changed = True
        while changed:
            changed = False
            for n in self.nonterminals:
                addition = set().union(*(begins[m] for m in begins[n])) if begins[n] else set()
                if not addition <= begins[n]:
                    begins[n] |= addition
                    changed = True
        return {n for n in self.nonterminals if n in begins[n]}

    # -------------------------------------------- нормальная форма Хомского

    def _fresh(self, base: str, taken: set[str]) -> str:
        """Свежее имя нетерминала, не пересекающееся с уже занятыми."""
        candidate, i = base, 0
        while candidate in taken:
            i += 1
            candidate = f"{base}{i}"
        return candidate

    def remove_epsilon(self) -> CFG:
        """Убрать ε-правила.

        Если ε принадлежит языку, оно сохраняется единственным правилом
        `S → ε` для стартового нетерминала; при этом стартовый нетерминал
        предварительно выносится наружу, чтобы не встречаться в правых
        частях (иначе `S → ε` пришлось бы применять внутри вывода).
        """
        grammar = self._with_fresh_start() if self.start in self.nullable() else self
        # Пересчитываем после выноса стартового: новый `S₀ → S` тоже аннулируем,
        # и без пересчёта правило `S₀ → ε` теряется вместе с пустым словом.
        nullable = grammar.nullable()

        fresh: set[Production] = set()
        for p in grammar.productions:
            positions = [i for i, s in enumerate(p.rhs) if s in nullable]
            for mask in itertools.product((True, False), repeat=len(positions)):
                drop = {pos for pos, keep in zip(positions, mask) if not keep}
                rhs = tuple(s for i, s in enumerate(p.rhs) if i not in drop)
                if rhs or p.lhs == grammar.start:
                    fresh.add(Production(p.lhs, rhs))
        if grammar.start in nullable:
            fresh.add(Production(grammar.start, ()))
        else:
            fresh = {p for p in fresh if p.rhs}

        return CFG(grammar.start, tuple(sorted(fresh)), grammar.nonterminals, grammar.terminals)

    def _with_fresh_start(self) -> CFG:
        """Новый стартовый нетерминал `S₀ → S`, не встречающийся справа."""
        start = self._fresh(self.start + "₀", set(self.nonterminals) | set(self.terminals))
        extra = Production(start, (self.start,))
        return CFG(
            start,
            (extra,) + self.productions,
            self.nonterminals | {start},
            self.terminals,
        )

    def remove_unit(self) -> CFG:
        """Убрать цепные правила `A → B`.

        Правила `A → B` заменяются на все нецепные правила, достижимые
        из `B` по цепочкам. Именно этого требует фаззер из ЛР3 2024
        («удалить цепные правила») и шаг приведения к НФХ.
        """
        pairs: dict[str, set[str]] = {n: {n} for n in self.nonterminals}
        changed = True
        while changed:
            changed = False
            for p in self.productions:
                if len(p.rhs) == 1 and p.rhs[0] in self.nonterminals:
                    for a in self.nonterminals:
                        if p.lhs in pairs[a] and p.rhs[0] not in pairs[a]:
                            pairs[a].add(p.rhs[0])
                            changed = True

        fresh = {
            Production(a, p.rhs)
            for a in self.nonterminals
            for b in pairs[a]
            for p in self.rules_for(b)
            if not (len(p.rhs) == 1 and p.rhs[0] in self.nonterminals)
        }
        return CFG(self.start, tuple(sorted(fresh)), self.nonterminals, self.terminals)

    def chomsky_normal_form(self) -> CFG:
        """Привести к нормальной форме Хомского.

        Все правила принимают вид `A → BC`, `A → a` и, если ε входит
        в язык, единственное `S → ε` при стартовом `S`, не встречающемся
        справа. Порядок шагов обязателен: сначала ε-правила, потом цепные
        (снятие ε-правил порождает новые цепные), и лишь затем разбиение
        длинных правых частей.

        Форма нужна алгоритму Кока–Янгера–Касами (`tfl.parse.cyk`) и лемме
        о накачке для КС-языков: именно из неё берётся оценка длины
        накачиваемого слова через высоту дерева вывода.
        """
        grammar = self.clean().remove_epsilon().remove_unit().clean()
        taken = set(grammar.nonterminals) | set(grammar.terminals)
        rules: list[Production] = []
        wrappers: dict[str, str] = {}
        pairs: dict[tuple[str, str], str] = {}

        def wrap(symbol: str) -> str:
            """Нетерминал-обёртка для терминала: `A_a → a`."""
            if symbol not in wrappers:
                name = grammar._fresh(f"⟨{symbol}⟩", taken)
                taken.add(name)
                wrappers[symbol] = name
                rules.append(Production(name, (symbol,)))
            return wrappers[symbol]

        def pair(left: str, right: str) -> str:
            """Нетерминал для пары символов — общий для всех правил.

            Без общей таблицы одна и та же пара получает разные имена
            в разных правилах, и грамматика распухает на ровном месте.
            """
            if (left, right) not in pairs:
                name = grammar._fresh(f"⟨{left}{right}⟩", taken)
                taken.add(name)
                pairs[(left, right)] = name
                rules.append(Production(name, (left, right)))
            return pairs[(left, right)]

        for p in grammar.productions:
            if len(p.rhs) == 1 and p.rhs[0] in grammar.terminals:
                rules.append(p)
                continue
            if not p.rhs:
                rules.append(p)  # единственное уцелевшее `S → ε`
                continue
            body = [s if s in grammar.nonterminals else wrap(s) for s in p.rhs]
            while len(body) > 2:
                body = body[:-2] + [pair(body[-2], body[-1])]
            rules.append(Production(p.lhs, tuple(body)))

        nonterminals = {p.lhs for p in rules} | set(grammar.nonterminals)
        terminals = {s for p in rules for s in p.rhs} - nonterminals
        return CFG(grammar.start, tuple(rules), frozenset(nonterminals), frozenset(terminals))

    def is_chomsky_normal_form(self) -> bool:
        """Проверка формы — дешевле, чем доверять построению."""
        for p in self.productions:
            if not p.rhs:
                if p.lhs != self.start:
                    return False
                if any(self.start in q.rhs for q in self.productions):
                    return False
            elif len(p.rhs) == 1:
                if p.rhs[0] not in self.terminals:
                    return False
            elif len(p.rhs) == 2:
                if any(s not in self.nonterminals for s in p.rhs):
                    return False
            else:
                return False
        return True

    # ---------------------------------------------------- LR(0) / SLR(1)

    # ----------------------------------------------------------------------
    # Числовые инварианты вывода
    # ----------------------------------------------------------------------

    def residues(self, coefficients: dict[str, int], modulus: int) -> dict[str, set[int]]:
        """Какие остатки принимает `Σ c_x·|w|_x mod m` на выводимых словах.

        Считается неподвижной точкой по правилам: остаток слова, выведенного
        из `A`, складывается из вклада терминалов правой части и остатков
        слов, выведенных из её нетерминалов. Множество остатков конечно,
        поэтому обход **завершается и точен** — это доказательство для всех
        слов сразу, а не проверка на выборке.

        Приём из проверенных работ РК2 (`reports/rk2-2026-photos/NOTES.md`,
        photo_116 и photo_155): числовой инвариант грамматики против
        числового условия отбора.

        >>> g = parse_cfg("S -> a S b S b | a S a | a")
        >>> sorted(g.residues({"a": 1, "b": -2}, 2)["S"])
        [1]
        """
        if modulus < 2:
            raise ValueError("модуль должен быть не меньше двух")
        table: dict[str, set[int]] = {n: set() for n in self.nonterminals}
        changed = True
        while changed:
            changed = False
            for production in self.productions:
                base = sum(
                    coefficients.get(symbol, 0)
                    for symbol in production.rhs
                    if not self.is_nonterminal(symbol)
                )
                parts = [
                    table[symbol]
                    for symbol in production.rhs
                    if self.is_nonterminal(symbol)
                ]
                if any(not part for part in parts):
                    continue  # нетерминал пока ничего не выводит
                for combination in itertools.product(*parts):
                    value = (base + sum(combination)) % modulus
                    if value not in table[production.lhs]:
                        table[production.lhs].add(value)
                        changed = True
        return table

    def counting_condition(
        self,
        coefficients: dict[str, int],
        moduli: Iterable[int] = range(2, 13),
    ) -> Verdict:
        """Есть ли в языке слово с `Σ c_x·|w|_x = 0`.

        Условия РК2 вида «букв `a` ровно вдвое больше, чем `b`» переводятся
        в равенство `|w|_a − 2·|w|_b = 0`. Если найдётся модуль, при котором
        ноль недостижим, язык **пуст** — и это доказано, а не проверено.

        Обратное неверно: достижимость нуля по всем испробованным модулям
        ничего не доказывает, поэтому вердикт тогда «не выяснено».

        >>> g = parse_cfg("S -> a S a S a | b S b | b")
        >>> g.counting_condition({"a": 1, "b": -1}).value
        False
        """
        for modulus in moduli:
            table = self.residues(coefficients, modulus)
            if 0 not in table.get(self.start, set()):
                return refuted(
                    f"по модулю {modulus} достижимы только остатки "
                    f"{sorted(table.get(self.start, set()))}, нуля среди них нет",
                    modulus,
                )
        return unknown(
            "ни один из испробованных модулей не даёт противоречия; "
            "это не значит, что слово есть"
        )

    def augmented(self) -> tuple[CFG, Production]:
        """Пополненная грамматика `S' → S` — основа LR-построений."""
        fresh = self.start + "'"
        while fresh in self.nonterminals:
            fresh += "'"
        extra = Production(fresh, (self.start,))
        return (
            CFG(
                fresh,
                (extra,) + self.productions,
                self.nonterminals | {fresh},
                self.terminals,
            ),
            extra,
        )

    def closure(self, items: Iterable[LRItem]) -> frozenset[LRItem]:
        """Замыкание множества пунктов LR(0)."""
        result = set(items)
        stack = list(result)
        while stack:
            item = stack.pop()
            symbol = item.next_symbol
            if symbol is None or symbol not in self.nonterminals:
                continue
            for p in self.rules_for(symbol):
                fresh = LRItem(p, 0)
                if fresh not in result:
                    result.add(fresh)
                    stack.append(fresh)
        return frozenset(result)

    def goto(self, items: Iterable[LRItem], symbol: str) -> frozenset[LRItem]:
        return self.closure(
            [item.advance() for item in items if item.next_symbol == symbol]
        )

    def lr0_states(self) -> tuple[list[frozenset[LRItem]], dict[tuple[int, str], int]]:
        """Канонический набор состояний LR(0) и функция переходов.

        Это и есть «позиционный автомат» из ЛР3: пересечение языка грамматики
        с языком этого автомата даёт регулярную аппроксимацию сверху.
        """
        grammar, extra = self.augmented()
        start = grammar.closure([LRItem(extra, 0)])
        states = [start]
        index = {start: 0}
        transitions: dict[tuple[int, str], int] = {}
        queue = [start]
        symbols = sorted(grammar.nonterminals | grammar.terminals)
        while queue:
            current = queue.pop(0)
            for symbol in symbols:
                target = grammar.goto(current, symbol)
                if not target:
                    continue
                if target not in index:
                    index[target] = len(states)
                    states.append(target)
                    queue.append(target)
                transitions[(index[current], symbol)] = index[target]
        return states, transitions

    def lr0_conflicts(self) -> list[Conflict]:
        """Конфликты LR(0): сдвиг/свёртка и свёртка/свёртка."""
        states, transitions = self.lr0_states()
        conflicts: list[Conflict] = []
        for i, state in enumerate(states):
            reductions = [item for item in state if item.at_end]
            shifts = sorted(
                {
                    item.next_symbol
                    for item in state
                    if item.next_symbol in self.terminals
                }
            )
            if reductions and shifts:
                conflicts.append(
                    Conflict(
                        "shift/reduce",
                        f"состояние {i}",
                        tuple([f"сдвиг по {', '.join(shifts)}"] + [str(r) for r in reductions]),
                    )
                )
            if len(reductions) > 1:
                conflicts.append(
                    Conflict(
                        "reduce/reduce",
                        f"состояние {i}",
                        tuple(str(r) for r in reductions),
                    )
                )
        return conflicts

    def slr1_conflicts(self) -> list[Conflict]:
        """Конфликты SLR(1): свёртка разрешена только на символах Follow."""
        states, _ = self.lr0_states()
        follow = self.follow_k(1)
        conflicts: list[Conflict] = []
        for i, state in enumerate(states):
            actions: dict[str, list[str]] = {}
            for item in state:
                if item.at_end:
                    if item.production.lhs not in follow:
                        continue
                    for token in follow[item.production.lhs]:
                        key = token[0] if token else END
                        actions.setdefault(key, []).append(f"свёртка {item}")
                elif item.next_symbol in self.terminals:
                    actions.setdefault(item.next_symbol, []).append(
                        f"сдвиг по {item.next_symbol}"
                    )
            for token, options in sorted(actions.items()):
                unique = sorted(set(options))
                if len(unique) > 1:
                    kind = (
                        "shift/reduce"
                        if any(o.startswith("сдвиг") for o in unique)
                        else "reduce/reduce"
                    )
                    conflicts.append(
                        Conflict(kind, f"состояние {i}, символ {token}", tuple(unique))
                    )
        return conflicts

    def is_lr0(self) -> bool:
        return not self.lr0_conflicts()

    def is_slr1(self) -> bool:
        return not self.slr1_conflicts()


# --------------------------------------------------------------------------
# Разбор текста грамматики
# --------------------------------------------------------------------------


def parse_cfg(text: str, start: str | None = None, nonterminals: str = "") -> CFG:
    """Разобрать грамматику из текста.

    Формат — как в условиях курса: по правилу на строку, альтернативы через
    `|`. Понимает стрелки `->`, `→`, `::=`, `-->`, `=>`.

    Символы правой части определяются так: если в правой части есть пробелы,
    она делится по ним, иначе разбирается посимвольно. Нетерминалами
    считаются заглавные буквы (со штрихами и цифрами: `S'`, `A1`) —
    именно эта договорённость принята в заданиях. Явный список
    `nonterminals` перекрывает эвристику.

    >>> g = parse_cfg("S -> a S b | ε")
    >>> len(g)
    2
    """
    explicit = set(nonterminals)
    productions: list[Production] = []
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
            productions.append((head, alternative.strip()))

    known = explicit | set(heads)

    def tokenize(body: str) -> tuple[str, ...]:
        """Разбить правую часть на символы по нотации курса.

        Пробелы ничего не значат: в условиях одно и то же правило пишут
        и как `S → b T a a T`, и как `S → ab S bb S`. Поэтому строка всегда
        разбирается посимвольно, а заглавная буква вместе со следующими
        за ней штрихами и цифрами (`S'`, `A1`, `S₀`) склеивается в один
        нетерминал.

        Следствие: многосимвольных терминалов вроде `id` эта нотация
        не поддерживает — в заданиях курса их и не бывает.
        """
        if body.strip() in EPSILON_TOKENS:
            return ()
        out: list[str] = []
        i = 0
        while i < len(body):
            ch = body[i]
            if ch.isspace() or ch in EPSILON_TOKENS:
                i += 1
                continue
            if ch.isupper():
                j = i + 1
                while j < len(body) and body[j] in "'′₀₁₂₃0123456789":
                    j += 1
                out.append(body[i:j])
                i = j
            else:
                out.append(ch)
                i += 1
        return tuple(out)

    rules = tuple(Production(head, tokenize(body)) for head, body in productions)
    inferred = known | {
        s for p in rules for s in p.rhs if s[0].isupper()
    }
    start_symbol = start or heads[0]
    terminals = {s for p in rules for s in p.rhs} - inferred
    return CFG(start_symbol, rules, frozenset(inferred), frozenset(terminals))
