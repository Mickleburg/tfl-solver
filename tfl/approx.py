"""Регулярные аппроксимации КС-грамматики сверху и пересечение с ними.

Задание ЛР3 требует пересечь грамматику с двумя регулярными
аппроксимациями сверху — «LL(1)-автоматом» и «LR(0)-автоматом»
(позиционным) — и автоматически проверить, что язык при этом не
изменился.

Обе аппроксимации получаются одним приёмом: **берём магазинный
распознаватель и забываем стек**. Автомат без стека не может проверить,
что возврат из нетерминала происходит именно в то место, откуда был
вызов, поэтому возврат разрешается в любое подходящее — язык от этого
может только расшириться. Для LR(0)-автомата это конструкция
Перейры–Райта; в лекции 9 разобран пример: язык `(ⁿ a )ⁿ`
аппроксимируется языком `(* a )*`.

Одно свойство здесь машинно проверяемо и потому обязательно проверяется:
аппроксимация — надмножество. Если пересечение с ней отрезало от языка
хоть одно слово, ошибка в построении, а не в грамматике.

Осторожно с направлением вывода: совпадение языков **не** означает,
что аппроксимация точна — оно означает лишь, что на словах до заданной
длины расхождений нет.
"""

from __future__ import annotations

from typing import Iterator

from tfl.automata import DFA, NFA, State
from tfl.cfg import CFG, LRItem, Production
from tfl.parse import language
from tfl.words import iter_words

__all__ = [
    "lr0_automaton",
    "ll1_automaton",
    "to_dfa",
    "intersect",
    "over_approximates",
    "triple_name",
    "surplus",
]


def to_dfa(nfa: NFA) -> DFA:
    """Детерминизировать и перенумеровать — чтобы состояния в отчёте были числами."""
    return nfa.determinize().relabel()


# --------------------------------------------------------------------------
# Аппроксимации
# --------------------------------------------------------------------------


def lr0_automaton(grammar: CFG) -> NFA:
    """Позиционный («LR(0)-») автомат: аппроксимация сверху по Перейре–Райту.

    Состояния — множества пунктов LR(0), переходы по терминалам взяты
    из канонического набора. Свёртка `A → α•` в настоящем разборе снимает
    со стека `|α|` символов и переходит в `goto(верхушка, A)`; забыв стек,
    мы не знаем верхушку и разрешаем ε-переход в `goto(j, A)` **для любого**
    состояния `j`, где переход по `A` определён.

    Отсюда и расширение языка: скобки перестают считаться, потому что
    возврат больше не привязан к месту вызова.
    """
    aug, extra = grammar.augmented()
    states, moves = grammar.lr0_states()

    delta: dict[tuple[State, str], frozenset[State]] = {}
    for (index, symbol), target in moves.items():
        if symbol in grammar.terminals:
            delta[(index, symbol)] = frozenset({target})

    returns: dict[str, set[int]] = {}
    for (index, symbol), target in moves.items():
        if symbol in grammar.nonterminals:
            returns.setdefault(symbol, set()).add(target)

    eps: dict[State, frozenset[State]] = {}
    for index, state in enumerate(states):
        targets: set[State] = set()
        for item in state:
            if item.at_end and item.production != extra:
                targets |= returns.get(item.production.lhs, set())
        if targets:
            eps[index] = frozenset(targets)

    return NFA(
        frozenset(grammar.terminals),
        0,
        frozenset({moves[(0, grammar.start)]}),
        delta,
        eps,
    )


def ll1_automaton(grammar: CFG) -> NFA:
    """«LL(1)-автомат»: та же аппроксимация, но для нисходящего разбора.

    Состояния — пункты `A → α•β` (они же ситуации, по которым строится
    таблица LL(1) через First и Follow). Переходы:

    * по терминалу — сдвиг точки;
    * ε-вызов `A → α•Bβ` ⟶ `B → •γ` для каждого правила `B → γ`;
    * ε-возврат `B → γ•` ⟶ `A → αB•β` для **каждого** места вызова `B`.

    Расширение языка возникает в возврате — ровно как в `lr0_automaton`,
    только сеть построена сверху вниз, а не снизу вверх.

    Формулировка задания («аппроксимация, полученная использованием First,
    Follow множеств») точной конструкции не задаёт, а в лекциях назван лишь
    LR(0)-случай. Здесь взят прямой нисходящий двойник Перейры–Райта;
    расхождение с ожиданиями преподавателя не исключено — см.
    `docs/OPEN-GAPS.md`.
    """
    aug, extra = grammar.augmented()
    items = [
        LRItem(p, dot)
        for p in aug.productions
        for dot in range(len(p.rhs) + 1)
    ]

    delta: dict[tuple[State, str], frozenset[State]] = {}
    eps: dict[State, set[State]] = {}
    for item in items:
        symbol = item.next_symbol
        if symbol is None:
            if item.production != extra:
                eps.setdefault(item, set()).update(
                    call.advance() for call in items if call.next_symbol == item.production.lhs
                )
        elif symbol in aug.nonterminals:
            eps.setdefault(item, set()).update(
                LRItem(p, 0) for p in aug.rules_for(symbol)
            )
        else:
            delta[(item, symbol)] = frozenset({item.advance()})

    return NFA(
        frozenset(grammar.terminals),
        LRItem(extra, 0),
        frozenset({LRItem(extra, 1)}),
        delta,
        {k: frozenset(v) for k, v in eps.items()},
    )


# --------------------------------------------------------------------------
# Пересечение грамматики с автоматом
# --------------------------------------------------------------------------


def triple_name(source: State, symbol: str, target: State) -> str:
    """Имя нетерминала `⟨p, A, q⟩` — как в отчётах по ЛР3."""
    return f"⟨{source},{symbol},{target}⟩"


def intersect(
    grammar: CFG, dfa: DFA, start: str = "S₀", max_productions: int = 100_000
) -> CFG:
    """Грамматика пересечения `L(G) ∩ L(M)` — конструкция Бар-Хиллела.

    Нетерминал `⟨p, A, q⟩` порождает те слова, которые выводятся из `A`
    и переводят автомат из `p` в `q`. Правило `A → X₁…X_n` превращается в

        ⟨p₀, A, p_n⟩ → ⟨p₀, X₁, p₁⟩ … ⟨p_{n−1}, X_n, p_n⟩,

    где терминал `a` требует настоящего перехода `δ(p, a) = q` и остаётся
    терминалом.

    Наивный перебор всех наборов промежуточных состояний даёт `|Q|ⁿ⁺¹`
    правил и на грамматиках ЛР3 не проходит, поэтому сначала неподвижной
    точкой вычисляются **порождающие** тройки, и перебор идёт только по
    ним: получаются сразу те правила, которые переживут чистку.
    """
    states = sorted(dfa.states, key=str)
    terminals = grammar.terminals
    delta = dfa.delta

    # Неподвижная точка: какие пары (p, q) достижимы выводом из нетерминала.
    spans: dict[str, set[tuple[State, State]]] = {n: set() for n in grammar.nonterminals}
    changed = True
    while changed:
        changed = False
        for p in grammar.productions:
            for begin in states:
                current = {begin}
                for symbol in p.rhs:
                    following: set[State] = set()
                    for state in current:
                        if symbol in terminals:
                            target = delta.get((state, symbol))
                            if target is not None:
                                following.add(target)
                        else:
                            following |= {
                                end for (src, end) in spans[symbol] if src == state
                            }
                    current = following
                    if not current:
                        break
                fresh = {(begin, end) for end in current} - spans[p.lhs]
                if fresh:
                    spans[p.lhs] |= fresh
                    changed = True

    def walk(rhs: tuple[str, ...], state: State) -> Iterator[tuple[tuple[str, ...], State]]:
        """Все разметки правой части путями автомата, идущими из `state`."""
        if not rhs:
            yield (), state
            return
        head, tail = rhs[0], rhs[1:]
        if head in terminals:
            target = delta.get((state, head))
            if target is None:
                return
            for body, end in walk(tail, target):
                yield (head,) + body, end
        else:
            for source, target in sorted(spans[head], key=str):
                if source != state:
                    continue
                for body, end in walk(tail, target):
                    yield (triple_name(state, head, target),) + body, end

    productions: list[Production] = []
    for p in grammar.productions:
        for begin in states:
            for body, end in walk(p.rhs, begin):
                productions.append(Production(triple_name(begin, p.lhs, end), body))
                if len(productions) > max_productions:
                    raise ValueError(
                        f"пересечение вышло за {max_productions} правил; "
                        "минимизируйте автомат или поднимите max_productions"
                    )

    productions += [
        Production(start, (triple_name(dfa.start, grammar.start, final),))
        for final in sorted(dfa.finals, key=str)
        if (dfa.start, final) in spans[grammar.start]
    ]

    nonterminals = {start} | {p.lhs for p in productions}
    nonterminals |= {s for p in productions for s in p.rhs if s.startswith("⟨")}
    rest = {s for p in productions for s in p.rhs} - nonterminals
    return CFG(start, tuple(productions), frozenset(nonterminals), frozenset(rest)).clean()


def over_approximates(grammar: CFG, dfa: DFA, max_len: int = 8) -> list[str]:
    """Слова языка грамматики, которые автомат **не** принимает.

    Список обязан быть пустым: аппроксимация строится сверху. Непустой —
    признак ошибки в построении автомата, и проверять это надо до того,
    как считать пересечение.
    """
    return [word for word in sorted(language(grammar, max_len), key=lambda w: (len(w), w))
            if not dfa.accepts(word)]


def surplus(dfa: DFA, grammar: CFG, alphabet: str, max_len: int = 8, limit: int = 10) -> list[str]:
    """Слова, которые аппроксимация приняла лишними — мера её грубости.

    Для отчёта полезнее пустого списка: они показывают, что именно
    автомат перестал различать, забыв стек.
    """
    inside = language(grammar, max_len)
    extra: list[str] = []
    for word in iter_words(alphabet, max_len):
        if dfa.accepts(word) and word not in inside:
            extra.append(word)
            if len(extra) >= limit:
                break
    return extra
