"""Ограничение совпадениями (match-bounds) для строковых систем.

Метод Гезера, Хофбауэра и Вальдмана. В отличие от порядков и интерпретаций
он не ищет убывающую меру на словах, а **строит регулярный язык**,
замкнутый относительно переписывания с пометками, и завершимость получается
как следствие.

## Как устроено

Каждой букве приписывается высота: работаем над алфавитом `Σ × ℕ`,
буква `a` высоты `h` записывается `aʰ`. Система `R` поднимается
до `match(R)`: для правила `l → r` и любой расстановки высот
`h₁ … h_k` на буквах `l` правая часть получает высоты, **все равные**
`1 + min(h₁ … h_k)`.

`R` **ограничена совпадениями числом `c`**, если в любом выводе
из слова с нулевыми высотами высоты не превосходят `c`. Свидетель этого —
конечный автомат `A` над `Σ × {0…c}`, у которого

1. `L(A) ⊇ (Σ × {0})*` — все стартовые слова принимаются;
2. `L(A)` замкнут относительно `match(R)`;
3. высоты в алфавите не выше `c`.

## Почему это доказательство завершимости

Пусть `K > |r|` для всех правил. Припишем букве высоты `h` вес `K^{c−h}`,
слову — сумму весов букв. Шаг `u·l′·v → u·r′·v` берёт левую часть
с высотами `h₁ … h_k`, `m = min hᵢ`, и даёт `|r|` букв высоты `m+1`.
Тогда

$$\\text{вес}(l') \\;\\geqslant\\; K^{c-m} \\;>\\; |r| \\cdot K^{c-m-1}
\\;=\\; \\text{вес}(r'),$$

потому что `K > |r|`. Веса — целые неотрицательные числа, значит вывод
конечен. Любой вывод исходной `R` из слова `w` поднимается до вывода
`match(R)` из `w` с нулевыми высотами, а тот целиком лежит в `L(A)`
по замкнутости, значит высоты не выше `c` и вес ограничен. Отсюда

$$\\text{длина вывода из } w \\;\\leqslant\\; |w| \\cdot K^{c},$$

то есть **длина вывода линейна по длине слова**. Это же и граница метода:
система с квадратичной длиной вывода ограниченной совпадениями не бывает
в принципе.

## Что здесь считается, а что проверяется

`find_match_bound` строит автомат пополнением: пока в языке находится
слово `u·l′·v`, для которого `u·r′·v` в языке нет, к языку добавляется
недостающий путь, автомат детерминизируется и минимизируется. Процедура
может и не сойтись — тогда честный ответ «не выяснено».

`MatchBound.check` — **арбитр**, и он от пополнения не зависит: заново
проверяет все три условия операциями над автоматами. Проверка условия 2
точна, а не достаточна: для детерминированного автомата условие
«из `u·l′·v ∈ L` следует `u·r′·v ∈ L`» равносильно тому, что для каждого
достижимого `p` с `δ*(p, l′) = q` определено `δ*(p, r′) = q′`
и правый язык `q` вложен в правый язык `q′`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from tfl.automata import DFA, NFA, difference
from tfl.verdict import Verdict, proved, unknown

__all__ = [
    "Coding",
    "MatchBound",
    "new_height",
    "find_match_bound",
]

SUPERSCRIPT = "⁰¹²³⁴⁵⁶⁷⁸⁹"


@dataclass(frozen=True)
class Coding:
    """Пара «буква, высота» одним символом — автоматы работают с символами."""

    letters: str
    bound: int

    @property
    def symbols(self) -> list[str]:
        return [
            self.encode(letter, height)
            for letter in self.letters
            for height in range(self.bound + 1)
        ]

    @property
    def alphabet(self) -> frozenset[str]:
        return frozenset(self.symbols)

    def encode(self, letter: str, height: int) -> str:
        return chr(0xE000 + self.letters.index(letter) * (self.bound + 1) + height)

    def decode(self, symbol: str) -> tuple[str, int]:
        index = ord(symbol) - 0xE000
        return self.letters[index // (self.bound + 1)], index % (self.bound + 1)

    def lift(self, word: str, height: int = 0) -> str:
        return "".join(self.encode(letter, height) for letter in word)

    def show(self, word: str) -> str:
        """Читаемая запись: `a⁰b¹`. Для отчёта и для сообщений об ошибке."""
        out = []
        for symbol in word:
            letter, height = self.decode(symbol)
            out.append(letter + "".join(SUPERSCRIPT[int(d)] for d in str(height)))
        return "".join(out) or "ε"


def new_height(heights: tuple[int, ...]) -> int:
    """Высота, которую получат все буквы правой части: `1 + min` слева.

    Минимум, а не максимум: иначе высоты росли бы от одного «высокого»
    вхождения и ограниченными не были бы никогда.
    """
    return min(heights) + 1


def _base_automaton(coding: Coding) -> DFA:
    """Автомат языка `(Σ × {0})*` — стартовые слова."""
    delta = {("q", coding.encode(letter, 0)): "q" for letter in coding.letters}
    return DFA(coding.alphabet, "q", frozenset({"q"}), delta)


def _walk(machine: DFA, state, word: str):
    """Куда ведёт слово из состояния; None, если перехода нет."""
    current = state
    for symbol in word:
        current = machine.delta.get((current, symbol))
        if current is None:
            return None
    return current


def _annotations(machine: DFA, state, letters: str, coding: Coding):
    """Все размеченные вхождения `letters`, читаемые из `state`.

    Перебираются не все `(c+1)^|l|` расстановок высот, а только те,
    для которых в автомате есть путь: это и делает пополнение
    осуществимым на словах длины восемь.
    """
    frontier = [(state, ())]
    for letter in letters:
        following = []
        for current, heights in frontier:
            for height in range(coding.bound + 1):
                nxt = machine.delta.get((current, coding.encode(letter, height)))
                if nxt is not None:
                    following.append((nxt, (*heights, height)))
        frontier = following
        if not frontier:
            break
    return frontier


def _right_inclusion(machine: DFA) -> set:
    """Пары `(p, q)`, у которых правый язык `p` вложен в правый язык `q`.

    Считается один раз на автомат обратным обходом: пара плоха, если `p`
    финально, а `q` нет, либо если по какой-то букве плоха пара-преемник.
    Прямая проверка разностью автоматов строила бы произведение на каждую
    пару состояний и делает пополнение неподъёмным — замерено на варианте 10
    ЛР1, где раунд не заканчивался за десять минут.
    """
    full = machine.complete()
    states = sorted(full.states, key=repr)
    alphabet = sorted(full.alphabet)
    bad = {
        (p, q)
        for p in states
        for q in states
        if p in full.finals and q not in full.finals
    }
    back: dict = {}
    for p in states:
        for q in states:
            for symbol in alphabet:
                key = (full.delta[(p, symbol)], full.delta[(q, symbol)])
                back.setdefault(key, []).append((p, q))
    work = list(bad)
    while work:
        pair = work.pop()
        for previous in back.get(pair, ()):
            if previous not in bad:
                bad.add(previous)
                work.append(previous)
    return {(p, q) for p in states for q in states if (p, q) not in bad}


@dataclass(frozen=True)
class MatchBound:
    """Свидетельство ограниченности совпадениями: автомат и потолок высот."""

    coding: Coding
    automaton: DFA
    rounds: int = 0

    @property
    def bound(self) -> int:
        return self.coding.bound

    def violations(self, system, limit: int = 0):
        """Места, где замкнутость нарушена: `(p, r′, q)` и повод.

        Возвращается список; пустой список означает замкнутость.
        Отдельным исходом идёт превышение потолка высот — тогда
        второй элемент пары не пусто.
        """
        machine = self.automaton
        # Состояния с пустым правым языком проверять незачем: контекстов `v`
        # за ними нет, и условие замкнутости на них выполняется впустую.
        alive = machine.coreachable()
        reachable = machine.reachable()
        included = _right_inclusion(machine)
        found = []
        for rule in system.rules:
            for state in sorted(reachable & alive, key=repr):
                for target, heights in _annotations(
                    machine, state, rule.lhs, self.coding
                ):
                    if target not in alive:
                        continue
                    height = new_height(heights)
                    if height > self.coding.bound:
                        return None, (
                            f"высота {height} выше потолка {self.coding.bound}: "
                            f"правило «{rule}» на разметке "
                            f"{self.coding.show(''.join(self.coding.encode(ch, h) for ch, h in zip(rule.lhs, heights)))}"
                        )
                    right = self.coding.lift(rule.rhs, height)
                    landing = _walk(machine, state, right)
                    if landing is not None and (target, landing) in included:
                        continue
                    found.append((state, right, target))
                    if limit and len(found) >= limit:
                        return found, ""
        return found, ""

    def check(self, system) -> Verdict:
        """Арбитр: доказывает ли автомат ограниченность совпадениями.

        Считает заново все три условия и от процедуры пополнения
        не зависит совсем.
        """
        letters = sorted({ch for rule in system.rules for ch in rule.lhs + rule.rhs})
        if not set(letters) <= set(self.coding.letters):
            return unknown(
                f"кодировка не покрывает букв системы: "
                f"{''.join(sorted(set(letters) - set(self.coding.letters)))}"
            )
        used = {
            symbol
            for (_, symbol) in self.automaton.delta
            if symbol in self.automaton.alphabet
        }
        too_high = [s for s in used if self.coding.decode(s)[1] > self.coding.bound]
        if too_high:
            return unknown(
                f"в автомате есть высота выше потолка: "
                f"{self.coding.show(''.join(sorted(too_high)))}"
            )
        missing = difference(_base_automaton(self.coding), self.automaton).shortest_word()
        if missing is not None:
            return unknown(
                f"стартовые слова принимаются не все: «{self.coding.show(missing)}» "
                f"не в языке автомата"
            )
        broken, overflow = self.violations(system, limit=1)
        if broken is None:
            return unknown(f"автомат не замкнут: {overflow}")
        if broken:
            state, right, target = broken[0]
            return unknown(
                f"язык автомата не замкнут относительно match(R): из состояния "
                f"{state!r} размеченная левая часть ведёт в {target!r}, "
                f"а правая «{self.coding.show(right)}» туда не ведёт"
            )
        longest = max((len(rule.rhs) for rule in system.rules), default=0)
        factor = (longest + 1) ** self.coding.bound
        return proved(
            f"система ограничена совпадениями числом {self.coding.bound}: "
            f"автомат из {len(self.automaton)} состояний принимает все слова "
            f"с нулевыми высотами и замкнут относительно match(R). "
            f"Отсюда завершимость, и вывод из слова длины n не длиннее "
            f"{factor}·n шагов",
            self,
        )

    def markdown(self) -> str:
        lines = [
            f"Ограничение совпадениями: **{self.coding.bound}**, "
            f"состояний в автомате: **{len(self.automaton)}**, "
            f"раундов пополнения: {self.rounds}.",
            "",
            "```dot",
            self.automaton.to_dot("match"),
            "```",
        ]
        return "\n".join(lines)


def _grow(machine: DFA, additions, coding: Coding) -> DFA:
    """Добавить недостающие пути и вернуть детерминированный минимальный автомат."""
    delta: dict[tuple, set] = {}
    for (src, symbol), dst in machine.delta.items():
        delta.setdefault((src, symbol), set()).add(dst)
    eps: dict = {}
    fresh = 0
    for state, right, target in additions:
        if not right:
            eps.setdefault(state, set()).add(target)
            continue
        current = state
        for index, symbol in enumerate(right):
            if index == len(right) - 1:
                nxt = target
            else:
                fresh += 1
                nxt = ("+", fresh)
            delta.setdefault((current, symbol), set()).add(nxt)
            current = nxt
    nfa = NFA(
        machine.alphabet,
        machine.start,
        machine.finals,
        {key: frozenset(value) for key, value in delta.items()},
        {key: frozenset(value) for key, value in eps.items()},
    )
    return nfa.determinize().minimize().relabel()


def find_match_bound(
    system,
    bound: int = 3,
    max_states: int = 400,
    max_rounds: int = 40,
    timeout_s: float = 60.0,
    batch: int = 12,
) -> Verdict:
    """Искать доказательство ограниченности совпадениями.

    Пополнение идёт раундами: за раунд берётся не больше `batch`
    нарушений замкнутости, добавляются недостающие пути, автомат
    детерминизируется и минимизируется. Минимизация здесь не косметика —
    без неё автомат растёт на каждом раунде и процедура не сходится
    никогда. Порция тоже не косметика: детерминизацию нельзя прервать
    на полпути, и раунд, добавляющий сотню путей сразу, уходит в неё
    на минуты — замерено на варианте 11 ЛР1, где третий раунд
    не заканчивался вовсе.

    Три исхода. `True` — построен автомат, прошедший проверку арбитром.
    «Не выяснено» — потолок высот пробит, автомат перерос `max_states`,
    раунды или время кончились: про завершимость отсюда не следует
    ничего, метод её только доказывает.
    """
    letters = "".join(sorted({ch for rule in system.rules for ch in rule.lhs + rule.rhs}))
    if not letters:
        return proved("в системе нет букв: переписывать нечего")
    if any(not rule.lhs for rule in system.rules):
        return unknown("правило с пустой левой частью: система не завершима")
    coding = Coding(letters, bound)
    witness = MatchBound(coding, _base_automaton(coding))
    deadline = time.monotonic() + timeout_s
    for round_number in range(1, max_rounds + 1):
        if time.monotonic() > deadline:
            return unknown(
                f"пополнение не уложилось в {timeout_s:.0f} с: остановлено "
                f"на раунде {round_number}, автомат — "
                f"{len(witness.automaton)} состояний"
            )
        additions, overflow = witness.violations(system, limit=batch)
        if additions is None:
            return unknown(
                f"пополнение уперлось в потолок высот на раунде {round_number}: "
                f"{overflow}. Попробуйте больший bound"
            )
        if not additions:
            found = MatchBound(coding, witness.automaton, round_number - 1)
            verified = found.check(system)
            if verified.value is not True:
                return unknown(
                    f"пополнение сошлось, но арбитр не принял автомат: "
                    f"{verified.reason}"
                )
            return verified
        # Детерминизация — самая дорогая часть раунда, и остановить её
        # на полпути нельзя. Поэтому размер НКА оценивается заранее:
        # без этого вариант 11 ЛР1 уходил в один раунд на минуты.
        planned = len(witness.automaton) + sum(
            max(len(right) - 1, 0) for _, right, _ in additions
        )
        if planned > 4 * max_states:
            return unknown(
                f"на раунде {round_number} к автомату из "
                f"{len(witness.automaton)} состояний пришлось бы добавить "
                f"{planned - len(witness.automaton)} — это выше потолка "
                f"{4 * max_states}"
            )
        grown = _grow(witness.automaton, additions, coding)
        if len(grown) > max_states:
            return unknown(
                f"автомат пополнения вырос до {len(grown)} состояний "
                f"на раунде {round_number} при потолке {max_states}"
            )
        witness = MatchBound(coding, grown, round_number)
    return unknown(
        f"пополнение не сошлось за {max_rounds} раундов; последний автомат — "
        f"{len(witness.automaton)} состояний"
    )
