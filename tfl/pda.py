"""Магазинные автоматы (PDA/DPDA): симуляция, детерминизм, переходы к КС-грамматике.

Обслуживает ЛР3 целиком и вопрос экзамена «построить распознаватель языка».

Разделение труда здесь такое же, как во всём проекте: **строит человек
(или LLM), проверяет оракул**. Автомат для языка из условия придумывается
содержательно — из наблюдения, что именно нужно помнить в стеке. Модуль
берёт на себя то, что механизируется:

* `PDA.accepts` — симуляция с бюджетом, вердикт вместо `bool`;
* `PDA.nondeterminism` — структурная проверка детерминизма по критерию
  из лекции 9 (см. `nondeterminism`), с предъявлением конфликтующей пары;
* `to_cfg` — тройная конструкция, дающая грамматику языка автомата,
  после чего автомат можно сравнить с любой другой грамматикой словом
  за словом.

Пара `from_cfg` / `to_cfg` — независимые в обе стороны переходы, и
`to_cfg(from_cfg(G))` обязан задавать тот же язык, что `G`. Это не
тавтология: конструкции разные, и расхождение означает ошибку.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

from tfl.cfg import CFG, Production
from tfl.verdict import Verdict, proved, refuted, unknown
from tfl.words import iter_words

__all__ = [
    "Transition",
    "PDA",
    "Choice",
    "from_cfg",
    "to_cfg",
    "parse_pda",
    "disagreements",
    "EMPTY_TOKENS",
]

State = Hashable
EMPTY_TOKENS = {"ε", "eps", "-", "", "λ", "/"}
ARROWS = ("-->", "->", "→", "=>")


@dataclass(frozen=True)
class Transition:
    """Переход `(source, read, pop) → (target, push)`.

    `read` — читаемый символ, пустая строка означает ε-переход.
    `pop` и `push` — цепочки стековых символов, **вершина стека первая**.
    Обобщённая форма (снимаем и кладём цепочки) нужна для свёрток:
    свёртка по правилу `A → α` снимает со стека сразу `|α|` символов.
    """

    source: State
    read: str
    pop: tuple[str, ...]
    push: tuple[str, ...]
    target: State

    def __str__(self) -> str:
        def show(chain: tuple[str, ...]) -> str:
            return " ".join(chain) if chain else "ε"

        return (
            f"{self.source}, {self.read or 'ε'}, {show(self.pop)}"
            f" → {self.target}, {show(self.push)}"
        )

    @property
    def is_epsilon(self) -> bool:
        return self.read == ""


@dataclass(frozen=True)
class Choice:
    """Недетерминированная развилка: два перехода, применимых одновременно.

    В отчёте по ЛР3 каждую такую развилку нужно либо устранить, либо
    обосновать — привести пару слов с общим длинным префиксом, на которых
    поведение стека существенно расходится.
    """

    left: Transition
    right: Transition
    reason: str

    def __str__(self) -> str:
        return f"{self.reason}:\n    {self.left}\n    {self.right}"


@dataclass
class PDA:
    """Магазинный автомат.

    `accept_by`:

    * `"empty"` — по пустому стеку (так принимает автомат, построенный
      по грамматике, и так удобно проверять беспрефиксность);
    * `"state"` — по финальному состоянию;
    * `"both"` — по обоим условиям сразу.

    Допуск по пустому стеку и по состоянию равномощны для PDA, но **не**
    для DPDA: по пустому стеку DPDA распознаёт ровно беспрефиксные
    детерминированные языки. Из-за этого в ЛР3 беспрефиксность и стоит
    рядом с детерминизмом — см. `tfl.parse.prefix_free`.
    """

    start: State
    start_stack: tuple[str, ...]
    transitions: tuple[Transition, ...]
    accepting: frozenset[State] = field(default=frozenset())
    accept_by: str = "empty"

    def __post_init__(self) -> None:
        if self.accept_by not in {"empty", "state", "both"}:
            raise ValueError(f"неизвестный вид допуска: {self.accept_by!r}")
        if self.accept_by != "empty" and not self.accepting:
            raise ValueError("допуск по состоянию требует непустого множества финальных")

    def __str__(self) -> str:
        head = [
            f"start: {self.start}",
            f"stack: {' '.join(self.start_stack) or 'ε'}",
            "accept: "
            + (
                self.accept_by
                if self.accept_by == "empty"
                else f"{self.accept_by} " + " ".join(map(str, sorted(map(str, self.accepting))))
            ),
        ]
        return "\n".join(head + [str(t) for t in self.transitions])

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def states(self) -> set[State]:
        found: set[State] = {self.start, *self.accepting}
        for t in self.transitions:
            found |= {t.source, t.target}
        return found

    @property
    def stack_alphabet(self) -> set[str]:
        found = set(self.start_stack)
        for t in self.transitions:
            found |= set(t.pop) | set(t.push)
        return found

    @property
    def alphabet(self) -> set[str]:
        return {t.read for t in self.transitions if t.read}

    def outgoing(self, state: State) -> list[Transition]:
        return [t for t in self.transitions if t.source == state]

    # ------------------------------------------------------------ симуляция

    def min_yield(self) -> dict[str, int] | None:
        """Сколько символов входа минимально нужно, чтобы снять символ со стека.

        Оценка снизу и намеренно оптимистичная: состояния не учитываются,
        поэтому настоящая стоимость не меньше. Этого достаточно для
        отсечения — конфигурация выбрасывается, только если даже
        оптимистичная оценка превышает остаток входа.

        Возвращает `None`, если хоть один переход снимает не ровно один
        символ: тогда стоимость не раскладывается по символам и отсечения
        не будет.
        """
        if any(len(t.pop) != 1 for t in self.transitions):
            return None
        cost: dict[str, float] = {x: float("inf") for x in self.stack_alphabet}
        changed = True
        while changed:
            changed = False
            for t in self.transitions:
                estimate = len(t.read) + sum(cost[y] for y in t.push)
                if estimate < cost[t.pop[0]]:
                    cost[t.pop[0]] = estimate
                    changed = True
        return {x: int(c) for x, c in cost.items() if c != float("inf")}

    def _is_final(self, state: State, stack: tuple[str, ...]) -> bool:
        empty = not stack
        final = state in self.accepting
        if self.accept_by == "empty":
            return empty
        if self.accept_by == "state":
            return final
        return empty and final

    def accepts(
        self,
        word: str,
        max_stack: int | None = None,
        max_configs: int = 200_000,
    ) -> Verdict:
        """Принимается ли слово. Возвращает вердикт, а не `bool`.

        Множество конфигураций бесконечно (стек не ограничен), поэтому обход
        ведётся с бюджетом. Исходы честно разделены:

        * найдена принимающая конфигурация — **доказано**, свидетель — путь;
        * обход исчерпан целиком, ничего не найдено — **опровергнуто**;
        * бюджет кончился раньше — **не выяснено**.

        Третий случай нельзя выдавать за второй: у автомата с растущим
        стеком принимающий путь может лежать за границей отсечения.
        """
        limit = max_stack if max_stack is not None else 2 * len(word) + 10
        # Стек придётся опустошить, значит его содержимое обязано выдать
        # остаток входа. Отсечение по этой оценке снизу — единственное, что
        # спасает нисходящий автомат от левой рекурсии: раскрытие `S → S a S b`
        # не читает входа, но требуемая выдача растёт и быстро перерастает
        # длину слова. При допуске по состоянию стек опустошать не обязательно,
        # поэтому там отсечение неприменимо.
        cost = self.min_yield() if self.accept_by in {"empty", "both"} else None

        def hopeless(stack: tuple[str, ...], left: int) -> bool:
            if cost is None:
                return False
            total = 0
            for symbol in stack:
                total += cost.get(symbol, 0)
                if total > left:
                    return True
            return False

        start_config = (0, self.start, self.start_stack)
        seen = {start_config}
        queue: deque[tuple[tuple[int, State, tuple[str, ...]], list[Transition]]]
        queue = deque([(start_config, [])])
        truncated = False
        visited = 0

        while queue:
            (pos, state, stack), path = queue.popleft()
            visited += 1
            if pos == len(word) and self._is_final(state, stack):
                return proved("найден принимающий путь", path)
            if visited > max_configs:
                truncated = True
                break
            for t in self.transitions:
                if t.source != state:
                    continue
                if len(t.pop) > len(stack) or tuple(stack[: len(t.pop)]) != t.pop:
                    continue
                if t.read:
                    if pos >= len(word) or word[pos] != t.read:
                        continue
                    step = 1
                else:
                    step = 0
                fresh = t.push + stack[len(t.pop) :]
                if len(fresh) > limit:
                    truncated = True
                    continue
                config = (pos + step, t.target, fresh)
                if config in seen or hopeless(fresh, len(word) - pos - step):
                    continue
                seen.add(config)
                queue.append((config, path + [t]))

        if truncated:
            return unknown(
                f"обход обрезан (стек > {limit} или конфигураций > {max_configs})"
            )
        return refuted("все достижимые конфигурации перебраны, принимающей нет")

    def accepts_or_raise(self, word: str, **kwargs: object) -> bool:
        """`accepts`, но «не выяснено» — это ошибка, а не ответ.

        Годится там, где автомат заведомо мал и бюджет заведомо достаточен
        (например, в тестах), и категорически не годится для отчёта.
        """
        verdict = self.accepts(word, **kwargs)  # type: ignore[arg-type]
        if verdict.value is None:
            raise RuntimeError(f"на слове «{word or 'ε'}» бюджета не хватило: {verdict.reason}")
        return verdict.value

    # ----------------------------------------------------------- детерминизм

    def nondeterminism(self) -> list[Choice]:
        """Развилки, нарушающие детерминизм.

        Критерий из лекции 9: из одного состояния при одной и той же вершине
        стека **если есть ε-переход, то он единственный и других переходов
        нет**; переходов по одному и тому же символу тоже не больше одного.

        Два перехода могут сработать одновременно, если их снимаемые цепочки
        совместимы, то есть одна является префиксом другой: тогда найдётся
        стек, к которому применимы обе.
        """
        conflicts: list[Choice] = []
        for left, right in itertools.combinations(self.transitions, 2):
            if left.source != right.source:
                continue
            short, long = sorted((left.pop, right.pop), key=len)
            if long[: len(short)] != short:
                continue  # снимаемые цепочки несовместимы — оба сразу не сработают
            if left.is_epsilon or right.is_epsilon:
                reason = (
                    "ε-переход не единственный из состояния"
                    if left.is_epsilon and right.is_epsilon
                    else "из состояния есть и ε-переход, и переход по символу"
                )
            elif left.read == right.read:
                reason = f"два перехода по символу «{left.read}»"
            else:
                continue
            conflicts.append(Choice(left, right, reason))
        return conflicts

    def is_deterministic(self) -> bool:
        """DPDA ли автомат (структурно).

        Это свойство **автомата**, не языка. Недетерминированный автомат
        вполне может распознавать детерминированный язык — вывод «язык
        недетерминирован» из наличия развилок неправомерен и является
        типичной ошибкой в отчётах.
        """
        return not self.nondeterminism()

    def epsilon_transitions(self) -> list[Transition]:
        """Переходы, не читающие с ленты."""
        return [t for t in self.transitions if t.is_epsilon]

    def is_real_time(self) -> bool:
        """Читает ли автомат букву на каждом шаге — то есть нет ε-переходов.

        Real-time DPDA — это DPDA без ε-переходов. Свойство разбалловывается
        отдельно: по разбалловке РК2 2023 доказательство, что язык
        не real-time, стоит 1 балл в случае «не КС», и оно же идёт в счёт
        2-3 баллов в случае детерминированного языка.

        Как и `is_deterministic`, это свойство **автомата**, а не языка.
        Real-time называют язык, для которого такой автомат существует;
        предъявленный автомат с ε-переходами про язык не говорит ничего.
        """
        return not any(t.is_epsilon for t in self.transitions)

    def multi_pop_transitions(self) -> list[Transition]:
        """Переходы, снимающие со стека больше одного символа.

        > А сбросить сразу несколько символов за один шаг мы не можем
        > по определению. (авторский разбор РК2, задача про real-time)

        `Transition` здесь обобщён — снимается цепочка, — потому что это
        нужно для свёрток LR. Определение PDA в курсе такого не разрешает,
        поэтому автомат, предъявляемый как ответ, обязан обходиться
        снятием одного символа. Обычный приём — **слить снимаемые подряд
        символы в один**: класть `A` не на каждую букву, а на каждую
        вторую, и снимать по одному.
        """
        return [t for t in self.transitions if len(t.pop) > 1]

    # ---------------------------------------------------------------- вывод

    def to_dot(self, name: str = "PDA") -> str:
        labels = {s: i for i, s in enumerate(sorted(self.states, key=str))}
        lines = [f"digraph {name} {{", "  rankdir=LR;", '  __start [shape=point];']
        for state, i in labels.items():
            shape = "doublecircle" if state in self.accepting else "circle"
            lines.append(f'  s{i} [label="{state}", shape={shape}];')
        lines.append(f"  __start -> s{labels[self.start]};")
        for t in self.transitions:
            pop = " ".join(t.pop) or "ε"
            push = " ".join(t.push) or "ε"
            label = f"{t.read or 'ε'}, {pop}/{push}"
            lines.append(f'  s{labels[t.source]} -> s{labels[t.target]} [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Грамматика → автомат и обратно
# --------------------------------------------------------------------------


def from_cfg(grammar: CFG, state: State = "q") -> PDA:
    """Нисходящий распознаватель по грамматике: одно состояние, допуск по пустому стеку.

    Стек хранит ещё не разобранную часть сентенциальной формы:
    раскрытие нетерминала — ε-переход, чтение терминала — снятие его
    с вершины. Автомат почти никогда не детерминирован (раскрытий у
    нетерминала обычно несколько) и служит не ответом на задание, а
    отправной точкой и эталоном для сравнения.
    """
    transitions = [
        Transition(state, "", (p.lhs,), p.rhs, state) for p in grammar.productions
    ]
    transitions += [
        Transition(state, t, (t,), (), state) for t in sorted(grammar.terminals)
    ]
    return PDA(state, (grammar.start,), tuple(transitions), accept_by="empty")


def to_cfg(pda: PDA, start: str = "S₀", max_productions: int = 200_000) -> CFG:
    """Грамматика языка автомата (тройная конструкция).

    Нетерминал `⟨p, X, q⟩` порождает ровно те слова, читая которые автомат
    переходит из `p` в `q`, сняв со стека `X` и не тронув то, что под ним.
    Отсюда правила: переход `(p, a, X) → (r, Y₁…Y_k)` даёт

        ⟨p, X, s_k⟩ → a ⟨r, Y₁, s₁⟩⟨s₁, Y₂, s₂⟩ … ⟨s_{k−1}, Y_k, s_k⟩

    для всех наборов промежуточных состояний. Перебор наборов — источник
    комбинаторного взрыва, поэтому стоит ограничитель; лишнее снимается
    чисткой.

    Требует допуска по пустому стеку и снятия ровно одного символа за
    переход — к этому виду приводится любой PDA, а `from_cfg` строит его
    сразу таким.
    """
    if pda.accept_by != "empty":
        raise ValueError("тройная конструкция определена для допуска по пустому стеку")
    if any(len(t.pop) != 1 for t in pda.transitions):
        raise ValueError("каждый переход должен снимать ровно один стековый символ")
    if len(pda.start_stack) != 1:
        raise ValueError("стартовый стек должен состоять из одного символа")

    states = sorted(pda.states, key=str)

    def triple(p: State, x: str, q: State) -> str:
        return f"⟨{p},{x},{q}⟩"

    productions: list[Production] = [
        Production(start, (triple(pda.start, pda.start_stack[0], q),)) for q in states
    ]
    for t in pda.transitions:
        head = (t.read,) if t.read else ()
        k = len(t.push)
        for chain in itertools.product(states, repeat=k):
            path = (t.target,) + chain
            body = tuple(
                triple(path[i], t.push[i], path[i + 1]) for i in range(k)
            )
            productions.append(
                Production(triple(t.source, t.pop[0], chain[-1] if chain else t.target),
                           head + body)
            )
            if len(productions) > max_productions:
                raise ValueError(
                    f"тройная конструкция вышла за {max_productions} правил; "
                    "уменьшите автомат или поднимите max_productions"
                )

    nonterminals = {start} | {p.lhs for p in productions}
    nonterminals |= {s for p in productions for s in p.rhs if s.startswith("⟨")}
    terminals = {s for p in productions for s in p.rhs} - nonterminals
    return CFG(start, tuple(productions), frozenset(nonterminals), frozenset(terminals)).clean()


# --------------------------------------------------------------------------
# Чтение из текста
# --------------------------------------------------------------------------


def parse_pda(text: str) -> PDA:
    """Разобрать автомат из текста.

    Заголовок задаёт стартовое состояние, стартовый стек и вид допуска::

        start: q0
        stack: Z
        accept: state q2        # либо `accept: empty`, либо `accept: both q2`

    Дальше по переходу на строку в виде `состояние, символ, снять -> состояние, положить`::

        q0, a, Z -> q0, A Z
        q0, ε, Z -> q1, ε

    Цепочки стековых символов пишутся вершиной влево. Если в цепочке нет
    пробелов, она разбирается посимвольно — так же, как правые части правил
    в `tfl.cfg.parse_cfg`.
    """
    start: State | None = None
    stack: tuple[str, ...] = ()
    accept_by = "empty"
    accepting: set[State] = set()
    transitions: list[Transition] = []

    def chain(raw: str) -> tuple[str, ...]:
        body = raw.strip()
        if body in EMPTY_TOKENS:
            return ()
        if " " in body:
            return tuple(part for part in body.split() if part not in EMPTY_TOKENS)
        return tuple(body)

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("start:"):
            start = line[len("start:") :].strip()
            continue
        if line.startswith("stack:"):
            stack = chain(line[len("stack:") :])
            continue
        if line.startswith("accept:"):
            parts = line[len("accept:") :].split()
            if not parts:
                raise ValueError(f"пустая строка допуска: {raw!r}")
            accept_by, accepting = parts[0], set(parts[1:])
            continue

        for arrow in ARROWS:
            if arrow in line:
                left, right = line.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {raw!r}")
        source, read, pop = (part.strip() for part in left.split(",", 2))
        target, push = (part.strip() for part in right.split(",", 1))
        transitions.append(
            Transition(
                source,
                "" if read in EMPTY_TOKENS else read,
                chain(pop),
                chain(push),
                target,
            )
        )

    if start is None:
        raise ValueError("не задано стартовое состояние (строка `start:`)")
    return PDA(start, stack, tuple(transitions), frozenset(accepting), accept_by)


# --------------------------------------------------------------------------
# Сравнение с эталоном
# --------------------------------------------------------------------------


def disagreements(
    pda: PDA,
    predicate: Callable[[str], bool],
    alphabet: Iterable[str],
    max_len: int = 8,
    limit: int = 10,
) -> tuple[list[str], list[str]]:
    """Слова, на которых автомат расходится с эталоном, и слова без вердикта.

    Возвращает две части, и их нельзя смешивать: первая — настоящие
    расхождения (опровержение гипотезы «автомат распознаёт этот язык»),
    вторая — слова, на которых бюджет симуляции кончился. Пустая первая
    часть при непустой второй **не** означает, что проверка пройдена.
    """
    bad: list[str] = []
    undecided: list[str] = []
    for word in iter_words(alphabet, max_len):
        verdict = pda.accepts(word)
        if verdict.value is None:
            undecided.append(word)
        elif verdict.value != predicate(word):
            bad.append(word)
            if len(bad) >= limit:
                break
    return bad, undecided
