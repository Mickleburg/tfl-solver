"""Недетерминированный КС-разбор: графовидный стек и упакованный лес.

ЛР5 2023 (`lab_tfl_2023_5.pdf`). Задание — Generic-разбор слова по
**произвольной** грамматике, где таблица может содержать конфликты,
и все пути разбора отслеживаются одновременно. Со слайда 1:

> Элементарный недетерминизм: при каждом конфликте создаётся новая копия
> стека. Древовидный стек: нижняя часть стека остаётся неизменной,
> создаются только новые вершины. Лесовидный стек («древовидный стек
> наоборот»): если несколько стеков отслеживают одну и ту же позицию
> в слове и имеют одну и ту же вершину, они объединяются в общий лист.
> Графовидный стек объединяет свойства древовидного и лесовидного.

Отсюда два режима: `GRAPH` — вершины с одинаковыми «состояние + позиция»
сливаются (это и есть стек Томиты), `TREE` — не сливаются, каждый путь
разбора ведёт свой стек. `TREE` в задании назван «соло-версией» и стоит
10 базовых баллов вместо 8; разница между режимами **меряется**
(`ParseResult.nodes`), а не описывается словами.

Бонус на +6 баллов — «запакованный лес разбора (Shared Packed Parse
Forest)» — здесь строится всегда: без него неоткуда взять ответ
на вопрос «сколько разборов у слова», а он и есть смысл недетерминизма.
Узел леса — тройка `(символ, начало, конец)`; **упаковка** — это
несколько семейств потомков у одного узла, и ровно она означает
неоднозначность.

## Две тонкости, стоившие бы баллов

**Опечатка в самом условии.** На слайде 3 написано «действие — перенос
(reduce) или свёртка (shift)». Английские глоссы переставлены: перенос
это shift, свёртка это reduce. Считать шаги надо по действиям, а как
их называть — видно по таблице.

**Разные определения шага в разных вариантах.** Для Generic SLR(1)
(варианты 6, 7, 8) «`n`-ый шаг — состояние стека после `n` действий…,
а не состояние стека после чтения `n` букв»; для LR(0) (варианты 0, 1,
3, 9) — ровно наоборот, «после чтения `n` букв». Поэтому снимок стека
берётся по обоим счётчикам: `ParseResult.snapshot_by_action`
и `snapshot_by_letter`.

## Про полноту фазы свёрток

Классический приём (RNGLR Скотта и Джонстона) запоминает у отложенной
свёртки **первое ребро** пути и перезапускает свёртки только у той
вершины, из которой новое ребро выходит. Здесь фаза свёрток вместо этого
доводится **до неподвижной точки**: пока за проход появляется хоть одна
новая вершина, ребро или семейство в лесу, проход повторяется. Так
медленнее, зато полнота очевидна, а она здесь и есть предмет проверки —
лес сверяется с независимым перебором деревьев вывода.

ε-правила: условие ЛР явно говорит «произвольная КС-грамматика
**без ε-правил**», и они не поддерживаются — свёртка нулевой длины
требует отдельной конструкции (right-nulled GLR). При ε-правиле
поднимается `ValueError`, а не тихо считается неправильно.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tfl.cfg import END, EPSILON, CFG, LRItem, Production
from tfl.conj import ConjunctiveGrammar

__all__ = [
    "LR0",
    "SLR1",
    "GRAPH",
    "TREE",
    "SHIFT",
    "REDUCE",
    "ACCEPT",
    "Action",
    "actions_table",
    "SPPF",
    "GSSNode",
    "Snapshot",
    "ParseResult",
    "parse",
    "parse_ll",
    "parse_conj",
    "relaxed_cfg",
    "LL1",
    "BOTTOM",
]

#: Какая таблица строится: свёртка по любому символу или только по Follow.
LR0 = "LR(0)"
SLR1 = "SLR(1)"

#: Как устроен стек: сливать одинаковые вершины или нет.
GRAPH = "графовидный"
TREE = "древовидный"

SHIFT = "перенос"
REDUCE = "свёртка"
ACCEPT = "допуск"


# --------------------------------------------------------------------------
# Таблица действий
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """Одно действие таблицы. Конфликты не разрешаются, а сохраняются."""

    kind: str
    state: int = -1
    production: Production | None = None

    def __str__(self) -> str:
        if self.kind == SHIFT:
            return f"перенос → {self.state}"
        if self.kind == REDUCE:
            return f"свёртка {self.production}"
        return "допуск"


def actions_table(
    grammar: CFG, mode: str = SLR1
) -> tuple[list[frozenset[LRItem]], dict[tuple[int, str], int], dict[tuple[int, str], list[Action]]]:
    """Состояния LR(0), переходы и **все** действия, включая конфликтующие.

    Именно это отличает Generic-разбор от обычного: детерминированный
    парсер на конфликте останавливается, а здесь конфликт — это ветвление.
    """
    if mode not in (LR0, SLR1):
        raise ValueError(f"режим должен быть «{LR0}» либо «{SLR1}»")
    for production in grammar.productions:
        if production.is_epsilon:
            raise ValueError(
                f"правило «{production}» пустое: условие ЛР5 требует грамматику "
                "без ε-правил, а свёртка нулевой длины требует другой конструкции"
            )

    states, transitions = grammar.lr0_states()
    augmented, extra = grammar.augmented()
    follow = grammar.follow_k(1)
    lookaheads = sorted(grammar.terminals) + [END]

    table: dict[tuple[int, str], list[Action]] = defaultdict(list)
    for index, state in enumerate(states):
        for item in state:
            if item.at_end:
                if item.production == extra:
                    table[(index, END)].append(Action(ACCEPT))
                    continue
                if mode == LR0:
                    allowed = lookaheads
                else:
                    allowed = sorted(
                        token[0] if token else END
                        for token in follow.get(item.production.lhs, ())
                    )
                for token in allowed:
                    table[(index, token)].append(Action(REDUCE, production=item.production))
            elif item.next_symbol in grammar.terminals:
                target = transitions.get((index, item.next_symbol))
                if target is not None:
                    action = Action(SHIFT, state=target)
                    if action not in table[(index, item.next_symbol)]:
                        table[(index, item.next_symbol)].append(action)
    return states, transitions, dict(table)


# --------------------------------------------------------------------------
# Упакованный лес разбора
# --------------------------------------------------------------------------

#: Узел леса: символ и границы куска слова, который он выводит.
Label = tuple[str, int, int]

#: Семейство потомков: правило и метки детей.
Family = tuple[Production | None, tuple[Label, ...]]


@dataclass
class SPPF:
    """Shared Packed Parse Forest — бонус задания на +6 баллов.

    Общий подлес не дублируется (один узел на тройку «символ, начало,
    конец»), а неоднозначность **упакована**: у узла несколько семейств
    потомков. Число разборов слова считается по лесу, а не перебором.
    """

    families: dict[Label, set[Family]] = field(default_factory=lambda: defaultdict(set))

    #: Узлы бинаризации нисходящего разбора: у дерева их быть не должно,
    #: поэтому при выдаче деревьев их потомки вклеиваются в родителя.
    intermediate: set[Label] = field(default_factory=set)

    def add(self, label: Label, production: Production | None, children: tuple[Label, ...]) -> bool:
        """Добавить семейство. `True`, если оно новое."""
        before = len(self.families[label])
        self.families[label].add((production, children))
        return len(self.families[label]) != before

    def ambiguous(self) -> list[Label]:
        """Узлы с несколькими семействами — места неоднозначности."""
        return sorted(label for label, group in self.families.items() if len(group) > 1)

    def count(self, label: Label, seen: frozenset[Label] = frozenset()) -> int:
        """Число деревьев под узлом.

        Считается по лесу, а не перебором деревьев, поэтому числа
        Каталана достаются из линейного по размеру графа. Общий подлес
        считается один раз — без этого счёт был бы экспоненциальным
        на том же самом лесу.

        Цикл в лесу означал бы, что разборов бесконечно много (`A ⇒⁺ A`).
        В наших грамматиках его быть не может — при восходящем разборе
        ε-правила запрещены условием, при нисходящем запрещена левая
        рекурсия, — но если он появится, лучше сказать об этом, чем
        уйти в рекурсию.
        """
        memo: dict[Label, int] = {}

        def go(node: Label, path: frozenset[Label]) -> int:
            if node in path:
                raise ValueError(f"в лесу цикл через «{node[0]}»: разборов бесконечно много")
            if node in memo:
                return memo[node]
            group = self.families.get(node)
            if not group:
                return 1  # лист
            total = 0
            for _, children in group:
                product = 1
                for child in children:
                    product *= go(child, path | {node})
                total += product
            memo[node] = total
            return total

        return go(label, frozenset(seen))

    def _branches(self, label: Label, limit: int) -> list[tuple]:
        """Наборы поддеревьев, которые узел вносит в родителя.

        Обычный узел вносит ровно одно поддерево; узел бинаризации —
        целый набор, и его потомки вклеиваются в родителя. Так дерево
        получается таким, каким его рисуют по грамматике, а лес внутри
        остаётся бинарным.
        """
        if label not in self.intermediate:
            return [(tree,) for tree in self.trees(label, limit)]
        found: list[tuple] = []
        for _, children in sorted(self.families.get(label, ()), key=repr):
            combinations: list[tuple] = [()]
            for child in children:
                combinations = [
                    prefix + piece
                    for prefix in combinations
                    for piece in self._branches(child, limit)
                ][:limit]
            found.extend(combinations)
        return found[:limit]

    def trees(self, label: Label, limit: int = 200) -> list[tuple]:
        """Деревья под узлом как вложенные кортежи `(символ, потомки…)`."""
        group = self.families.get(label)
        if not group:
            return [(label[0],)]
        found: list[tuple] = []
        for _, children in sorted(group, key=repr):
            combinations: list[tuple] = [()]
            for child in children:
                combinations = [
                    prefix + piece
                    for prefix in combinations
                    for piece in self._branches(child, limit)
                ][:limit]
            for combination in combinations:
                found.append((label[0], *combination))
                if len(found) >= limit:
                    return found
        return found

    def yield_of(self, label: Label) -> str:
        """Слово, выводимое узлом — для сверки границ."""
        group = self.families.get(label)
        if not group:
            return "" if label[0] == EPSILON else label[0]
        _, children = next(iter(group))
        if children and all(child[1:] == label[1:] for child in children):
            # Конъюнкция: потомки покрывают **один и тот же** кусок слова,
            # а не разные его части, поэтому склеивать их нельзя.
            return self.yield_of(children[0])
        return "".join(self.yield_of(child) for child in children)

    def to_dot(self, name: str = "SPPF") -> str:
        lines = [f'digraph "{name}" {{', "  rankdir=TB;", '  node [shape=box];']
        packed = 0
        for label, group in sorted(self.families.items(), key=repr):
            head = f'"{label[0]}, {label[1]}..{label[2]}"'
            if label in self.intermediate:
                lines.append(f"  {head} [style=dashed];")
            for _, children in sorted(group, key=repr):
                if len(group) > 1:
                    packed += 1
                    middle = f'"пакет {packed}"'
                    lines.append(f"  {middle} [shape=circle,label=\"\"];")
                    lines.append(f"  {head} -> {middle};")
                    source = middle
                else:
                    source = head
                for child in children:
                    tail = f'"{child[0]}, {child[1]}..{child[2]}"'
                    lines.append(f"  {source} -> {tail};")
        lines.append("}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Графовидный стек
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GSSNode:
    """Вершина гиперстека.

    > Метки листьев гиперстека — пары из отслеживаемой текущей позиции
    > в слове и вершины (классического) стека разбора.

    `tag` различает вершины в древовидном режиме: там одинаковые пары
    не сливаются, и это ровно разница между соло-версией и полной.
    """

    state: int | str
    level: int
    tag: int = 0

    def __str__(self) -> str:
        base = f"{self.state}@{self.level}"
        return base if self.tag == 0 else f"{base}#{self.tag}"


@dataclass(frozen=True)
class Snapshot:
    """Состояние гиперстека после очередного действия."""

    actions: int
    letters: int
    kind: str
    detail: str
    nodes: tuple[GSSNode, ...]
    edges: tuple[tuple[GSSNode, GSSNode, Label], ...]

    def to_dot(self, name: str = "GSS") -> str:
        lines = [f'digraph "{name}" {{', "  rankdir=RL;", "  node [shape=circle];"]
        for node in self.nodes:
            lines.append(f'  "{node}";')
        for source, target, label in self.edges:
            lines.append(f'  "{source}" -> "{target}" [label="{label[0]}"];')
        lines.append("}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ParseResult:
    """Итог разбора: допуск или первая ошибочная позиция."""

    accepted: bool
    error_position: int | None
    forest: SPPF
    root: Label | None
    snapshots: tuple[Snapshot, ...]
    nodes: int
    edges: int
    mode: str
    sharing: str

    @property
    def parses(self) -> int:
        """Число разборов слова. `0`, если разбора нет."""
        return self.forest.count(self.root) if self.root else 0

    def snapshot_by_action(self, step: int) -> Snapshot | None:
        """Стек после `step` действий — счёт вариантов 6, 7, 8."""
        for snapshot in self.snapshots:
            if snapshot.actions == step:
                return snapshot
        return None

    def snapshot_by_letter(self, step: int) -> Snapshot | None:
        """Стек после чтения `step` букв — счёт вариантов 0, 1, 3, 9."""
        found = [s for s in self.snapshots if s.letters == step]
        return found[-1] if found else None

    def summary(self) -> str:
        if self.accepted:
            head = f"разбор успешен, разборов {self.parses}"
        else:
            head = f"разбора нет, первая ошибочная позиция {self.error_position}"
        return (
            f"{head}; стек {self.sharing}, таблица {self.mode}: "
            f"вершин {self.nodes}, рёбер {self.edges}, "
            f"действий {self.snapshots[-1].actions if self.snapshots else 0}"
        )


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def parse(
    grammar: CFG,
    word: str,
    mode: str = SLR1,
    sharing: str = GRAPH,
    tokens: list[str] | None = None,
    max_actions: int = 100_000,
) -> ParseResult:
    """Generic LR-разбор слова с гиперстеком.

    `mode` — какая таблица (`LR0` или `SLR1`), `sharing` — сливать ли
    одинаковые вершины (`GRAPH`) или вести отдельный стек на каждый путь
    (`TREE`, соло-версия задания).

    При неуспехе возвращается **первая позиция, в которой не остаётся
    ни одного пути разбора** — это и требует условие.
    """
    if sharing not in (GRAPH, TREE):
        raise ValueError(f"стек должен быть «{GRAPH}» либо «{TREE}»")
    _, transitions, table = actions_table(grammar, mode)

    stream = list(tokens) if tokens is not None else list(word)
    stream.append(END)
    forest = SPPF()

    counter = [0]
    branches: dict[tuple[int, int, GSSNode], GSSNode] = {}

    def make(state: int, level: int, below: GSSNode) -> GSSNode:
        """Вершина стека над `below`.

        В графовидном режиме вершина определяется парой «состояние +
        позиция», и разные пути в неё сливаются — у вершины появляется
        несколько рёбер вниз. В древовидном нижняя часть по-прежнему
        общая, но вершины не сливаются: каждая помнит, из-под чего
        выросла, и ребро вниз у неё ровно одно.
        """
        if sharing == GRAPH:
            return GSSNode(state, level)
        key = (state, level, below)
        if key not in branches:
            counter[0] += 1
            branches[key] = GSSNode(state, level, counter[0])
        return branches[key]

    levels: list[set[GSSNode]] = [set() for _ in range(len(stream) + 1)]
    edges: dict[GSSNode, dict[GSSNode, Label]] = defaultdict(dict)
    start = GSSNode(0, 0)
    levels[0].add(start)

    snapshots: list[Snapshot] = []
    performed = 0

    def all_edges() -> tuple[tuple[GSSNode, GSSNode, Label], ...]:
        return tuple(
            (source, target, label)
            for source, group in sorted(edges.items(), key=repr)
            for target, label in sorted(group.items(), key=repr)
        )

    def record(kind: str, detail: str, letters: int, level: int) -> None:
        nonlocal performed
        performed += 1
        snapshots.append(
            Snapshot(
                performed,
                letters,
                kind,
                detail,
                tuple(sorted(levels[level], key=repr)),
                all_edges(),
            )
        )

    def paths(node: GSSNode, length: int):
        """Все пути длины `length` вниз от вершины: конец и метки рёбер.

        Метки возвращаются **слева направо**, в порядке символов правой
        части: обход идёт сверху вниз, то есть от последнего символа
        к первому, и метка верхнего ребра дописывается в конец уже
        собранного нижнего куска.
        """
        if length == 0:
            yield node, ()
            return
        for target, label in list(edges[node].items()):
            for end, rest in paths(target, length - 1):
                yield end, rest + (label,)

    accepted = False
    root: Label | None = None
    error_position: int | None = None

    for position, token in enumerate(stream):
        # --- фаза свёрток до неподвижной точки --------------------------
        changed = True
        while changed and performed < max_actions:
            changed = False
            for node in sorted(levels[position], key=repr):
                for action in table.get((node.state, token), ()):
                    if action.kind == ACCEPT and position == len(stream) - 1:
                        accepted = True
                        root = (grammar.start, 0, len(stream) - 1)
                        continue
                    if action.kind != REDUCE:
                        continue
                    production = action.production
                    for end, labels in paths(node, len(production.rhs)):
                        label: Label = (production.lhs, end.level, position)
                        fresh = forest.add(label, production, labels)
                        target_state = transitions.get((end.state, production.lhs))
                        if target_state is None:
                            continue
                        top = make(target_state, position, end)
                        added = top not in levels[position] or end not in edges[top]
                        if added:
                            levels[position].add(top)
                            edges[top][end] = label
                        if added or fresh:
                            changed = True
                            record(
                                REDUCE,
                                f"{production} на позиции {position}",
                                position,
                                position,
                            )

        # --- фаза переноса ----------------------------------------------
        if position == len(stream) - 1:
            break
        moved = False
        for node in sorted(levels[position], key=repr):
            for action in table.get((node.state, token), ()):
                if action.kind != SHIFT:
                    continue
                label = (token, position, position + 1)
                top = make(action.state, position + 1, node)
                levels[position + 1].add(top)
                edges[top][node] = label
                moved = True
                record(SHIFT, f"«{token}» в состояние {action.state}", position + 1, position + 1)
        if not moved or not levels[position + 1]:
            error_position = position
            break

    if not accepted and error_position is None:
        error_position = len(stream) - 1
    if accepted:
        error_position = None

    return ParseResult(
        accepted=accepted,
        error_position=error_position,
        forest=forest,
        root=root if accepted else None,
        snapshots=tuple(snapshots),
        nodes=sum(len(group) for group in levels),
        edges=sum(len(group) for group in edges.values()),
        mode=mode,
        sharing=sharing,
    )


# --------------------------------------------------------------------------
# Generic LL(1) — варианты 2, 4, 5
# --------------------------------------------------------------------------

#: Третья таблица задания: нисходящий разбор по LL(1).
LL1 = "LL(1)"

#: Дно гиперстека в нисходящем разборе.
BOTTOM = "⊥"




def parse_ll(
    grammar: CFG,
    word: str,
    sharing: str = GRAPH,
    tokens: list[str] | None = None,
    max_actions: int = 100_000,
) -> ParseResult:
    """Generic LL(1)-разбор слова с гиперстеком.

    > Реализовать Generic LL(1)-разбор слова `w` по грамматике `G`
    > с использованием графовидного стека. Входные данные: грамматика `G`
    > (произвольная КС-грамматика **без левой рекурсии**), слово `w`…
    > `n`-ый шаг здесь — состояние стека после `n` действий (действие —
    > **применение правила из таблицы**), а не состояние стека после
    > чтения `n` букв.

    Гиперстек тот же, но вершина помечена **не символом**, а точкой
    возврата: пунктом вида `A → α B • β` вместе с позицией, на которой
    в `B` вошли. Разница не косметическая, и на ней легко потерять всё.

    Если помечать вершину просто символом и позицией, то нетерминал
    `S`, лежащий в стеке на разной глубине, склеивается в одну вершину.
    На грамматике `S → aSb | ab` со словом `abb` из этого немедленно
    вырастает петля `b@0 → b@0`, стек начинает порождать любое число
    `b`, и разбор принимает слово, которого в языке нет. Проверка
    против Эрли это ловит сразу — потому она в тестах и стоит.

    Конфликт в ячейке `(A, a)` не разрешается, а ветвится: применяются
    все правила ячейки. Снятие вершины и добавление ребра к уже снятой
    вершине обрабатываются множеством `popped`, как в алгоритме GLL, —
    иначе теряются пути, открывшиеся позже.

    Левая рекурсия запрещена условием, и проверяется явно: с ней разбор
    не зациклится (вершины конечны), но и правильного ответа не даст.

    ## Лес разбора: бинаризация

    Бонус на +6 баллов здесь строится тоже, но лес получается
    **бинаризованным** — иначе при разборе сверху вниз его не собрать.
    Причина простая: нисходящий разбор дочитывает правую часть слева
    направо и в момент прочтения `i`-го символа ещё не знает, где
    кончится `(i+1)`-ый. Поэтому вместо узла на всё правило заводится
    цепочка узлов на **пункты** `A → x₁…xᵢ • xᵢ₊₁…`, каждый ровно
    с двумя потомками: «что уже разобрано» и «что разобрано сейчас».
    Это конструкция Скотта и Джонстона для GLL.

    Два узла, которых при восходящем разборе не бывает:

    * `A → α • β` при `|α| ⩾ 2` — узел бинаризации; в дереве его быть
      не должно, поэтому `SPPF.trees` вклеивает его потомков в родителя;
    * `(ε, i, i)` — лист для пустой правой части. При восходящем разборе
      ε-правила запрещены условием, при нисходящем разрешены.

    Узел на один символ не заводится вовсе: пункт `A → x • β` при
    непустом `β` отдаёт наверх узел самого `x`. Без этой оговорки лес
    распухает вдвое, а число разборов не меняется.

    Счёт разборов от бинаризации не страдает: узел бинаризации имеет
    столько семейств, сколько способов разрезать разобранный кусок,
    и произведение по потомкам даёт то же число, что перебор деревьев.
    Это и проверяется в тестах — против независимого перебора.
    """
    if sharing not in (GRAPH, TREE):
        raise ValueError(f"стек должен быть «{GRAPH}» либо «{TREE}»")
    recursive = grammar.left_recursive()
    if recursive:
        raise ValueError(
            "условие ЛР5 требует грамматику без левой рекурсии, "
            f"а левая рекурсия есть у {', '.join(sorted(recursive))}"
        )

    table, _ = grammar.ll1_table()
    stream = list(tokens) if tokens is not None else list(word)
    length = len(stream)

    counter = [0]
    branches: dict[tuple[object, int, GSSNode], GSSNode] = {}
    bottom = GSSNode(BOTTOM, 0)
    forest = SPPF()

    def make(slot: LRItem, level: int, below: GSSNode) -> GSSNode:
        if sharing == GRAPH:
            return GSSNode(slot, level)
        key = (slot, level, below)
        if key not in branches:
            counter[0] += 1
            branches[key] = GSSNode(slot, level, counter[0])
        return branches[key]

    def joined(slot: LRItem, left: Label | None, right: Label) -> Label:
        """Узел леса для пункта `slot` из разобранного `left` и нового `right`.

        `slot` — пункт уже **после** очередного символа. Если до него
        разобран ровно один символ и правило не кончилось, узел не нужен:
        наверх идёт сам `right`. Иначе заводится узел — на всё правило
        (`A`, если точка в конце) либо промежуточный (сам пункт).
        """
        if slot.dot == 1 and not slot.at_end:
            return right
        symbol = slot.production.lhs if slot.at_end else str(slot)
        start = right[1] if left is None else left[1]
        label: Label = (symbol, start, right[2])
        forest.add(label, slot.production, (right,) if left is None else (left, right))
        if not slot.at_end:
            forest.intermediate.add(label)
        return label

    edges: dict[GSSNode, dict[GSSNode, Label | None]] = defaultdict(dict)
    popped: set[tuple[GSSNode, int, Label]] = set()
    descriptors: set[tuple[LRItem, GSSNode, int, Label | None]] = set()
    pending: list[tuple[LRItem, GSSNode, int, Label | None]] = []

    snapshots: list[Snapshot] = []
    performed = 0
    accepted = False
    reached = 0
    root: Label | None = None

    def all_edges() -> tuple[tuple[GSSNode, GSSNode, Label], ...]:
        return tuple(
            (source, target, label or (str(source.state), source.level, source.level))
            for source, group in sorted(edges.items(), key=repr)
            for target, label in sorted(group.items(), key=repr)
        )

    def record(detail: str, position: int) -> None:
        """Действие в этом варианте — применение правила из таблицы."""
        nonlocal performed
        performed += 1
        snapshots.append(
            Snapshot(
                performed,
                position,
                REDUCE,
                detail,
                tuple(sorted({item[1] for item in descriptors} | {bottom}, key=repr)),
                all_edges(),
            )
        )

    def add(slot: LRItem, node: GSSNode, position: int, made: Label | None) -> None:
        key = (slot, node, position, made)
        if key not in descriptors:
            descriptors.add(key)
            pending.append(key)

    for production in grammar.rules_for(grammar.start):
        add(LRItem(production, 0), bottom, 0, None)

    while pending and performed < max_actions:
        slot, node, position, left = pending.pop()
        reached = max(reached, position)

        if slot.at_end:
            done = left if left is not None else joined(slot, None, (EPSILON, position, position))
            popped.add((node, position, done))
            if (
                node == bottom
                and position == length
                and slot.production.lhs == grammar.start
            ):
                accepted = True
                root = done
            if node != bottom:
                for below, carried in list(edges[node].items()):
                    add(node.state, below, position, joined(node.state, carried, done))
            continue

        symbol = slot.next_symbol
        if symbol not in grammar.nonterminals:
            if position < length and stream[position] == symbol:
                leaf: Label = (symbol, position, position + 1)
                ahead = slot.advance()
                add(ahead, node, position + 1, joined(ahead, left, leaf))
                reached = max(reached, position + 1)
            continue

        lookahead = stream[position] if position < length else END
        chosen = table.get((symbol, lookahead), ())
        if not chosen:
            continue
        top = make(slot.advance(), position, node)
        fresh = node not in edges[top]
        edges[top][node] = left
        if fresh:
            for known, step, made in list(popped):
                if known == top:
                    add(top.state, node, step, joined(top.state, left, made))
        for production in chosen:
            add(LRItem(production, 0), top, position, None)
            record(f"{production} на позиции {position}", position)

    return ParseResult(
        accepted=accepted,
        error_position=None if accepted else min(reached, length),
        forest=forest,
        root=root if accepted else None,
        snapshots=tuple(snapshots),
        nodes=len(set(edges) | {bottom}),
        edges=sum(len(group) for group in edges.values()),
        mode=LL1,
        sharing=sharing,
    )


# --------------------------------------------------------------------------
# Конъюнктивные грамматики гиперстеком — бонус +4
# --------------------------------------------------------------------------


def relaxed_cfg(
    grammar: ConjunctiveGrammar,
) -> tuple[CFG, dict[Production, tuple[tuple[int, int], ...]]]:
    """Дизъюнктивное послабление: каждый конъюнкт становится отдельным правилом.

    `A → Φ₁ & Φ₂` превращается в `A → Φ₁ | Φ₂`. Язык при этом только
    растёт (`&` заменён на `|`), поэтому таблица LR послабления годится
    для конъюнктивного разбора: она разрешает **не меньше** свёрток,
    чем нужно, а лишние отсекает сама конъюнкция.

    Возвращается ещё и обратное отображение «правило послабления → все
    пары (номер конъюнктивного правила, номер конъюнкта)». Оно не
    однозначно нарочно: у `A → B & C | B` конъюнкт `B` и отдельное
    правило `B` дают одно и то же правило послабления, и свёртка по нему
    засчитывается сразу обоим.
    """
    productions: list[Production] = []
    origin: dict[Production, list[tuple[int, int]]] = {}
    for index, rule in enumerate(grammar.rules):
        for number, conjunct in enumerate(rule.conjuncts):
            production = Production(rule.lhs, tuple(conjunct))
            if production not in origin:
                origin[production] = []
                productions.append(production)
            origin[production].append((index, number))
    relaxed = CFG(
        grammar.start,
        tuple(productions),
        frozenset(grammar.nonterminals),
        frozenset(grammar.terminals),
    )
    return relaxed, {p: tuple(pairs) for p, pairs in origin.items()}


def parse_conj(
    grammar: ConjunctiveGrammar,
    word: str,
    mode: str = SLR1,
    tokens: list[str] | None = None,
    max_actions: int = 100_000,
) -> ParseResult:
    """Generic-разбор по **конъюнктивной** грамматике — бонус на +4 балла.

    > Добавить возможность разбора конъюнктивных грамматик (+4 балла).
    > Актуально… только если реализуется графовидный стек (не древовидный).

    Идея в одну фразу: **свёртка конъюнкта вершину не кладёт**. Кладёт
    её правило целиком, и только когда все его конъюнкты собраны над
    одной и той же вершиной гиперстека и до одной и той же позиции.

    Почему этого достаточно. Разбор идёт по таблице послабления
    (`relaxed_cfg`), в котором `&` заменено на `|`. Всякий конъюнкт
    всякого применения конъюнктивного правила встречается в каком-нибудь
    полном разборе слова по послаблению — достаточно всюду выбирать
    первый конъюнкт, а в интересующем месте нужный, — а Generic-разбор
    перебирает все такие разборы. Значит ни одна нужная свёртка
    не потеряется.

    Почему это не даёт лишнего. Узел леса появляется у `A` над отрезком
    только тогда, когда над этим отрезком собраны **все** конъюнкты,
    а каждый из них собран из настоящих разборов подслов. Это ровно
    определение конъюнктивного вывода.

    Почему конъюнкты сходятся к **одной** вершине. Все конъюнкты одного
    правила предсказываются одним и тем же замыканием: пункты
    `A → • Φⱼ` лежат в одном состоянии LR, поэтому все они начинаются
    в одной вершине уровня `j`, и путь свёртки любого из них приводит
    обратно в неё же. В графовидном стеке вершины с одинаковыми
    «состояние + позиция» слиты, значит вершина буквально одна и та же —
    и в древовидном режиме приём не работает, что задание и оговаривает.

    Дерево вывода здесь имеет вид, каким его и рисуют для конъюнктивных
    грамматик: у узла `A` столько поддеревьев, сколько конъюнктов,
    и все они выводят **одно и то же** слово, а не разные его части.
    """
    relaxed, origin = relaxed_cfg(grammar)
    _, transitions, table = actions_table(relaxed, mode)

    stream = list(tokens) if tokens is not None else list(word)
    stream.append(END)
    forest = SPPF()

    levels: list[set[GSSNode]] = [set() for _ in range(len(stream) + 1)]
    edges: dict[GSSNode, dict[GSSNode, Label]] = defaultdict(dict)
    start = GSSNode(0, 0)
    levels[0].add(start)

    #: Какие конъюнкты правила уже собраны: (правило, вершина, конец) → номер → узел.
    parts: dict[tuple[int, GSSNode, int], dict[int, Label]] = defaultdict(dict)

    snapshots: list[Snapshot] = []
    performed = 0

    def all_edges() -> tuple[tuple[GSSNode, GSSNode, Label], ...]:
        return tuple(
            (source, target, label)
            for source, group in sorted(edges.items(), key=repr)
            for target, label in sorted(group.items(), key=repr)
        )

    def record(kind: str, detail: str, letters: int, level: int) -> None:
        nonlocal performed
        performed += 1
        snapshots.append(
            Snapshot(
                performed,
                letters,
                kind,
                detail,
                tuple(sorted(levels[level], key=repr)),
                all_edges(),
            )
        )

    def paths(node: GSSNode, length: int):
        if length == 0:
            yield node, ()
            return
        for target, label in list(edges[node].items()):
            for end, rest in paths(target, length - 1):
                yield end, rest + (label,)

    def push(end: GSSNode, symbol: str, label: Label, position: int) -> bool:
        """Положить вершину по переходу `goto(end, symbol)`. `True`, если новая."""
        target = transitions.get((end.state, symbol))
        if target is None:
            return False
        top = GSSNode(target, position)
        if top in levels[position] and end in edges[top]:
            return False
        levels[position].add(top)
        edges[top][end] = label
        return True

    accepted = False
    root: Label | None = None
    error_position: int | None = None

    for position, token in enumerate(stream):
        changed = True
        while changed and performed < max_actions:
            changed = False
            for node in sorted(levels[position], key=repr):
                for action in table.get((node.state, token), ()):
                    if action.kind == ACCEPT and position == len(stream) - 1:
                        accepted = True
                        root = (grammar.start, 0, len(stream) - 1)
                        continue
                    if action.kind != REDUCE:
                        continue
                    production = action.production
                    for end, labels in paths(node, len(production.rhs)):
                        children = labels
                        for index, number in origin[production]:
                            rule = grammar.rules[index]
                            if rule.is_plain:
                                label = (rule.lhs, end.level, position)
                                fresh = forest.add(label, production, children)
                                if push(end, rule.lhs, label, position) or fresh:
                                    changed = True
                                    record(
                                        REDUCE,
                                        f"{production} на позиции {position}",
                                        position,
                                        position,
                                    )
                                continue
                            body = " ".join(rule.conjuncts[number])
                            piece = (f"{rule.lhs}⟨{body}⟩", end.level, position)
                            fresh = forest.add(piece, production, children)
                            collected = parts[(index, end, position)]
                            if number not in collected or fresh:
                                collected[number] = piece
                                changed = True
                                record(
                                    REDUCE,
                                    f"конъюнкт {number + 1} правила «{rule}» "
                                    f"на позиции {position}",
                                    position,
                                    position,
                                )
                            if len(collected) < len(rule.conjuncts):
                                continue
                            whole = tuple(
                                collected[j] for j in range(len(rule.conjuncts))
                            )
                            label = (rule.lhs, end.level, position)
                            fresh = forest.add(label, None, whole)
                            if push(end, rule.lhs, label, position) or fresh:
                                changed = True
                                record(
                                    REDUCE,
                                    f"«{rule}» целиком на позиции {position}",
                                    position,
                                    position,
                                )

        if position == len(stream) - 1:
            break
        moved = False
        for node in sorted(levels[position], key=repr):
            for action in table.get((node.state, token), ()):
                if action.kind != SHIFT:
                    continue
                label = (token, position, position + 1)
                top = GSSNode(action.state, position + 1)
                levels[position + 1].add(top)
                edges[top][node] = label
                moved = True
                record(
                    SHIFT,
                    f"«{token}» в состояние {action.state}",
                    position + 1,
                    position + 1,
                )
        if not moved or not levels[position + 1]:
            error_position = position
            break

    if not accepted and error_position is None:
        error_position = len(stream) - 1
    if accepted:
        error_position = None

    return ParseResult(
        accepted=accepted,
        error_position=error_position,
        forest=forest,
        root=root if accepted else None,
        snapshots=tuple(snapshots),
        nodes=sum(len(group) for group in levels),
        edges=sum(len(group) for group in edges.values()),
        mode=f"{mode}, конъюнктивная",
        sharing=GRAPH,
    )
