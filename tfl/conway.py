"""Переписывания регулярных выражений по тождествам Конвея.

Разбалловка РК1 2023 (issue #21) просит регулярку, «которую нельзя сделать
короче комбинациями переписываний по Конвею–Кробу». **Термина в лекциях
курса нет** — он встречается только в этом тексте разбалловки; поиск
проведён по правилу §2d (русское написание, латиница, фамилии). Здесь
реализовано то, что за ним стоит в литературе: классические тождества
регулярной алгебры Конвея (1971). Кроб (1991) доказал, что эта система
вместе с **групповыми** тождествами полна для эквивалентности регулярных
выражений. Полная система Кроба параметризована всеми конечными группами;
здесь есть её кусок — тождества для циклических групп малых порядков,
`(Rⁿ)*(ε|R|…|Rⁿ⁻¹) = R*`. Именно их и не хватает классической системе:
оракул показывает, что `(aa)*|a(aa)*` одними классическими тождествами
к `a*` не сводится даже при обходе всех выражений размера до 22,
а с тождеством для группы порядка 2 сводится сразу.

Тождества применяются **в обе стороны**: цель — не нормальная форма,
а поиск более короткой записи, и по дороге выражение часто приходится
удлинить. Поэтому обход ограничен и по размеру, и по числу узлов,
а вывод односторонний: «короче не нашлось» — не «короче не существует».

Часть аксиом Конвея выполняется в проекте **даром**: умные конструкторы
`tfl.regex.alt/cat/star` сортируют и склеивают альтернативы, разворачивают
ассоциативность, сокращают `∅r = ∅`, `εr = r`, `r** = r*`, `ε* = ε`,
`∅* = ε`. Поэтому в списке ниже их нет: переписывать по ним нечего,
выражения такого вида просто не существуют как объекты.

Каждый шаг переписывания сверяется с оракулом эквивалентности
(`tfl.automata`): если тождество записано неверно, это видно сразу
на первом же выражении, а не превращается в тихо неправильный ответ.
"""

from __future__ import annotations

from dataclasses import dataclass

from tfl import regex as rx
from tfl.trs import Rule, Term, app, match, substitute, var
from tfl.verdict import Verdict, refuted, unknown

__all__ = [
    "IDENTITIES",
    "CYCLIC_ORDERS",
    "cyclic_collapse",
    "cyclic_expansions",
    "to_term",
    "from_term",
    "rewritings",
    "shorten",
    "is_minimal",
    "Shortening",
]

ALT, CAT, STAR = "⊕", "⊙", "✳"
EPS, EMPTY = "ε", "∅"

R, S, T = var("R"), var("S"), var("T")


def _alt(*args: Term) -> Term:
    return app(ALT, *args)


def _cat(*args: Term) -> Term:
    return app(CAT, *args)


def _star(arg: Term) -> Term:
    return app(STAR, arg)


def _fold(head: str, parts: list[Term]) -> Term:
    """Свернуть список в двуместное дерево вправо.

    Тождества записаны двуместными, а `Alt` и `Cat` в `tfl.regex`
    многоместные; правая свёртка их и связывает. Ассоциативность при этом
    не теряется: обратный перевод отдаёт список конструктору, который
    его снова расплющит.
    """
    if not parts:
        return app(EPS if head == CAT else EMPTY)
    result = parts[-1]
    for part in reversed(parts[:-1]):
        result = app(head, part, result)
    return result


#: Классические тождества Конвея, кроме тех, что уже встроены
#: в конструкторы `tfl.regex`. Читаются как равенства и применяются
#: в обе стороны.
IDENTITIES: tuple[tuple[str, Term, Term], ...] = (
    ("дистрибутивность слева", _cat(R, _alt(S, T)), _alt(_cat(R, S), _cat(R, T))),
    ("дистрибутивность справа", _cat(_alt(R, S), T), _alt(_cat(R, T), _cat(S, T))),
    ("разворот звёздочки слева", _star(R), _alt(app(EPS), _cat(R, _star(R)))),
    ("разворот звёздочки справа", _star(R), _alt(app(EPS), _cat(_star(R), R))),
    ("звёздочка суммы", _star(_alt(R, S)), _cat(_star(_cat(_star(R), S)), _star(R))),
    (
        "звёздочка произведения",
        _star(_cat(R, S)),
        _alt(app(EPS), _cat(R, _star(_cat(S, R)), S)),
    ),
    ("склейка звёздочек", _cat(_star(R), _star(R)), _star(R)),
    ("звёздочка с единицей", _star(_alt(app(EPS), R)), _star(R)),
)


#: Порядки циклических групп, для которых проект знает групповое тождество.
CYCLIC_ORDERS = (2, 3, 4)


def cyclic_collapse(node: rx.Node) -> rx.Node | None:
    """Свернуть `(Rⁿ)* | R(Rⁿ)* | … | Rⁿ⁻¹(Rⁿ)*` в `R*`, если это оно.

    Это групповое тождество для циклической группы порядка `n`: итерация
    разбита по остатку длины. Классической системе Конвея таких тождеств
    не хватает — ради них Кроб и добавил групповые аксиомы.

    Распознаётся **структурно**, а не сопоставлением с образцом, и вот
    почему. Тождество естественно записывается как `(Rⁿ)*·(ε|R|…) = R*`,
    но чтобы дойти до этой формы, надо сперва вынести общий множитель
    из альтернативы — а первое слагаемое пришлось бы для этого записать
    как `ε·(Rⁿ)*`. Конструктор `tfl.regex.cat` такой объект не создаёт:
    `εr = r` у него встроено. Промежуточная форма не существует, путь
    через дистрибутивность обрывается, и остаётся узнавать образец целиком.

    Вторая причина та же по духу: `Alt` хранит альтернативы **множеством**
    (отсортированным и без дублей), поэтому порядок слагаемых ничего
    не значит, и сравнивать надо множества, а не деревья.
    """
    if not isinstance(node, rx.Alt):
        return None
    parts = list(node.args)
    order = len(parts)
    if order not in CYCLIC_ORDERS:
        return None
    for candidate in parts:
        if not isinstance(candidate, rx.Star):
            continue
        core = candidate.arg
        factors = list(core.args) if isinstance(core, rx.Cat) else [core]
        if len(factors) != order or len(set(factors)) != 1:
            continue
        base = factors[0]
        tail = {rx.cat(*([base] * k), candidate) for k in range(order)}
        head = {rx.cat(candidate, *([base] * k)) for k in range(order)}
        if set(parts) in (tail, head):
            return rx.star(base)
    return None


def cyclic_expansions(node: rx.Node) -> list[rx.Node]:
    """Обратный ход: `R*` → `(Rⁿ)* | R(Rⁿ)* | …` для малых `n`.

    Удлиняет запись, но иногда только через неё и виден путь к более
    короткой — поэтому в обход входит наравне с остальными шагами.
    """
    if not isinstance(node, rx.Star):
        return []
    base = node.arg
    found = []
    for order in CYCLIC_ORDERS:
        power = rx.star(rx.cat(*([base] * order)))
        found.append(rx.alt(*(rx.cat(*([base] * k), power) for k in range(order))))
    return [other for other in found if other != node]


# --------------------------------------------------------------------------
# Перевод между AST регулярки и термом
# --------------------------------------------------------------------------


def to_term(node: rx.Node) -> Term:
    """Регулярное выражение как терм: `⊕` альтернатива, `⊙` конкатенация."""
    if isinstance(node, rx.Empty):
        return app(EMPTY)
    if isinstance(node, rx.Eps):
        return app(EPS)
    if isinstance(node, rx.Sym):
        return app(node.char)
    if isinstance(node, rx.Alt):
        return _fold(ALT, [to_term(a) for a in node.args])
    if isinstance(node, rx.Cat):
        return _fold(CAT, [to_term(a) for a in node.args])
    if isinstance(node, rx.Star):
        return _star(to_term(node.arg))
    raise TypeError(f"неизвестный узел регулярки: {node!r}")


def from_term(term: Term) -> rx.Node:
    """Обратный перевод. Нормализацию делают конструкторы `tfl.regex`."""
    if term.variable:
        raise ValueError(f"в терме осталась переменная «{term.head}»")
    if term.head == EMPTY and not term.args:
        return rx.EMPTY
    if term.head == EPS and not term.args:
        return rx.EPS
    if term.head == ALT:
        return rx.alt(*(from_term(a) for a in term.args))
    if term.head == CAT:
        return rx.cat(*(from_term(a) for a in term.args))
    if term.head == STAR:
        return rx.star(from_term(term.args[0]))
    if not term.args:
        return rx.Sym(term.head)
    raise ValueError(f"неизвестный символ «{term.head}» в терме")


# --------------------------------------------------------------------------
# Переписывание
# --------------------------------------------------------------------------


def _rules() -> list[tuple[str, Rule]]:
    """Каждое тождество — два правила: слева направо и справа налево."""
    both: list[tuple[str, Rule]] = []
    for name, left, right in IDENTITIES:
        both.append((name, Rule(left, right)))
        both.append((f"{name} (обратно)", Rule(right, left)))
    return both


def rewritings(node: rx.Node, groups: bool = True) -> list[tuple[str, rx.Node]]:
    """Все выражения, получаемые одним применением тождества в любом месте.

    Возвращает пары «название тождества, результат». Дубликаты снимаются:
    конструкторы `tfl.regex` нормализуют, и разные шаги часто приходят
    в одно и то же выражение.
    """
    source = to_term(node)
    seen: dict[rx.Node, str] = {}
    for name, rule in _rules():
        for path, sub in source.positions():
            binding = match(rule.lhs, sub)
            if binding is None:
                continue
            grown = source.replace(path, substitute(rule.rhs, binding))
            try:
                result = from_term(grown)
            except ValueError:
                continue
            if result != node and result not in seen:
                seen[result] = name

    if groups:
        for path, sub in _positions(node):
            collapsed = cyclic_collapse(sub)
            variants = [] if collapsed is None else [collapsed]
            variants += cyclic_expansions(sub)
            for variant in variants:
                result = _replace(node, path, variant)
                if result != node and result not in seen:
                    order = len(sub.args) if collapsed is not None and isinstance(
                        sub, rx.Alt
                    ) else len(variant.args) if isinstance(variant, rx.Alt) else 0
                    seen[result] = f"циклическая группа порядка {order}"
    return [(name, node) for node, name in seen.items()]


def _positions(node: rx.Node):
    """Пары «путь, подвыражение» — путь это индексы аргументов."""
    yield (), node
    for index, child in enumerate(_children(node)):
        for path, sub in _positions(child):
            yield (index, *path), sub


def _children(node: rx.Node) -> tuple[rx.Node, ...]:
    if isinstance(node, (rx.Alt, rx.Cat)):
        return node.args
    if isinstance(node, rx.Star):
        return (node.arg,)
    return ()


def _replace(node: rx.Node, path: tuple[int, ...], replacement: rx.Node) -> rx.Node:
    if not path:
        return replacement
    head, *rest = path
    children = list(_children(node))
    children[head] = _replace(children[head], tuple(rest), replacement)
    if isinstance(node, rx.Alt):
        return rx.alt(*children)
    if isinstance(node, rx.Cat):
        return rx.cat(*children)
    return rx.star(children[0])


def _weight(node: rx.Node) -> tuple[int, int, str]:
    """Мера «короче»: сперва число узлов, потом длина записи."""
    return (node.size(), len(str(node)), str(node))


@dataclass(frozen=True)
class Shortening:
    """Найденное сокращение: путь от исходной регулярки к более короткой."""

    start: rx.Node
    best: rx.Node
    steps: tuple[tuple[str, rx.Node], ...]
    visited: int
    exhausted: bool

    def markdown(self) -> str:
        lines = [f"`{self.start}` → `{self.best}`", ""]
        current = self.start
        for name, node in self.steps:
            lines.append(f"- `{current}` → `{node}` — {name}")
            current = node
        return "\n".join(lines)


def shorten(
    pattern: str | rx.Node,
    max_size: int = 14,
    max_nodes: int = 4000,
    alphabet: str | None = None,
    groups: bool = True,
) -> Shortening:
    """Искать более короткую эквивалентную запись переписываниями Конвея.

    Обход в ширину по выражениям, полученным применением тождеств в обе
    стороны. Ограничения нужны оба: без границы на размер обход уходит
    в бесконечность (тождества можно применять «на удлинение» сколько
    угодно), без границы на число узлов — считает слишком долго.

    Каждое встреченное выражение сверяется с исходным по эквивалентности
    автоматов. Это не перестраховка: тождества записаны руками, и ошибка
    в любом из них иначе прошла бы незамеченной.
    """
    start = rx.parse(pattern) if isinstance(pattern, str) else pattern
    letters = alphabet or _letters(start)
    reference = _dfa(start, letters)

    best = start
    parents: dict[rx.Node, tuple[rx.Node, str] | None] = {start: None}
    frontier = [start]
    visited = 1
    exhausted = True
    while frontier:
        following: list[rx.Node] = []
        for node in frontier:
            for name, other in rewritings(node, groups):
                if other in parents or other.size() > max_size:
                    continue
                if visited >= max_nodes:
                    exhausted = False
                    break
                from tfl.automata import equivalent

                if not equivalent(reference, _dfa(other, letters)):
                    raise AssertionError(
                        f"тождество «{name}» изменило язык: «{node}» → «{other}». "
                        f"Это ошибка в списке тождеств, а не в выражении"
                    )
                visited += 1
                parents[other] = (node, name)
                following.append(other)
                if _weight(other) < _weight(best):
                    best = other
            if not exhausted:
                break
        if not exhausted:
            break
        frontier = following

    steps: list[tuple[str, rx.Node]] = []
    cursor = best
    while parents.get(cursor):
        previous, name = parents[cursor]
        steps.append((name, cursor))
        cursor = previous
    return Shortening(start, best, tuple(reversed(steps)), visited, exhausted)


def is_minimal(
    pattern: str | rx.Node,
    max_size: int = 14,
    max_nodes: int = 4000,
    alphabet: str | None = None,
    groups: bool = True,
) -> Verdict:
    """Нельзя ли сделать регулярку короче переписываниями Конвея.

    Исходов два, и `True` среди них нет намеренно. Обход конечен по
    построению, а система тождеств — нет: из «в пределах обхода короче
    не нашлось» минимальность не следует. В отчёт идёт именно эта
    формулировка, с указанием границ обхода.
    """
    found = shorten(pattern, max_size, max_nodes, alphabet, groups)
    if _weight(found.best) < _weight(found.start):
        return refuted(
            f"короче: «{found.best}» ({found.best.size()} узлов против "
            f"{found.start.size()}). Вывод:\n{found.markdown()}",
            found,
        )
    limits = (
        f"обход пройден целиком: {found.visited} выражений размера ≤ {max_size}"
        if found.exhausted
        else f"обход обрезан на {max_nodes} выражениях"
    )
    return unknown(
        f"переписываниями Конвея сократить «{found.start}» не удалось "
        f"({limits}). Минимальностью это не является: система тождеств "
        f"бесконечна, а обход конечен",
        found,
    )


def _letters(node: rx.Node) -> str:
    found: set[str] = set()

    def walk(current: rx.Node) -> None:
        if isinstance(current, rx.Sym):
            found.add(current.char)
        for attribute in ("args", "arg"):
            value = getattr(current, attribute, None)
            if isinstance(value, tuple):
                for item in value:
                    walk(item)
            elif isinstance(value, rx.Node):
                walk(value)

    walk(node)
    return "".join(sorted(found)) or "a"


def _dfa(node: rx.Node, letters: str):
    from tfl.automata import dfa_of

    return dfa_of(node, letters)
