"""Граф Кэли и действие группы на множестве.

По списку «что освежить к лекции 12 сентября» из README репозитория
преподавателя: действие группы на множестве и граф Кэли стоят там
пунктами 2 и 4. Действие определено и в конспекте лекции 1 (слайд 7,
в посылке леммы о пинг-понге); **графа Кэли в слайдах нет**, он назван
только в README.

Оба объекта здесь — про языки, а не про алгебру, и в этом смысл модуля.

## Граф Кэли это автомат

Возьмём копредставление, ориентируем соотношения и пополним
(`tfl/presentation.py`). Если пополнение сошлось, у каждого элемента
группы ровно одна нормальная форма, и граф Кэли строится обходом:
вершины — нормальные формы, из вершины `u` по букве `x` ведёт ребро
в нормальную форму слова `u·x`.

Это в точности **детерминированный конечный автомат**: начальное
состояние — пустое слово, переходы — умножение справа. Объявив
финальным одно начальное состояние, получаем распознаватель **проблемы
равенства**: автомат принимает слово тогда и только тогда, когда оно
равно единице.

## Что отсюда следует, и это проверяется исполнением

Вычет языка проблемы равенства по слову `u` — это множество `{v : uv = 1}`,
то есть множество слов, равных `u⁻¹`. Значит **два слова дают один вычет
ровно тогда, когда они равны в группе**, и число классов
Майхилла–Нероуда равно порядку группы.

Отсюда сразу лёгкая половина теоремы Анисимова: язык проблемы равенства
регулярен тогда и только тогда, когда группа конечна. Обход, закрывшийся
за `n` вершин, **доказывает**, что в группе ровно `n` элементов;
не закрывшийся — не доказывает ничего (пополнение могло не сойтись,
а группа быть бесконечной).

## Что именно обходится

Вершины — классы **пустого слова тоже**, поэтому построенное — граф Кэли
моноида `A*/≡_R`, а не полугруппы `A⁺/≡_R` из определения лекции.
Для копредставления группы это одно и то же (единица там и так есть),
а для полугруппы отличается на единицу, приписанную снаружи. Разница
названа явно, чтобы не выдавать одно за другое.

## Действие проверяется, а не строится

Действие задаётся образами образующих — перестановками точек, — и модуль
его **проверяет**: соотношения обязаны действовать тождественно. Польза
в том, что действие даёт вывод, недоступный переписыванию: если
образующая переставляет точки нетривиально, она **не равна единице**.
Так доказывают, что группа не тривиальна; `Presentation.is_trivial`
умеет только противоположное.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tfl.automata import DFA
from tfl.presentation import GROUP, Presentation
from tfl.srs import SRS
from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "Action",
    "CayleyGraph",
    "cayley_graph",
    "irreducible_automaton",
    "is_finite",
    "normalize",
]


def normalize(system: SRS, word: str) -> str:
    """Нормальная форма слова: переписывать, пока применимо хоть одно правило.

    Завершается потому, что все правила пополненной системы строго
    убывают в армейском порядке. Единственность даёт конфлюэнтность,
    и за неё отвечает вызывающий: у непополненной системы результат
    зависит от порядка применения.
    """
    changed = True
    while changed:
        changed = False
        for rule in system.rules:
            at = word.find(rule.lhs)
            if at >= 0:
                word = word[:at] + rule.rhs + word[at + len(rule.lhs) :]
                changed = True
                break
    return word


# --------------------------------------------------------------------------
# Действие группы на множестве
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """Действие: каждой образующей — перестановка точек `0 … points−1`.

    `images[x][p]` — куда переходит точка `p` под действием `x`. Обратной
    букве отвечает обратная перестановка, и задавать её не нужно.
    """

    points: int
    images: dict[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        for letter, image in self.images.items():
            if len(image) != self.points:
                raise ValueError(f"у образа «{letter}» не {self.points} точек")
            if sorted(image) != list(range(self.points)):
                raise ValueError(
                    f"образ «{letter}» не перестановка: {image}. "
                    f"Группа действует биекциями"
                )

    def of(self, letter: str) -> tuple[int, ...]:
        """Перестановка буквы; заглавной отвечает обратная."""
        if letter in self.images:
            return self.images[letter]
        lower = letter.lower()
        if lower not in self.images:
            raise KeyError(f"буква «{letter}» в действии не задана")
        forward = self.images[lower]
        back = [0] * self.points
        for source, target in enumerate(forward):
            back[target] = source
        return tuple(back)

    def apply(self, word: str, point: int) -> int:
        """Куда уходит точка под действием слова. Слово читается слева направо."""
        for letter in word:
            point = self.of(letter)[point]
        return point

    def orbit(self, point: int) -> frozenset[int]:
        """Орбита точки: куда её можно увести образующими и обратными."""
        seen = {point}
        queue = deque([point])
        while queue:
            current = queue.popleft()
            for letter in self.images:
                for image in (self.of(letter), self.of(letter.upper())):
                    nxt = image[current]
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        return frozenset(seen)

    def check(self, presentation: Presentation) -> Verdict:
        """Действительно ли это действие: выполняются ли все соотношения.

        Проверка полная — точек конечное число, соотношений конечное
        число, — поэтому оба исхода доказательны.
        """
        missing = set(presentation.generators) - set(self.images)
        if missing:
            return unknown(f"не заданы образы образующих: {''.join(sorted(missing))}")
        for left, right in presentation.relations:
            for point in range(self.points):
                if self.apply(left, point) != self.apply(right, point):
                    return refuted(
                        f"соотношение «{left or 'ε'} = {right or 'ε'}» нарушено "
                        f"на точке {point}: {self.apply(left, point)} против "
                        f"{self.apply(right, point)}",
                        (left, right, point),
                    )
        return proved(
            f"все {len(presentation.relations)} соотношений действуют "
            f"тождественно на {self.points} точках — это действие группы"
        )

    def separates(self, left: str, right: str) -> Verdict:
        """Различает ли действие два слова. Различило — они не равны в группе.

        Это единственный здесь способ доказать **неравенство** без полной
        системы переписывания: гомоморфный образ различает, значит
        различаются и прообразы.
        """
        for point in range(self.points):
            if self.apply(left, point) != self.apply(right, point):
                return refuted(
                    f"на точке {point} слово «{left or 'ε'}» уводит "
                    f"в {self.apply(left, point)}, а «{right or 'ε'}» — "
                    f"в {self.apply(right, point)}: в группе они не равны",
                    point,
                )
        return unknown(
            "действие не различает эти слова — про равенство отсюда "
            "не следует ничего, гомоморфизм мог их склеить"
        )


# --------------------------------------------------------------------------
# Граф Кэли
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CayleyGraph:
    """Граф Кэли конечной группы: вершины — нормальные формы элементов."""

    presentation: Presentation
    system: SRS
    elements: tuple[str, ...]
    edges: dict[tuple[str, str], str]

    @property
    def order(self) -> int:
        return len(self.elements)

    def multiply(self, left: str, right: str) -> str:
        """Произведение элементов, приведённое к нормальной форме."""
        return normalize(self.system, left + right)

    def dfa(self) -> DFA:
        """Автомат проблемы равенства: принимает слова, равные единице.

        Состояния — элементы, переход — умножение справа, финальное
        состояние одно: единица.

        Равенство «классов Майхилла–Нероуда ровно столько, сколько
        элементов» — утверждение **про группу**: там вычет по `u` это
        множество слов, равных `u⁻¹`, и он непуст всегда. У моноида без
        обратных вычет необратимого элемента может оказаться пустым,
        и совпадения не будет.
        """
        alphabet = frozenset(self.presentation.alphabet)
        return DFA(alphabet, "", frozenset({""}), dict(self.edges))

    def is_commutative(self) -> Verdict:
        """Коммутативно ли умножение. Ответ точный: таблица конечна."""
        for left in self.elements:
            for right in self.elements:
                if self.multiply(left, right) != self.multiply(right, left):
                    return refuted(
                        f"«{left or 'ε'}» и «{right or 'ε'}» не коммутируют: "
                        f"{self.multiply(left, right) or 'ε'} против "
                        f"{self.multiply(right, left) or 'ε'}",
                        (left, right),
                    )
        return proved(f"таблица умножения из {self.order} элементов симметрична")

    def to_dot(self) -> str:
        return self.dfa().to_dot("cayley")

    def markdown(self) -> str:
        lines = [
            f"Граф Кэли: **{self.order}** элементов, "
            f"нормальные формы `{'`, `'.join(e or 'ε' for e in self.elements)}`.",
            "",
            "| элемент | " + " | ".join(f"`{x}`" for x in self.presentation.alphabet) + " |",
            "|---" * (len(self.presentation.alphabet) + 1) + "|",
        ]
        for element in self.elements:
            row = " | ".join(
                f"`{self.edges[(element, letter)] or 'ε'}`"
                for letter in self.presentation.alphabet
            )
            lines.append(f"| `{element or 'ε'}` | {row} |")
        return "\n".join(lines)


def irreducible_automaton(system: SRS, alphabet: str) -> DFA:
    """Автомат неприводимых слов: тех, где нет ни одной левой части.

    Состояния — самый длинный суффикс прочитанного, являющийся
    **собственным** префиксом какой-нибудь левой части; это построение
    Ахо–Корасик. Перехода нет ровно тогда, когда очередная буква
    достраивает левую часть, то есть слово стало приводимым.

    Финальны все состояния: неприводимо любое слово, дошедшее до конца.
    """
    lefts = [rule.lhs for rule in system.rules]
    prefixes = {""} | {left[:i] for left in lefts for i in range(1, len(left))}
    delta: dict[tuple[str, str], str] = {}
    for state in prefixes:
        for letter in alphabet:
            grown = state + letter
            if any(grown.endswith(left) for left in lefts):
                continue  # слово стало приводимым — из языка выпадает
            while grown and grown not in prefixes:
                grown = grown[1:]
            delta[(state, letter)] = grown
    return DFA(frozenset(alphabet), "", frozenset(prefixes), delta)


def is_finite(presentation: Presentation) -> Verdict:
    """Конечен ли моноид копредставления. Два исхода из трёх доказательны.

    Ход рассуждения весь про языки. Пополнили систему — у каждого элемента
    ровно одна нормальная форма, то есть элементы моноида и неприводимые
    слова это одно и то же. Неприводимые слова образуют **регулярный**
    язык (в нём просто нет подслов-левых частей), а конечность
    регулярного языка разрешима: она равносильна отсутствию цикла
    в обрезанном автомате.

    Поэтому «бесконечен» здесь — доказательство, а не «обход не закрылся».
    Единственное «не выяснено» приходит от несошедшегося пополнения.
    """
    system, completed = presentation.complete()
    if completed.value is not True:
        return unknown(
            f"конечность не выяснена: {completed.reason}. Без полной системы "
            f"неприводимые слова и элементы моноида — разные множества"
        )
    machine = irreducible_automaton(system, presentation.alphabet)
    reachable = machine.reachable()
    colour: dict[str, int] = {}
    loop: list[str] | None = None

    def visit(state: str, path: list[str]) -> list[str] | None:
        colour[state] = 1
        for letter in sorted(presentation.alphabet):
            nxt = machine.delta.get((state, letter))
            if nxt is None or nxt not in reachable:
                continue
            if colour.get(nxt) == 1:
                return [*path, letter]
            if colour.get(nxt, 0) == 0:
                found = visit(nxt, [*path, letter])
                if found is not None:
                    return found
        colour[state] = 2
        return None

    loop = visit(machine.start, [])
    if loop is not None:
        return refuted(
            f"моноид бесконечен: неприводимых слов бесконечно много — "
            f"в автомате неприводимых слов есть цикл, первый выход на него "
            f"даёт слово «{''.join(loop)}». Значит и язык проблемы равенства "
            f"не регулярен (теорема Анисимова)",
            "".join(loop),
        )
    counted = _count_words(machine, presentation.alphabet)
    return proved(
        f"моноид конечен: неприводимых слов ровно {counted}, "
        f"а после пополнения они и есть элементы",
        counted,
    )


def _count_words(machine: DFA, alphabet: str) -> int:
    """Сколько слов принимает автомат без циклов. Считается по путям."""
    memo: dict[str, int] = {}

    def paths(state: str) -> int:
        if state not in memo:
            memo[state] = 1 + sum(
                paths(machine.delta[(state, letter)])
                for letter in sorted(alphabet)
                if (state, letter) in machine.delta
            )
        return memo[state]

    return paths(machine.start)


def cayley_graph(presentation: Presentation, max_elements: int = 500) -> Verdict:
    """Конечен ли моноид копредставления, и если да — его граф Кэли.

    Три исхода, и два из них доказательны.

    * **Да** — обход закрылся: элементов ровно столько, сколько вершин,
      и в свидетеле лежит `CayleyGraph` вместе с автоматом проблемы
      равенства.
    * **Нет** — обход упёрся в потолок, и тогда вопрос передаётся
      `is_finite`: у пополненной системы конечность **разрешима**,
      и бесконечность доказывается циклом в автомате неприводимых слов.
    * «Не выяснено» — пополнение не сошлось, либо элементов больше
      потолка, а бесконечности доказать не удалось.

    Условие корректности одно и оно проверяется: система обязана быть
    **пополненной**, иначе нормальная форма не единственна и вершины
    склеиваются как попало.
    """
    system, completed = presentation.complete()
    if completed.value is not True:
        return unknown(
            f"граф Кэли не строится: {completed.reason}. Без полной системы "
            f"нормальная форма не единственна, и вершины склеятся неправильно"
        )
    alphabet = presentation.alphabet
    order: list[str] = [""]
    seen = {""}
    edges: dict[tuple[str, str], str] = {}
    queue = deque([""])
    while queue:
        current = queue.popleft()
        for letter in alphabet:
            target = normalize(system, current + letter)
            edges[(current, letter)] = target
            if target not in seen:
                if len(seen) >= max_elements:
                    # Обход не закрылся — но конечность тут **разрешима**,
                    # и незачем гадать: спросим у автомата неприводимых слов.
                    verdict = is_finite(presentation)
                    if verdict.value is False:
                        return verdict
                    return unknown(
                        f"элементов больше {max_elements}: обход не закрылся, "
                        f"а бесконечности не доказано ({verdict.reason})"
                    )
                seen.add(target)
                order.append(target)
                queue.append(target)
    graph = CayleyGraph(presentation, system, tuple(order), edges)
    what = "группа" if presentation.kind == GROUP else "моноид A*/≡"
    return proved(
        f"{what} конечен: ровно {graph.order} элементов. Обход закрылся, "
        f"а система полна, поэтому у каждого элемента одна нормальная форма "
        f"и вершины графа Кэли — это и есть элементы",
        graph,
    )
