"""Brainfuck-Рефал и Brainfuck-Лисп: сторона МАТа в ЛР2 2024.

Варианты 6, 8 (Рефал) и 2, 5, 7 (Лисп) отличаются от лабиринтных тем,
что автомат-цель не рисуется руками, а **собирается из автоматов лексем
по грамматике**. Задание требует от МАТа ровно трёх вещей:

> Генератор случайных входных данных (согласно ограничениям ТЗ);
> минимизация и каноническая нумерация автомата;
> визуализация — автоматы для лексем и общий автомат для лексера.

Здесь есть все три, плюс то, чего в задании нет, но без чего работа
не защищается: **исполняемая проверка ограничений**. Ограничения ТЗ
сформулированы через языки («конкатенация любых двух таких автоматов
имеет язык, не пересекающийся ни с каким языком автомата для одной
скобки»), а не через устройство генератора, поэтому и проверяются
операциями над автоматами, а не рассуждением о том, как генератор
устроен. Генератор выдаёт одно семейство, удовлетворяющее ТЗ;
проверка (`check`) годится для любого набора лексем, в том числе
написанного руками.

## Почему цель регулярна

Грамматики обоих языков контекстно-свободные: скобки вложены. Но
угадыватель «заранее знает максимальную вложенность скобок», а при
ограниченной вложенности рекурсия разворачивается в конечную
подстановку, и язык становится регулярным. Именно поэтому `L*` здесь
вообще применим. Вложенность считается **отдельно по каждому типу
скобок** — у Рефала их три, и счётчик у каждого свой.

## Что собирается

    [program]    ::= [eol]*([definition][eol]+)+
    [definition] ::= [const][lbr-1]([eol]*[sentence])*[eol]*[rbr-1]
    [sentence]   ::= [pattern][equal][expression][sep]
    [pattern]    ::= [lbr-3][pattern][rbr-3] | [pattern][blank][pattern]
                   | [var] | [const] | ε
    [expression] ::= [var] | [const] | [expression][blank][expression]
                   | [lbr-3][expression][rbr-3]
                   | [lbr-2][const][blank][expression][rbr-2]

Правило `[pattern][blank][pattern]` (и такое же для `[expression]`)
раскрывается в `atom ([blank] atom)*`: язык от этого не меняется,
а автомат получается вдвое меньше. Так можно потому, что алфавит
`[blank]` по ТЗ не пересекается ни с чьим другим.

Промежуточные автоматы минимизируются на каждом уровне вложенности.
Замер на Рефале: та же цель на вложенности 2 собирается за 0.2 с
со свёрткой и за 7.3 с без неё — автомат получается один и тот же,
время отличается в 36 раз.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from tfl.automata import DFA, NFA, intersection
from tfl.verdict import Verdict, proved, refuted

__all__ = [
    "Lexicon",
    "REFAL",
    "LISP",
    "REFAL_LEXEMES",
    "LISP_LEXEMES",
    "random_lexicon",
    "words_automaton",
    "concat",
    "alternative",
    "repeat",
    "plus",
    "optional",
    "epsilon",
    "refal_automaton",
    "lisp_automaton",
    "automaton_for",
    "sample_program",
    "teacher_for",
]

#: Два варианта задания и их алфавиты.
REFAL = "рефал"
LISP = "лисп"

REFAL_ALPHABET = frozenset("abc012")
LISP_ALPHABET = frozenset("0123456789")

#: Лексемы, для которых грамматика определения не даёт: их задают автоматы.
REFAL_LEXEMES = (
    "eol",
    "blank",
    "equal",
    "sep",
    "lbr-1",
    "rbr-1",
    "lbr-2",
    "rbr-2",
    "lbr-3",
    "rbr-3",
    "var",
    "const",
)
LISP_LEXEMES = ("eol", "lbr", "rbr", "dot", "atom")

_REFAL_BRACKETS = ("lbr-1", "rbr-1", "lbr-2", "rbr-2", "lbr-3", "rbr-3")
_LISP_BRACKETS = ("lbr", "rbr")


# --------------------------------------------------------------------------
# Сборка автоматов
# --------------------------------------------------------------------------


def _as_nfa(machine: DFA) -> NFA:
    """ДКА как НКА — чтобы его можно было склеивать с другими."""
    delta: dict[tuple[object, str], frozenset] = {
        key: frozenset({value}) for key, value in machine.delta.items()
    }
    return NFA(machine.alphabet, machine.start, machine.finals, delta, {})


def _rename(machine: NFA, mark: int) -> NFA:
    """Развести состояния по меткам, чтобы склейка ничего не смешала."""
    delta = {
        ((mark, src), char): frozenset((mark, dst) for dst in targets)
        for (src, char), targets in machine.delta.items()
    }
    eps = {
        (mark, src): frozenset((mark, dst) for dst in targets)
        for src, targets in machine.eps.items()
    }
    return NFA(
        machine.alphabet,
        (mark, machine.start),
        frozenset((mark, state) for state in machine.finals),
        delta,
        eps,
    )


def _join(pieces: list[NFA]) -> tuple[dict, dict, frozenset[str]]:
    delta: dict = {}
    eps: dict = {}
    alphabet: set[str] = set()
    for piece in pieces:
        delta.update(piece.delta)
        for src, targets in piece.eps.items():
            eps[src] = eps.get(src, frozenset()) | targets
        alphabet |= set(piece.alphabet)
    return delta, eps, frozenset(alphabet)


def concat(left: NFA, right: NFA) -> NFA:
    first, second = _rename(left, 0), _rename(right, 1)
    delta, eps, alphabet = _join([first, second])
    for final in first.finals:
        eps[final] = eps.get(final, frozenset()) | {second.start}
    return NFA(alphabet, first.start, second.finals, delta, eps)


def alternative(*machines: NFA) -> NFA:
    if not machines:
        raise ValueError("альтернатива без вариантов — это пустой язык")
    parts = [_rename(machine, index) for index, machine in enumerate(machines)]
    delta, eps, alphabet = _join(parts)
    start = "⊤"
    eps[start] = frozenset(part.start for part in parts)
    finals = frozenset(state for part in parts for state in part.finals)
    return NFA(alphabet, start, finals, delta, eps)


def repeat(machine: NFA) -> NFA:
    """Звёздочка Клини."""
    inner = _rename(machine, 0)
    delta, eps, alphabet = _join([inner])
    start = "⊤"
    eps[start] = frozenset({inner.start})
    for final in inner.finals:
        eps[final] = eps.get(final, frozenset()) | {start}
    return NFA(alphabet, start, frozenset({start}), delta, eps)


def plus(machine: NFA) -> NFA:
    return concat(machine, repeat(machine))


def optional(machine: NFA) -> NFA:
    return alternative(machine, epsilon(machine.alphabet))


def epsilon(alphabet) -> NFA:
    return NFA(frozenset(alphabet), "ε", frozenset({"ε"}), {}, {})


def words_automaton(words, alphabet=None) -> DFA:
    """Минимальный ДКА по конечному списку слов — так задаются лексемы."""
    words = list(words)
    letters = frozenset(alphabet or {char for word in words for char in word})
    delta: dict[tuple[str, str], str] = {}
    finals = set()
    for word in words:
        for index in range(len(word)):
            delta[(word[:index], word[index])] = word[: index + 1]
        finals.add(word)
    return DFA(letters, "", frozenset(finals), delta).minimize()


def _compact(machine: NFA) -> NFA:
    """Свернуть промежуточный автомат: детерминизация плюс минимизация.

    Результат от свёртки не зависит — зависит время: на вложенности 2
    сборка идёт 0.2 с со свёрткой и 7.3 с без неё. Измерено, а не
    предположено.
    """
    return _as_nfa(machine.determinize().minimize())


# --------------------------------------------------------------------------
# Набор лексем
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Lexicon:
    """Автоматы лексем одного варианта задания."""

    kind: str
    machines: dict[str, DFA] = field(default_factory=dict)

    @property
    def alphabet(self) -> frozenset[str]:
        return REFAL_ALPHABET if self.kind == REFAL else LISP_ALPHABET

    @property
    def names(self) -> tuple[str, ...]:
        return REFAL_LEXEMES if self.kind == REFAL else LISP_LEXEMES

    def __getitem__(self, name: str) -> DFA:
        return self.machines[name]

    def nfa(self, name: str) -> NFA:
        return _as_nfa(self.machines[name])

    def words(self, name: str, max_len: int = 6) -> list[str]:
        from tfl.words import iter_words

        machine = self.machines[name]
        letters = "".join(sorted(letters_of(machine)))
        return [w for w in iter_words(letters, max_len) if machine.accepts(w)]

    def markdown(self, max_len: int = 5) -> str:
        lines = ["| лексема | букв | состояний | короткие слова |", "|---|---|---|---|"]
        for name in self.names:
            machine = self.machines[name]
            sample = self.words(name, max_len)[:6]
            shown = ", ".join(f"`{word or 'ε'}`" for word in sample) or "—"
            lines.append(
                f"| `[{name}]` | `{''.join(sorted(letters_of(machine)))}` "
                f"| {len(machine)} | {shown} |"
            )
        return "\n".join(lines)

    def to_dot(self) -> str:
        return "\n\n".join(
            self.machines[name].to_dot(name) for name in self.names
        )

    def check(self) -> Verdict:
        return check_refal(self) if self.kind == REFAL else check_lisp(self)


# --------------------------------------------------------------------------
# Свойства языков — то, через что сформулированы ограничения ТЗ
# --------------------------------------------------------------------------


def letters_of(machine: DFA) -> frozenset[str]:
    """Буквы, реально встречающиеся в словах языка."""
    live = machine.trim()
    return frozenset(char for (_, char) in live.delta)


def is_finite(machine: DFA) -> bool:
    """Конечен ли язык: нет цикла среди живых состояний."""
    live = machine.trim()
    graph: dict[object, set[object]] = {}
    for (src, _), dst in live.delta.items():
        graph.setdefault(src, set()).add(dst)
    colour: dict[object, int] = {}

    def cyclic(node) -> bool:
        colour[node] = 1
        for nxt in graph.get(node, ()):
            if colour.get(nxt) == 1:
                return True
            if colour.get(nxt) is None and cyclic(nxt):
                return True
        colour[node] = 2
        return False

    return not any(
        colour.get(node) is None and cyclic(node) for node in list(graph) or []
    )


def border_letters(machine: DFA) -> tuple[frozenset[str], frozenset[str]]:
    """Первые и последние буквы слов языка."""
    live = machine.trim()
    first = frozenset(
        char for (src, char) in live.delta if src == live.start
    )
    last = frozenset(
        char for (src, char), dst in live.delta.items() if dst in live.finals
    )
    return first, last


def _empty_intersection(left: DFA, right: DFA) -> bool:
    return intersection(left.complete(), right.complete()).is_empty()


def _concatenation(left: DFA, right: DFA) -> DFA:
    return concat(_as_nfa(left), _as_nfa(right)).determinize().minimize()


def _bracket_rules(lexicon: Lexicon, brackets, others) -> str | None:
    """Общая для обоих вариантов проверка скобочных ограничений."""
    for index, name in enumerate(brackets):
        for other in brackets[index + 1 :]:
            if not _empty_intersection(lexicon[name], lexicon[other]):
                return f"языки `[{name}]` и `[{other}]` пересекаются"
        for other in others:
            if not _empty_intersection(lexicon[name], lexicon[other]):
                return f"языки `[{name}]` и `[{other}]` пересекаются"
    for left in brackets:
        for right in brackets:
            glued = _concatenation(lexicon[left], lexicon[right])
            for single in brackets:
                if not _empty_intersection(glued, lexicon[single]):
                    return (
                        f"конкатенация `[{left}][{right}]` пересекается "
                        f"с одиночной скобкой `[{single}]`"
                    )
    return None


def check_refal(lexicon: Lexicon) -> Verdict:
    """Проверить ограничения ТЗ для Brainfuck-Рефала — по слайду 11."""
    missing = [name for name in REFAL_LEXEMES if name not in lexicon.machines]
    if missing:
        return refuted(f"нет автоматов для лексем: {', '.join(missing)}")

    used = {name: letters_of(lexicon[name]) for name in REFAL_LEXEMES}
    for special in ("eol", "blank"):
        rest = [name for name in REFAL_LEXEMES if name != special]
        clash = [name for name in rest if used[special] & used[name]]
        if clash:
            return refuted(
                f"алфавит `[{special}]` пересекается с `[{clash[0]}]`, "
                "а по ТЗ обязан отличаться от всех остальных"
            )

    ordinary = [
        name for name in REFAL_LEXEMES if name not in ("eol", "blank", "equal", "sep")
    ]
    for special in ("equal", "sep"):
        if not is_finite(lexicon[special]):
            return refuted(f"язык `[{special}]` бесконечен, а по ТЗ обязан быть конечным")
        first, last = border_letters(lexicon[special])
        clash = [name for name in ordinary if (first | last) & used[name]]
        if clash:
            return refuted(
                f"крайние буквы `[{special}]` встречаются и в `[{clash[0]}]`, "
                "а по ТЗ обязаны лежать в другом алфавите"
            )

    trouble = _bracket_rules(lexicon, _REFAL_BRACKETS, ("var", "const"))
    if trouble:
        return refuted(trouble)

    if not _empty_intersection(lexicon["var"], lexicon["const"]):
        return refuted("языки `[var]` и `[const]` пересекаются")
    for name in ("var", "const"):
        if is_finite(lexicon[name]):
            return refuted(f"язык `[{name}]` конечен, а по ТЗ обязан быть бесконечным")

    return proved(
        "все ограничения ТЗ Brainfuck-Рефала выполнены: алфавиты `[eol]` "
        "и `[blank]` обособлены, языки `[equal]` и `[sep]` конечны и окаймлены "
        "своими буквами, скобки попарно не пересекаются и не склеиваются "
        "в одиночную, `[var]` и `[const]` бесконечны и не пересекаются"
    )


def check_lisp(lexicon: Lexicon) -> Verdict:
    """Проверить ограничения ТЗ для Brainfuck-Лиспа — по слайду 14."""
    missing = [name for name in LISP_LEXEMES if name not in lexicon.machines]
    if missing:
        return refuted(f"нет автоматов для лексем: {', '.join(missing)}")

    used = {name: letters_of(lexicon[name]) for name in LISP_LEXEMES}
    clash = [name for name in LISP_LEXEMES[1:] if used["eol"] & used[name]]
    if clash:
        return refuted(
            f"алфавит `[eol]` пересекается с `[{clash[0]}]`, "
            "а по ТЗ обязан отличаться от всех остальных"
        )

    trouble = _bracket_rules(lexicon, _LISP_BRACKETS, ("atom",))
    if trouble:
        return refuted(trouble)

    if not _empty_intersection(lexicon["dot"], lexicon["atom"]):
        return refuted("языки `[dot]` и `[atom]` пересекаются")
    if is_finite(lexicon["atom"]):
        return refuted("язык `[atom]` конечен, а по ТЗ обязан быть бесконечным")

    return proved(
        "все ограничения ТЗ Brainfuck-Лиспа выполнены: алфавит `[eol]` "
        "обособлен, скобки попарно не пересекаются, не пересекаются "
        "с атомами и не склеиваются в одиночную, `[dot]` от атомов отделён, "
        "язык атомов бесконечен"
    )


# --------------------------------------------------------------------------
# Генератор
# --------------------------------------------------------------------------


#: Роли букв в каждом варианте. Разделение не косметическое, см. ниже.
ROLES = {
    REFAL: {
        "eol": "0",
        "blank": "1",
        "sentinel": "2",
        "brackets": "ab",
        "head": "c",
        "body": "abc",
        "bracket_len": 4,
    },
    LISP: {
        "eol": "0",
        "blank": "",
        "sentinel": "",
        "brackets": "12",
        "head": "3456789",
        "body": "3456789",
        "bracket_len": 3,
    },
}


def _delimited_words(border: str, mark: str, body: str) -> DFA:
    """Имя: `border`, метка `mark`, потом что угодно из `body`, потом `border`.

    Буква-ограничитель внутри слова не встречается, поэтому имя читается
    однозначно, а метка отличает `[var]` от `[const]`. Язык бесконечен,
    и с языками скобок (в них ограничителя нет вовсе) не пересекается.
    """
    letters = frozenset(body) | {border, mark}
    delta: dict[tuple[object, str], object] = {("q0", border): "q1", ("q1", mark): "q2"}
    for char in sorted(body):
        delta[("q2", char)] = "q2"
    delta[("q2", border)] = "q3"
    return DFA(letters, "q0", frozenset({"q3"}), delta).minimize()


def _prefixed_words(heads: str, body: str) -> DFA:
    """Слова, начинающиеся с любой буквы из `heads`. Язык бесконечен."""
    letters = frozenset(body) | set(heads)
    delta: dict[tuple[object, str], object] = {}
    for char in sorted(heads):
        delta[("q0", char)] = "q1"
    for char in sorted(letters):
        delta[("q1", char)] = "q1"
    return DFA(letters, "q0", frozenset({"q1"}), delta).minimize()


def random_lexicon(
    kind: str = REFAL,
    seed: int = 0,
    size: int = 4,
    bracket_len: int | None = None,
) -> Lexicon:
    """Случайный набор лексем, удовлетворяющий ТЗ.

    Ограничения ТЗ проверяются `check`, но одного их выполнения мало:
    набор, формально законный, может дать **вырожденную** цель. Первая
    версия генератора отдавала `[var]` как «любое достаточно длинное
    слово, начинающееся с буквы `a`», — все ограничения выполнялись,
    а конструкция `[lbr-2][const][blank][expression][rbr-2]` целиком
    разбиралась как `[var][blank][var]` и не добавляла к языку ничего.
    Автомат при этом честно схлопывался с 16 состояний до 5.

    Отсюда разделение ролей букв (`ROLES`):

    * `[eol]` и `[blank]` — каждый в своей однобуквенной части алфавита,
      как и требует ТЗ;
    * слова `[equal]` и `[sep]` окаймлены буквой `2`, которой больше
      нигде нет; языки конечны;
    * **скобки живут в своём подалфавите** (`ab` у Рефала, `12` у Лиспа)
      и состоят из слов одной длины. Одинаковая длина даёт условие ТЗ
      про склейку даром: два слова вдвое длиннее одного;
    * **имена начинаются с буквы, которой у скобок нет** (`c` у Рефала,
      цифра от 3 у Лиспа). Поэтому скобочная конструкция никогда
      не читается как одно имя, и вырождения не происходит;
    * `[var]` и `[const]` различаются последней буквой — этого хватает,
      чтобы языки не пересекались и оба были бесконечны.
    """
    if kind not in (REFAL, LISP):
        raise ValueError(f"вариант должен быть «{REFAL}» либо «{LISP}»")
    roles = ROLES[kind]
    width = bracket_len or roles["bracket_len"]
    rng = random.Random(seed)
    machines: dict[str, DFA] = {}

    def counted(letter: str) -> DFA:
        """Непустой конечный язык из повторов одной буквы."""
        lengths = sorted(rng.sample([1, 2, 3], rng.randint(1, 2)))
        return words_automaton([letter * length for length in lengths], {letter})

    machines["eol"] = counted(roles["eol"])
    if kind == REFAL:
        machines["blank"] = counted(roles["blank"])

    brackets = _REFAL_BRACKETS if kind == REFAL else _LISP_BRACKETS
    pool = [
        "".join(letters)
        for letters in itertools.product(roles["brackets"], repeat=width)
    ]
    rng.shuffle(pool)
    per = max(1, min(size - 1, len(pool) // (len(brackets) + 1)))
    for index, name in enumerate(brackets):
        machines[name] = words_automaton(
            pool[index * per : (index + 1) * per], set(roles["brackets"])
        )

    if kind == REFAL:
        sentinel = roles["sentinel"]
        middles = [""] + list(roles["body"])
        used: list[str] = []
        for name in ("equal", "sep"):
            free = [middle for middle in middles if middle not in used]
            chosen = sorted(rng.sample(free, rng.randint(1, min(2, len(free)))))
            used.extend(chosen)
            machines[name] = words_automaton(
                [sentinel + middle + sentinel for middle in chosen],
                set(roles["body"]) | {sentinel},
            )
        marks = list(roles["brackets"])
        rng.shuffle(marks)
        machines["var"] = _delimited_words(
            roles["head"], marks[0], roles["brackets"]
        )
        machines["const"] = _delimited_words(
            roles["head"], marks[1], roles["brackets"]
        )
    else:
        machines["dot"] = words_automaton(
            [pool[-1] + roles["brackets"][0]], set(roles["brackets"])
        )
        machines["atom"] = _prefixed_words(roles["head"], roles["body"])

    return Lexicon(kind, machines)


# --------------------------------------------------------------------------
# Сборка целевого автомата
# --------------------------------------------------------------------------


def refal_automaton(lexicon: Lexicon, depth: int = 1) -> DFA:
    """Автомат лексического анализа Brainfuck-Рефала при вложенности `depth`.

    Счётчики вложенности у трёх типов скобок **свои**, как и сказано
    в задании: вход в `[lbr-3]` уменьшает только третий.
    """
    if lexicon.kind != REFAL:
        raise ValueError("набор лексем не от Brainfuck-Рефала")
    if depth < 0:
        raise ValueError("вложенность не бывает отрицательной")
    get = lexicon.nfa

    def pattern(third: int) -> NFA:
        parts = [epsilon(lexicon.alphabet), get("var"), get("const")]
        if third > 0:
            parts.append(concat(concat(get("lbr-3"), pattern(third - 1)), get("rbr-3")))
        atom = _compact(alternative(*parts))
        return _compact(concat(atom, repeat(concat(get("blank"), atom))))

    def expression(second: int, third: int) -> NFA:
        parts = [get("var"), get("const")]
        if third > 0:
            parts.append(
                concat(concat(get("lbr-3"), expression(second, third - 1)), get("rbr-3"))
            )
        if second > 0:
            parts.append(
                concat(
                    concat(
                        concat(concat(get("lbr-2"), get("const")), get("blank")),
                        expression(second - 1, third),
                    ),
                    get("rbr-2"),
                )
            )
        atom = _compact(alternative(*parts))
        return _compact(concat(atom, repeat(concat(get("blank"), atom))))

    sentence = concat(
        concat(concat(pattern(depth), get("equal")), expression(depth, depth)),
        get("sep"),
    )
    inner = repeat(concat(repeat(get("eol")), _compact(sentence)))
    definition = concat(
        concat(concat(get("const"), get("lbr-1")), inner),
        concat(repeat(get("eol")), get("rbr-1")),
    )
    program = concat(
        repeat(get("eol")), plus(concat(_compact(definition), plus(get("eol"))))
    )
    return _finish(program, lexicon.alphabet)


def lisp_automaton(lexicon: Lexicon, depth: int = 1) -> DFA:
    """Автомат лексического анализа Brainfuck-Лиспа при вложенности `depth`.

    Правила `[eol][expression]` и `[expression][eol]` вместе дают
    обрамление любым числом `[eol]` с обеих сторон — так они здесь
    и раскрыты.
    """
    if lexicon.kind != LISP:
        raise ValueError("набор лексем не от Brainfuck-Лиспа")
    if depth < 0:
        raise ValueError("вложенность не бывает отрицательной")
    get = lexicon.nfa

    def expression(left: int) -> NFA:
        parts = [get("atom")]
        if left > 0:
            inner = expression(left - 1)
            parts.append(
                concat(
                    concat(concat(concat(get("lbr"), inner), get("dot")), inner),
                    get("rbr"),
                )
            )
            parts.append(concat(concat(get("lbr"), plus(inner)), get("rbr")))
        core = _compact(alternative(*parts))
        return _compact(concat(concat(repeat(get("eol")), core), repeat(get("eol"))))

    program = concat(
        repeat(get("eol")), plus(concat(expression(depth), repeat(get("eol"))))
    )
    return _finish(program, lexicon.alphabet)


def _finish(machine: NFA, alphabet) -> DFA:
    """Детерминизация, минимизация и каноническая нумерация — как просит ТЗ."""
    machine.alphabet = frozenset(alphabet)
    return machine.determinize().minimize().relabel()


def automaton_for(lexicon: Lexicon, depth: int = 1) -> DFA:
    builder = refal_automaton if lexicon.kind == REFAL else lisp_automaton
    return builder(lexicon, depth)


def sample_program(lexicon: Lexicon, depth: int = 1, seed: int = 0) -> str:
    """Случайная программа, построенная **выводом по грамматике**.

    Нужна затем, что автомат собирается операциями над автоматами,
    а вывод идёт по правилам: если они расходятся, ошибка видна сразу.
    Проверка «автомат принимает всё, что вывелось» — арбитр сборки,
    и она не зависит от того, как сборка устроена.
    """
    rng = random.Random(seed)
    catalogue = {name: lexicon.words(name, 4) for name in lexicon.names}

    def pick(name: str) -> str:
        return rng.choice(catalogue[name])

    def many(name: str, least: int = 0, most: int = 2) -> str:
        return "".join(pick(name) for _ in range(rng.randint(least, most)))

    if lexicon.kind == REFAL:

        def atom_pattern(third: int) -> str:
            choices = ["", "var", "const"]
            if third > 0:
                choices.append("nest")
            choice = rng.choice(choices)
            if choice == "nest":
                return pick("lbr-3") + pattern(third - 1) + pick("rbr-3")
            return pick(choice) if choice else ""

        def pattern(third: int) -> str:
            parts = [atom_pattern(third) for _ in range(rng.randint(1, 2))]
            return pick("blank").join(parts)

        def atom_expression(second: int, third: int) -> str:
            choices = ["var", "const"]
            if third > 0:
                choices.append("three")
            if second > 0:
                choices.append("two")
            choice = rng.choice(choices)
            if choice == "three":
                return (
                    pick("lbr-3") + expression(second, third - 1) + pick("rbr-3")
                )
            if choice == "two":
                return (
                    pick("lbr-2")
                    + pick("const")
                    + pick("blank")
                    + expression(second - 1, third)
                    + pick("rbr-2")
                )
            return pick(choice)

        def expression(second: int, third: int) -> str:
            parts = [atom_expression(second, third) for _ in range(rng.randint(1, 2))]
            return pick("blank").join(parts)

        def sentence() -> str:
            return (
                pattern(depth)
                + pick("equal")
                + expression(depth, depth)
                + pick("sep")
            )

        def definition() -> str:
            body = "".join(
                many("eol") + sentence() for _ in range(rng.randint(0, 2))
            )
            return pick("const") + pick("lbr-1") + body + many("eol") + pick("rbr-1")

        return many("eol") + "".join(
            definition() + many("eol", 1, 2) for _ in range(rng.randint(1, 2))
        )

    def expression_lisp(left: int) -> str:
        choices = ["atom"]
        if left > 0:
            choices += ["dotted", "list"]
        choice = rng.choice(choices)
        if choice == "dotted":
            core = (
                pick("lbr")
                + expression_lisp(left - 1)
                + pick("dot")
                + expression_lisp(left - 1)
                + pick("rbr")
            )
        elif choice == "list":
            inner = "".join(
                expression_lisp(left - 1) for _ in range(rng.randint(1, 2))
            )
            core = pick("lbr") + inner + pick("rbr")
        else:
            core = pick("atom")
        return many("eol") + core + many("eol")

    return many("eol") + "".join(
        expression_lisp(depth) + many("eol") for _ in range(rng.randint(1, 2))
    )


def teacher_for(lexicon: Lexicon, depth: int = 1, **options):
    """МАТ для этой цели: `tfl.lstar.DFATeacher` над собранным автоматом."""
    from tfl.lstar import DFATeacher

    return DFATeacher(target=automaton_for(lexicon, depth), **options)
