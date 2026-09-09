"""Копредставления полугрупп и групп: проблема равенства через переписывание.

По лекции 1 курса 2026 (`corpus/txt/FormalLanguageTheory_Lecture1_Trace.txt`,
слайды 4–6, 10–20). Копредставление это множество образующих `A`
и множество определяющих соотношений `R ⊆ A⁺ × A⁺`:

$$\\langle A \\mid R\\rangle_{sg} = A^{+}/\\equiv_R,$$

а для группы — то же самое с добавленными обратными элементами
и свободным сокращением.

## Зачем здесь система переписывания

Соотношение — равенство, у него нет направления. Система переписывания
направление имеет, и в этом весь фокус: **ориентируем** соотношения
армейским порядком, пополняем по Кнуту–Бендиксу и получаем полную
систему. Дальше работает цепочка из той же лекции:

* каждое правило строго убывает в армейском порядке, порядок фундирован,
  значит система **завершима** (слайды 16–18);
* пополнение сошлось — значит все критические пары соединяются, то есть
  система **локально конфлюэнтна** (слайд 14);
* по **лемме Ньюмана** (слайд 15) отсюда конфлюэнтность, а у завершимой
  конфлюэнтной системы нормальная форма единственна.

Поэтому после сошедшегося пополнения `u =_R v` равносильно совпадению
нормальных форм, и проблема равенства решается **точно**.

## Где ответ остаётся неполным

Пополнение может не сойтись, и это законный исход, а не сбой: проблема
равенства в полугруппе неразрешима (теорема Цейтина, слайд 5). Поэтому
`equal` трёхзначна:

* совпали нормальные формы — **равны**, и это верно независимо от того,
  сошлось пополнение или нет: совместимость даёт равенство сразу;
* пополнение сошлось, а формы разные — **не равны**, доказано;
* пополнение не сошлось и формы разные — **не выяснено**.

Обратный элемент записывается заглавной буквой: `A` это `a⁻¹`. Ввод
понимает и `a^-1`, и `a'`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tfl.srs import SRS, Rule, shortlex_key
from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = [
    "Presentation",
    "parse_presentation",
    "GROUP",
    "SEMIGROUP",
]

GROUP = "группа"
SEMIGROUP = "полугруппа"

_HEADER = re.compile(r"^(группа|полугруппа|group|semigroup)\s*[:=]?\s*(.*)$", re.I)
_KIND = {"группа": GROUP, "group": GROUP, "полугруппа": SEMIGROUP, "semigroup": SEMIGROUP}


def _inverse_letter(letter: str) -> str:
    """`a ↦ A`, `A ↦ a`. Заглавная буква и есть обратный элемент."""
    return letter.lower() if letter.isupper() else letter.upper()


def inverse(word: str) -> str:
    """Обратное слово: порядок разворачивается, каждая буква обращается."""
    return "".join(_inverse_letter(letter) for letter in reversed(word))


@dataclass(frozen=True)
class Presentation:
    """Копредставление `⟨A | R⟩`. `kind` — полугруппа или группа."""

    generators: str
    relations: tuple[tuple[str, str], ...]
    kind: str = GROUP

    def __post_init__(self) -> None:
        if self.kind not in (GROUP, SEMIGROUP):
            raise ValueError(f"неизвестный вид копредставления: {self.kind!r}")
        if any(letter.isupper() for letter in self.generators):
            raise ValueError(
                "образующие пишутся строчными: заглавная буква занята "
                "под обратный элемент"
            )
        outside = {
            letter
            for left, right in self.relations
            for letter in left + right
            if letter.lower() not in self.generators
        }
        if outside:
            raise ValueError(f"в соотношениях есть чужие буквы: {''.join(sorted(outside))}")
        if self.kind == SEMIGROUP:
            wrong = {
                letter
                for left, right in self.relations
                for letter in left + right
                if letter.isupper()
            }
            if wrong:
                raise ValueError(
                    f"обратные элементы в полугруппе не определены: "
                    f"{''.join(sorted(wrong))}"
                )

    @property
    def alphabet(self) -> str:
        """Буквы системы: образующие, а у группы ещё и обратные к ним."""
        if self.kind == SEMIGROUP:
            return self.generators
        return "".join(
            letter + letter.upper() for letter in self.generators
        )

    def __str__(self) -> str:
        body = ", ".join(f"{left or 'ε'} = {right or 'ε'}" for left, right in self.relations)
        return f"⟨{', '.join(self.generators)} | {body}⟩_{'gr' if self.kind == GROUP else 'sg'}"

    def markdown(self) -> str:
        body = ",\\; ".join(
            f"{left or '\\varepsilon'} = {right or '\\varepsilon'}"
            for left, right in self.relations
        )
        mark = "gr" if self.kind == GROUP else "sg"
        return rf"$\langle {', '.join(self.generators)} \mid {body} \rangle_{{{mark}}}$"

    # ------------------------------------------------------------------ SRS

    def free_rules(self) -> tuple[Rule, ...]:
        """Свободное сокращение `aa⁻¹ = ε`. У полугруппы его нет."""
        if self.kind == SEMIGROUP:
            return ()
        return tuple(
            Rule(pair, "")
            for letter in self.generators
            for pair in (letter + letter.upper(), letter.upper() + letter)
        )

    def precedence(self) -> str:
        """Приоритет букв для армейского порядка: как выписаны образующие.

        Порядок выбран, а не выведен, и это существенно: от него зависит,
        сойдётся пополнение или разойдётся. `complete` перебирает варианты.
        """
        return self.alphabet

    def rewriting(self, precedence: str | None = None) -> SRS:
        """Ориентировать соотношения армейским порядком.

        Каждое правило после этого строго убывает, поэтому система
        **завершима по построению** — а вот конфлюэнтна ли она,
        решает пополнение.
        """
        order = precedence or self.precedence()
        rules = list(self.free_rules())
        for left, right in self.relations:
            if left == right:
                continue
            lo, hi = left, right
            if shortlex_key(lo, order) < shortlex_key(hi, order):
                lo, hi = hi, lo
            rules.append(Rule(lo, hi))
        return SRS(tuple(rules), frozenset(self.alphabet))

    def complete(
        self, max_rules: int = 60, max_rounds: int = 30, max_len: int = 16
    ) -> tuple[SRS, Verdict]:
        """Пополнить по Кнуту–Бендиксу, перебирая приоритеты букв.

        Приоритет перебирается не для красоты: на одном порядке процедура
        расходится, на другом сходится за несколько раундов, и заранее
        сказать нельзя.
        """
        orders = [self.precedence()]
        if self.kind == GROUP:
            orders.append("".join(letter.upper() + letter for letter in self.generators))
            orders.append(self.generators + self.generators.upper())
        else:
            orders.append(self.generators[::-1])
        last: tuple[SRS, Verdict] | None = None
        for order in orders:
            system, verdict = self.rewriting(order).complete(
                order, max_rules, max_rounds, max_len
            )
            if verdict.value is True:
                return system, proved(
                    f"пополнение сошлось при приоритете букв {order}: "
                    f"{len(system)} правил, все критические пары соединяются. "
                    f"Система завершима (каждое правило убывает в армейском "
                    f"порядке) и локально конфлюэнтна, значит по лемме Ньюмана "
                    f"конфлюэнтна: нормальная форма единственна",
                    system,
                )
            last = (system, verdict)
        system, verdict = last
        return system, unknown(
            f"пополнение не сошлось ни при одном из перебранных приоритетов "
            f"({', '.join(orders)}): {verdict.reason}. Это законный исход — "
            f"проблема равенства в полугруппе неразрешима"
        )

    # -------------------------------------------------------- равенство слов

    def equal(self, left: str, right: str, max_len: int = 16) -> Verdict:
        """Равны ли слова в копредставлении. Три исхода.

        Совпадение нормальных форм доказывает равенство **всегда**:
        это цепочка соотношений, применённых в обе стороны. Различие
        доказательно только при сошедшемся пополнении.
        """
        system, verdict = self.complete(max_len=max_len)
        forms = {}
        for word in (left, right):
            found = system.normal_forms(word, max_len)
            forms[word] = min(
                found.words or {word}, key=lambda w: shortlex_key(w, self.precedence())
            )
        if forms[left] == forms[right]:
            return proved(
                f"«{left or 'ε'}» и «{right or 'ε'}» приводятся к одной "
                f"нормальной форме «{forms[left] or 'ε'}»",
                forms[left],
            )
        if verdict.value is True:
            return refuted(
                f"нормальные формы разные: «{forms[left] or 'ε'}» против "
                f"«{forms[right] or 'ε'}», а система полна — значит слова "
                f"не равны",
                (forms[left], forms[right]),
            )
        return unknown(
            f"нормальные формы разные («{forms[left] or 'ε'}» и "
            f"«{forms[right] or 'ε'}»), но система не пополнена до полной, "
            f"поэтому вывода нет: {verdict.reason}"
        )

    def is_trivial(self, max_len: int = 16) -> Verdict:
        """Тривиальна ли группа: каждая образующая равна пустому слову."""
        for letter in self.generators:
            verdict = self.equal(letter, "", max_len)
            if verdict.value is not True:
                return Verdict(
                    verdict.value,
                    f"образующая «{letter}» единице не равна ({verdict.reason})",
                    letter,
                )
        return proved(
            f"каждая из образующих {', '.join(self.generators)} равна единице, "
            f"значит группа тривиальна"
        )

    def is_commutative(self, max_len: int = 16) -> Verdict:
        """Коммутативно ли копредставление.

        Достаточно проверить образующие: если они попарно коммутируют,
        коммутируют и любые произведения — перестановка соседних букв
        переносит любую букву куда угодно.
        """
        unresolved: list[str] = []
        for i, first in enumerate(self.generators):
            for second in self.generators[i + 1 :]:
                verdict = self.equal(first + second, second + first, max_len)
                if verdict.value is False:
                    return refuted(
                        f"образующие «{first}» и «{second}» не коммутируют: "
                        f"{verdict.reason}",
                        (first, second),
                    )
                if verdict.value is None:
                    unresolved.append(f"{first}{second}")
        if unresolved:
            return unknown(
                f"не удалось решить для пар: {', '.join(unresolved)} — "
                f"пополнение не сошлось"
            )
        return proved(
            "все образующие попарно коммутируют, значит коммутируют и любые "
            "элементы: перестановка соседних букв переносит букву куда угодно"
        )


def parse_presentation(text: str, kind: str = GROUP) -> Presentation:
    """Разобрать копредставление из текста.

    Первая строка может задавать вид и образующие (`группа: a, b`), дальше
    идут соотношения `слово = слово`. Обратный элемент — заглавная буква;
    ввод понимает также `a^-1` и `a'`.

    >>> p = parse_presentation("группа: a, b\\na b A = b b")
    >>> str(p)
    '⟨a, b | abA = bb⟩_gr'
    """
    generators = ""
    relations: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header and "=" not in header.group(2):
            kind = _KIND[header.group(1).lower()]
            generators = "".join(header.group(2).replace(",", " ").split())
            continue
        if "=" not in line:
            raise ValueError(f"в строке нет знака равенства: {raw!r}")
        left, right = line.split("=", 1)
        relations.append((_word(left), _word(right)))
    if not generators:
        generators = "".join(
            sorted({letter.lower() for left, right in relations for letter in left + right})
        )
    return Presentation(generators, tuple(relations), kind)


def _word(raw: str) -> str:
    """Слово из записи: пробелы выкидываются, `a^-1` и `a'` становятся `A`."""
    text = raw.strip()
    if text in {"1", "e", "ε", "eps"}:
        return ""
    text = re.sub(r"([A-Za-z])\s*(\^\s*-\s*1|')", lambda m: _inverse_letter(m.group(1)), text)
    return "".join(text.split())


def as_srs(presentation: Presentation) -> SRS:
    """Ориентированная система копредставления — для `tfl/srs.py` целиком."""
    return presentation.rewriting()
