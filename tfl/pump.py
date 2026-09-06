"""Леммы о накачке: проверка свидетеля перебором всех разбиений.

Обслуживает `RK1-*`, `RK2-B`, `EXAM-1` вместе с `tfl/lang.py`.

Что здесь механизируется. Лемма о накачке устроена как игра: противник
выбирает длину накачки `p`, мы выбираем слово `w ∈ L` длины не меньше `p`,
противник разбивает его, мы выбираем степень. Наш ход — содержательный,
ход противника — конечный перебор. Модуль берёт на себя перебор:
предъявленное слово проверяется **на всех** разбиениях сразу, а не на том
одном, которое пришло в голову.

Именно на этом шаге чаще всего ошибаются в отчётах: разбирают удобное
разбиение и объявляют язык нерегулярным, не заметив, что какое-то другое
разбиение накачивается и остаётся в языке.

Про границы вывода — как и везде в проекте. Отбив накачку для `p ≤ N`,
мы **не** доказали нерегулярность: доказательство требует произвольного
`p`, и обобщение пишет человек. Зато обратный исход точен: если для
какого-то `p` предъявленное слово не отбивает накачку, свидетель негоден,
и это видно сразу.
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

from tfl.lang import Language
from tfl.verdict import Verdict, refuted, unknown

__all__ = [
    "regular_splits",
    "cf_splits",
    "defeats_regular",
    "defeats_cf",
    "nonregular_by_pumping",
    "noncf_by_pumping",
]

#: Степени, которыми пробуем накачивать. Ноль (стирание) — самый
#: результативный случай, поэтому идёт первым.
POWERS = (0, 2, 3, 4)


def regular_splits(word: str, p: int) -> Iterator[tuple[str, str, str]]:
    """Все разбиения `w = xyz` с `|xy| ≤ p` и `|y| > 0` — как требует лемма."""
    for i in range(min(p, len(word)) + 1):
        for j in range(i + 1, min(p, len(word)) + 1):
            yield word[:i], word[i:j], word[j:]


def cf_splits(word: str, p: int) -> Iterator[tuple[str, str, str, str, str]]:
    """Все разбиения `w = uvxyz` с `|vxy| ≤ p` и `|vy| > 0`.

    Накачиваются `v` и `y` **синхронно** — в этом вся разница с регулярным
    случаем, и именно поэтому языки вида `aⁿbⁿcⁿ` накачку не проходят,
    а `aⁿbⁿ` проходит.
    """
    n = len(word)
    for start in range(n + 1):
        for end in range(start, min(start + p, n) + 1):
            block = word[start:end]
            for i in range(len(block) + 1):
                for j in range(i, len(block) + 1):
                    v, x, y = block[:i], block[i:j], block[j:]
                    if not v and not y:
                        continue
                    yield word[:start], v, x, y, word[end:]


def defeats_regular(
    language: Language, word: str, p: int, powers: Sequence[int] = POWERS
) -> Verdict:
    """Отбивает ли слово накачку для данной длины `p`.

    `True` — для **каждого** разбиения нашлась степень, выводящая слово
    из языка. Это точный результат: перебор конечен и пройден целиком.
    `False` — нашлось разбиение, которое накачивается и остаётся в языке
    при всех проверенных степенях; свидетель предъявляется.
    """
    if word not in language:
        return refuted(f"слово «{word}» не принадлежит языку — свидетель негоден", word)
    if len(word) < p:
        return refuted(f"|«{word}»| < {p}: лемма к такому слову неприменима", word)

    for x, y, z in regular_splits(word, p):
        if all((x + y * k + z) in language for k in powers):
            return refuted(
                f"разбиение x=«{x}» y=«{y}» z=«{z}» накачивается, оставаясь в языке",
                (x, y, z),
            )
    return Verdict(True, f"все разбиения при p = {p} выводят слово из языка", word)


def defeats_cf(
    language: Language, word: str, p: int, powers: Sequence[int] = POWERS
) -> Verdict:
    """То же для леммы о накачке КС-языков: `w = uvxyz`, качаем `v` и `y`."""
    if word not in language:
        return refuted(f"слово «{word}» не принадлежит языку — свидетель негоден", word)
    if len(word) < p:
        return refuted(f"|«{word}»| < {p}: лемма к такому слову неприменима", word)

    for u, v, x, y, z in cf_splits(word, p):
        if all((u + v * k + x + y * k + z) in language for k in powers):
            return refuted(
                f"разбиение u=«{u}» v=«{v}» x=«{x}» y=«{y}» z=«{z}» "
                "накачивается, оставаясь в языке",
                (u, v, x, y, z),
            )
    return Verdict(True, f"все разбиения при p = {p} выводят слово из языка", word)


def _by_pumping(
    language: Language,
    witness: Callable[[int], str],
    upto: int,
    powers: Sequence[int],
    check: Callable[[Language, str, int, Sequence[int]], Verdict],
    kind: str,
) -> Verdict:
    for p in range(1, upto + 1):
        word = witness(p)
        verdict = check(language, word, p, powers)
        if verdict.value is not True:
            return refuted(
                f"при p = {p} свидетель «{word}» не отбивает накачку: {verdict.reason}",
                (p, word, verdict.witness),
            )
    return unknown(
        f"свидетель отбивает накачку при всех p ≤ {upto}; для доказательства, "
        f"что язык не {kind}, нужно то же рассуждение при произвольном p",
        witness(upto),
    )


def nonregular_by_pumping(
    language: Language,
    witness: Callable[[int], str],
    upto: int = 8,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """Проверить свидетеля нерегулярности на всех `p ≤ upto`.

    `witness(p)` — слово, которое мы предъявляем против длины накачки `p`;
    обычно что-то вроде `lambda p: "a" * p + "b" * p`.

    Исходов два, и «доказано» среди них нет: перебор конечного числа `p`
    доказательством не является. Зато `False` точен — такой свидетель
    негоден, и переходить к оформлению отчёта с ним нельзя.
    """
    return _by_pumping(language, witness, upto, powers, defeats_regular, "регулярен")


def noncf_by_pumping(
    language: Language,
    witness: Callable[[int], str],
    upto: int = 6,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """То же для контекстной свободы. `upto` меньше: разбиений здесь `O(|w|⁴)`."""
    return _by_pumping(language, witness, upto, powers, defeats_cf, "контекстно-свободен")
