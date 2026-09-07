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

from collections.abc import Set as AbstractSet
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
    "prefix_splits",
    "dcfl_splits",
    "defeats_dcfl",
    "nondcfl_by_pumping",
    "ogden_splits",
    "defeats_ogden",
    "noncf_by_ogden",
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


# --------------------------------------------------------------------------
# Лемма Ю: накачка для детерминированных КС-языков
# --------------------------------------------------------------------------


def prefix_splits(word: str, p: int) -> Iterator[tuple[str, str, str]]:
    """Разбиения `x = x₁x₂x₃` с `|x₂x₃| ≤ p` и `|x₂| > 0`.

    Условие `|x₂x₃| ≤ p` означает, что накачиваемый кусок лежит
    в **последних** `p` буквах префикса — это и отличает лемму Ю
    от обычной накачки, где кусок может стоять где угодно.
    """
    n = len(word)
    for i in range(max(0, n - p), n):
        for j in range(i + 1, n + 1):
            yield word[:i], word[i:j], word[j:]


def _three_splits(word: str) -> Iterator[tuple[str, str, str]]:
    for i in range(len(word) + 1):
        for j in range(i, len(word) + 1):
            yield word[:i], word[i:j], word[j:]


def dcfl_splits(
    x: str, y: str, z: str, p: int
) -> Iterator[tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]]:
    """Все разбиения из случая 2 теоремы Ю.

    `x = x₁x₂x₃` (`|x₂x₃| ≤ p`, `|x₂| > 0`), `y = y₁y₂y₃`, `z = z₁z₂z₃`;
    накачиваются `x₂`, `y₂` и `z₂` **одновременно и одной степенью**.
    Красная пометка на проверенной работе — «можно качать x с y синхронно
    в одном слове» — про то, что разбирать их по отдельности нельзя.
    """
    for xs in prefix_splits(x, p):
        for ys in _three_splits(y):
            for zs in _three_splits(z):
                yield xs, ys, zs


def defeats_dcfl(
    language: Language,
    x: str,
    y: str,
    z: str,
    p: int,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """Отбивает ли пара слов накачку по лемме Ю при данной `p`.

    Теорема (S. Yu), лекция 9 курса: пусть `L` — DCFL. Тогда существует `p`
    такое, что для **всех** пар `w = xy ∈ L`, `w′ = xz ∈ L` с `|x| > p`
    и совпадающими первыми буквами `y` и `z` выполнено одно из двух:

    1. существует накачка только префикса `x` в привычном смысле;
    2. существует разбиение `x = x₁x₂x₃`, `y = y₁y₂y₃`, `z = z₁z₂z₃`
       с `|x₂x₃| ≤ p`, `|x₂| > 0` и `∀i` оба слова
       `x₁x₂ⁱx₃y₁y₂ⁱy₃` и `x₁x₂ⁱx₃z₁z₂ⁱz₃` лежат в `L`.

    `True` — оба случая провалились на всех разбиениях, то есть пара
    годится как свидетель при этой `p`.

    Прочтение случая 1 определяется самим примером из лекции. Словами
    там сказано только «накачка только префикса x (в привычном смысле)»,
    и остаётся вопрос: обязаны ли остаться в языке оба слова пары или
    достаточно одного. На `L = {aⁿbⁿ} ∪ {aⁿb²ⁿ}` со свидетелем
    `x = aⁿbⁿ⁻¹`, `y = b`, `z = bⁿ⁺¹` при слабом прочтении случай 1
    **выполняется** (синхронная накачка `a` и `b` даёт `a^{n+k}b^{n+k}`),
    а лекция утверждает, что «в случае 1 нет подходящей накачки».
    Значит требуются оба слова — так здесь и сделано, и на этом примере
    результат совпадает с лекцией.
    """
    for word, name in ((x + y, "xy"), (x + z, "xz")):
        if word not in language:
            return refuted(f"слово {name} = «{word}» не в языке — пара негодна", word)
    if len(x) <= p:
        return refuted(f"|x| = {len(x)} ≤ {p}: лемма требует |x| > p", x)
    if not y or not z or y[0] != z[0]:
        return refuted("первые буквы y и z обязаны совпадать", (y, z))

    # случай 1: привычная КС-накачка внутри префикса — для обоих слов сразу
    for u, v, m, t, s in cf_splits(x, p):
        pumped = [u + v * k + m + t * k + s for k in powers]
        if all(
            (head + y) in language and (head + z) in language for head in pumped
        ):
            return refuted(
                f"случай 1: префикс накачивается разбиением "
                f"v=«{v}» y=«{t}», оба слова остаются в языке",
                (u, v, m, t, s),
            )

    # случай 2: синхронная накачка префикса и обоих хвостов
    for (x1, x2, x3), (y1, y2, y3), (z1, z2, z3) in dcfl_splits(x, y, z, p):
        if all(
            (x1 + x2 * k + x3 + y1 + y2 * k + y3) in language
            and (x1 + x2 * k + x3 + z1 + z2 * k + z3) in language
            for k in powers
        ):
            return refuted(
                f"случай 2: разбиение x₂=«{x2}» y₂=«{y2}» z₂=«{z2}» "
                "накачивается, оба слова остаются в языке",
                ((x1, x2, x3), (y1, y2, y3), (z1, z2, z3)),
            )

    return Verdict(
        True, f"оба случая леммы Ю провалились при p = {p}", (x, y, z)
    )


def nondcfl_by_pumping(
    language: Language,
    witness: Callable[[int], tuple[str, str, str]],
    upto: int = 4,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """Прогнать свидетеля-пару по лемме Ю при всех `p ≤ upto`.

    `witness(p)` возвращает тройку `(x, y, z)`: общий префикс и два хвоста.
    Классический пример из лекции — `L = {aⁿbⁿ} ∪ {aⁿb²ⁿ}`,
    `x = aⁿbⁿ⁻¹`, `y = b`, `z = bⁿ⁺¹` при `n − 1 > p`.

    Как и в остальных леммах о накачке, «не выяснено» — потолок:
    обобщение на произвольное `p` пишет человек.
    """
    for p in range(1, upto + 1):
        x, y, z = witness(p)
        verdict = defeats_dcfl(language, x, y, z, p, powers)
        if verdict.value is not True:
            return refuted(
                f"при p = {p} пара («{x}», «{y}», «{z}») не отбивает накачку: "
                f"{verdict.reason}",
                (p, x, y, z, verdict.witness),
            )
    x, y, z = witness(upto)
    return unknown(
        f"пара отбивает накачку по лемме Ю при всех p ≤ {upto}; чтобы получить "
        "доказательство, что язык не DCFL, нужно то же рассуждение "
        "при произвольном p",
        (x, y, z),
    )


# --------------------------------------------------------------------------
# Лемма Огдена: накачка с отмеченными позициями
# --------------------------------------------------------------------------


def ogden_splits(
    word: str, marks: AbstractSet[int], n: int
) -> Iterator[tuple[str, str, str, str, str]]:
    """Все разбиения `w = x₁y₁zy₂x₂`, допустимые леммой Огдена.

    Формулировка курса (лекция 6, «Бонус: лемма Огдена») требует, чтобы
    отмеченные буквы были **во всех трёх** частях одной из двух троек —
    либо в `x₁, y₁, z`, либо в `z, y₂, x₂`, — и чтобы в `y₁zy₂` было
    отмечено не более `n` букв.

    Это не то же самое, что расхожая формулировка «`y₁` и `y₂` вместе
    содержат отмеченную позицию»: курсовая сильнее, и разбор случаев
    получается другой. Реализована именно курсовая.
    """
    marked = sorted(marks)
    size = len(word)

    def count(lo: int, hi: int) -> int:
        return sum(1 for m in marked if lo <= m < hi)

    for i in range(size + 1):
        for j in range(i, size + 1):
            for k in range(j, size + 1):
                for l in range(k, size + 1):
                    if count(i, l) > n:  # отмечено в y₁zy₂ не более n
                        continue
                    left = count(0, i) and count(i, j) and count(j, k)
                    right = count(j, k) and count(k, l) and count(l, size)
                    if not (left or right):
                        continue
                    yield word[:i], word[i:j], word[j:k], word[k:l], word[l:]


def defeats_ogden(
    language: Language,
    word: str,
    marks: AbstractSet[int],
    n: int,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """Отбивает ли отмеченное слово накачку по Огдену при данном `n`.

    Отметки — это наш ход, и в них вся сила леммы: выбирая, какие позиции
    отметить, мы запрещаем противнику качать удобный ему блок.
    """
    if word not in language:
        return refuted(f"слово «{word}» не принадлежит языку — свидетель негоден", word)
    if len(word) < n:
        return refuted(f"|«{word}»| < {n}: лемма к такому слову неприменима", word)
    if not marks <= set(range(len(word))):
        return refuted("отмечены позиции за пределами слова", sorted(marks))
    if len(marks) < n:
        return refuted(
            f"отмечено {len(marks)} букв, а лемма требует не менее {n}", sorted(marks)
        )

    seen = False
    for x1, y1, z, y2, x2 in ogden_splits(word, marks, n):
        seen = True
        if all((x1 + y1 * k + z + y2 * k + x2) in language for k in powers):
            return refuted(
                f"разбиение x₁=«{x1}» y₁=«{y1}» z=«{z}» y₂=«{y2}» x₂=«{x2}» "
                "накачивается, оставаясь в языке",
                (x1, y1, z, y2, x2),
            )
    if not seen:
        return unknown(
            f"при n = {n} и такой отметке допустимых разбиений нет вовсе — "
            "лемма здесь ничего не говорит. Чтобы обе тройки были достижимы, "
            "отмеченных букв нужно не меньше трёх",
            (word, sorted(marks)),
        )
    return Verdict(
        True, f"все допустимые разбиения при n = {n} выводят слово из языка", word
    )


def noncf_by_ogden(
    language: Language,
    witness: Callable[[int], tuple[str, AbstractSet[int]]],
    upto: int = 4,
    powers: Sequence[int] = POWERS,
) -> Verdict:
    """Проверить свидетеля по Огдену на всех `n ≤ upto`.

    `witness(n)` возвращает пару «слово, множество отмеченных позиций».
    Пример из лекции — язык `{aᵐbⁿcⁿdⁿ | m > 0} ∪ {bⁱcʲdᵏ}`, слово
    `ab²ⁿc²ⁿd²ⁿ` с отметкой `n` последних букв `d`.

    Исходы те же, что у обычной накачки: «доказано» среди них нет, потому
    что перебор конечного числа `n` доказательством не является. Зато
    `False` точен — свидетель или отметки негодны.
    """
    vacuous: list[int] = []
    useful = 0
    for n in range(1, upto + 1):
        word, marks = witness(n)
        verdict = defeats_ogden(language, word, marks, n, powers)
        if verdict.value is False:
            return refuted(
                f"при n = {n} свидетель «{word}» не отбивает накачку: {verdict.reason}",
                (n, word, verdict.witness),
            )
        if verdict.value is None:
            vacuous.append(n)
        else:
            useful += 1
    last, _ = witness(upto)
    if not useful:
        return unknown(
            f"при всех n ≤ {upto} допустимых разбиений не нашлось ни одного: "
            "отметка вырождена, проверка ничего не дала",
            last,
        )
    if vacuous:
        return unknown(
            f"свидетель отбивает накачку по Огдену при всех n ≤ {upto}, "
            f"но при n из {vacuous} разбиений нет вовсе и проверка там пустая",
            last,
        )
    return unknown(
        f"свидетель отбивает накачку по Огдену при всех n ≤ {upto}; для "
        "доказательства, что язык не контекстно-свободен, нужно то же "
        "рассуждение при произвольном n",
        last,
    )
