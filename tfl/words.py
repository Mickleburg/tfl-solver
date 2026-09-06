"""Перечисление слов — фундамент всех конечных проверок.

Почти каждый оракул в `tfl/` устроен одинаково: перебрать все слова длины
не больше N и сравнить два вердикта. Здесь единственная реализация этого
перебора, чтобы «все слова до длины N» везде означало ровно одно и то же.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, Iterator

__all__ = ["iter_words", "count_words", "random_word", "sample_words"]


def iter_words(alphabet: Iterable[str], max_len: int, min_len: int = 0) -> Iterator[str]:
    """Все слова длины от `min_len` до `max_len` в шортлекс-порядке.

    Шортлекс (сначала по длине, потом лексикографически) важен: первое
    найденное слово-свидетель автоматически оказывается кратчайшим, а это
    ровно то, что нужно предъявлять в отчёте.

    >>> list(iter_words("ab", 2))
    ['', 'a', 'b', 'aa', 'ab', 'ba', 'bb']
    """
    letters = sorted(set(alphabet))
    for length in range(min_len, max_len + 1):
        for tup in itertools.product(letters, repeat=length):
            yield "".join(tup)


def count_words(alphabet: Iterable[str], max_len: int, min_len: int = 0) -> int:
    """Сколько слов переберёт `iter_words` — для оценки стоимости проверки."""
    k = len(set(alphabet))
    if k == 0:
        return 1 if min_len == 0 else 0
    if k == 1:
        return max(0, max_len - min_len + 1)
    return sum(k**n for n in range(min_len, max_len + 1))


def random_word(alphabet: Iterable[str], max_len: int, rng: random.Random) -> str:
    """Случайное слово длины 0..max_len (длина равномерна, не слова)."""
    letters = sorted(set(alphabet))
    if not letters:
        return ""
    length = rng.randint(0, max_len)
    return "".join(rng.choice(letters) for _ in range(length))


def sample_words(
    alphabet: Iterable[str], count: int, max_len: int = 12, seed: int = 0
) -> list[str]:
    """Воспроизводимая выборка случайных слов для фазз-тестирования.

    Seed фиксирован по умолчанию: отчёт должен быть воспроизводим, а «прогнал
    10⁴ случайных слов» без указания seed непроверяемо.
    """
    rng = random.Random(seed)
    return [random_word(alphabet, max_len, rng) for _ in range(count)]
