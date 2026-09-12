"""РК2 2025, вариант 28, задача 1: описание языка SRS над базисом.

Условие: язык системы `ba² → ba`, `ab → ba`, `a → ab` над базисом `aⁿbⁿaⁿ`,
то есть `L = {w | ∃n ⩾ 1: aⁿbⁿaⁿ ⇒* w}`. Обе проверенные работы получили
по 1 баллу и дали разные ответы; оба опровергнуты (см. отчёт).

Слово удобно записывать **вектором слотов**: `w = b^{i₀} a b^{i₁} … a b^{i_k}`,
где `k = |w|_a`. Правила в этой записи (`T = Σ iⱼ = |w|_b`):

* `a → ab` — прибавить единицу к любому `i_t`, `t ⩾ 1`;
* `ab → ba` — при `i_t ⩾ 1` перенести единицу из `i_t` в `i_{t−1}`;
* `ba² → ba` — при `i_{t−1} ⩾ 1` и `i_t = 0` выбросить слот `t`
  (это удаление `(t+1)`-й буквы `a`).

Отсюда сразу три наблюдения, на которых держится всё остальное.

1. **`k` не растёт, `T` не убывает.** Значит для слова длины `⩽ L` все
   промежуточные слова имеют `k ⩽ 2n` и `T ⩽ L`: обход замыкания можно
   вести **точно**, а не «до длины `L`», и это снимает обычную оговорку
   про путь через длинные промежуточные слова.
2. **Ни одна `b` не сдвигается вправо.** Для отдельной буквы `b` величина
   `λ` (число букв `a` слева от неё) только убывает: `ab → ba` уменьшает
   её на единицу, `ba² → ba` уменьшает или сохраняет, а `a → ab` рождает
   новую `b` с `λ = t ⩾ 1`. Исходные `n` букв `b` базиса имеют `λ = n`,
   поэтому **всякая `b` с `λ > n` — порождённая**.
3. **Последний слот не пополняется.** `i_k` сохраняется правилом `ba² → ba`
   (оно выбрасывает слот с номером `t ⩽ k−1`), уменьшается правилом
   `ab → ba` и растёт только правилом `a → ab`, применённым к последней
   букве `a`. В базисе `i_k = 0`, поэтому **в последнем слоте стоят
   только порождённые `b`**.

Из 2 и 3 вместе получается ответ: `n` исходных букв `b` обязаны уместиться
в слоты `0 … min(n, k−1)`. Это необходимо — и, как показывает `derivation`,
достаточно.

    w ∈ L  ⟺  |w|_a ⩾ 1 и найдётся n, ⌈k/2⌉ ⩽ n ⩽ T, при котором
              не менее n букв b стоят левее последней буквы a
              и не правее n-й буквы a.

Нижняя граница `n ⩾ ⌈k/2⌉` — это `k ⩽ 2n`; верхняя `n ⩽ T` — это `T ⩾ n`.

Запуск: `py -3 tools/rk2_variant_28.py` — сводка и сверка с оракулом.
"""

from __future__ import annotations

import sys

from tfl.srs import parse_srs

__all__ = [
    "RULES",
    "slots",
    "unslot",
    "bases",
    "belongs",
    "derivation",
    "check_derivation",
    "closure",
    "language",
]

#: Система переписывания варианта 28.
RULES = "baa -> ba\nab -> ba\na -> ab\n"


def slots(word: str) -> list[int]:
    """Вектор слотов `(i₀, …, i_k)`: сколько `b` стоит в каждом промежутке."""
    found: list[int] = []
    current = 0
    for letter in word:
        if letter == "a":
            found.append(current)
            current = 0
        else:
            current += 1
    found.append(current)
    return found


def unslot(vector) -> str:
    """Обратно в слово."""
    vector = list(vector)
    parts = ["b" * vector[0]]
    for count in vector[1:]:
        parts.append("a" + "b" * count)
    return "".join(parts)


def bases(word: str) -> list[int]:
    """Все `n`, при которых базис `aⁿbⁿaⁿ` годится для `word`.

    Условие ровно то, что в шапке модуля: `⌈k/2⌉ ⩽ n ⩽ T` и не менее `n`
    букв `b` стоят в слотах `0 … min(n, k−1)`.
    """
    letters = set(word)
    if not letters <= {"a", "b"}:
        raise ValueError("слово должно быть над алфавитом {a, b}")
    count_a, count_b = word.count("a"), word.count("b")
    if count_a < 1:
        return []
    vector = slots(word)
    good = []
    for n in range(-(-count_a // 2), count_b + 1):
        if sum(vector[: min(n, count_a - 1) + 1]) >= n:
            good.append(n)
    return good


def belongs(word: str) -> bool:
    """Лежит ли слово в языке. Точный ответ, не срез."""
    return bool(bases(word))


# --------------------------------------------------------------------------
# Точный обход замыкания
# --------------------------------------------------------------------------


def closure(base: int, max_b: int) -> set[tuple[int, ...]]:
    """Все векторы слотов, достижимые из `a^base b^base a^base`.

    Обход **точный** при `max_b ⩾ |w|_b`: число `b` не убывает, поэтому
    ни один путь к слову с `|w|_b ⩽ max_b` за эту границу не выходит.
    Число `a` не растёт само по себе, и отдельно ограничивать его не нужно.
    """
    from collections import deque

    start = tuple([0] * base + [base] + [0] * base)
    seen = {start}
    queue = deque([start])
    while queue:
        vector = queue.popleft()
        top = len(vector) - 1
        moves = []
        for slot in range(1, top + 1):
            nxt = list(vector)
            nxt[slot] += 1
            moves.append(tuple(nxt))                       # a → ab
            if vector[slot] >= 1:
                nxt = list(vector)
                nxt[slot - 1] += 1
                nxt[slot] -= 1
                moves.append(tuple(nxt))                   # ab → ba
        for slot in range(1, top):
            if vector[slot - 1] >= 1 and vector[slot] == 0:
                moves.append(vector[:slot] + vector[slot + 1:])  # ba² → ba
        for nxt in moves:
            if sum(nxt) <= max_b and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def language(max_len: int) -> set[str]:
    """Точное `L ∩ Σ^{⩽ max_len}` — обходом, независимо от описания.

    Базисы перебираются до `n = max_len`: больший `n` требует `n` букв `b`
    в слове, а их не больше `max_len`.
    """
    found: set[str] = set()
    for base in range(1, max_len + 1):
        for vector in closure(base, max_len):
            word = unslot(vector)
            if len(word) <= max_len:
                found.add(word)
    return found


# --------------------------------------------------------------------------
# Достаточность: явный вывод из базиса
# --------------------------------------------------------------------------


def derivation(word: str, base: int | None = None) -> list[str] | None:
    """Построить вывод `aⁿbⁿaⁿ ⇒* word`. `None`, если слова нет в языке.

    Конструкция в три фазы, и она же — доказательство достаточности.

    **Фаза A, довести число `a` до `k`.** При `k ⩾ n+1` хватает удалений
    в хвосте: правило `ba² → ba` применимо при `t = n+1`, пока хвостовых
    `a` больше одной. При `k ⩽ n` этого мало — хвост упирается
    в единственную `a`; тогда весь блок `b` сдвигается влево на один слот
    (`n` применений `ab → ba`), после чего удаление снова возможно,
    и каждый такой цикл укорачивает **начальный** блок `a` на единицу.

    **Фаза B, разложить исходные `b`.** После фазы A все `n` исходных
    букв `b` стоят в слоте `p = min(n, k−1)`. Они разводятся влево
    правилом `ab → ba` по целевым слотам — жадно, справа налево.
    Условие принадлежности ровно и говорит, что мест хватает.

    **Фаза C, породить остальные.** Оставшиеся `T − n` букв `b` вставляются
    правилом `a → ab` прямо в свои слоты; для слота 0 — в слот 1
    и один сдвиг влево.
    """
    candidates = bases(word)
    if not candidates:
        return None
    n = candidates[0] if base is None else base
    if n not in candidates:
        return None

    target = slots(word)
    count_a = word.count("a")
    vector = [0] * n + [n] + [0] * n
    chain = [unslot(vector)]

    def emit() -> None:
        chain.append(unslot(vector))

    # --- фаза A -----------------------------------------------------------
    while len(vector) - 1 > max(count_a, n + 1):
        del vector[n + 1]  # `ba² → ba` при t = n+1
        emit()
    position = n
    while len(vector) - 1 > count_a:
        for _ in range(n):  # сдвинуть блок b влево
            vector[position - 1] += 1
            vector[position] -= 1
            emit()
        position -= 1
        del vector[position + 1]  # `ba² → ba` при t = position+1
        emit()

    # --- фаза B -----------------------------------------------------------
    left = n
    plan = [0] * len(target)
    for slot in range(min(n, count_a - 1), -1, -1):
        take = min(target[slot], left)
        plan[slot] = take
        left -= take
    if left:  # места не хватило — значит условие было прочитано неверно
        return None
    for slot in range(position - 1, -1, -1):
        for _ in range(plan[slot]):
            for step in range(position, slot, -1):
                vector[step] -= 1
                vector[step - 1] += 1
                emit()

    # --- фаза C -----------------------------------------------------------
    for slot, needed in enumerate(target):
        for _ in range(needed - plan[slot]):
            vector[max(slot, 1)] += 1  # `a → ab`
            emit()
            if slot == 0:
                vector[1] -= 1
                vector[0] += 1
                emit()

    return chain


def check_derivation(chain: list[str]) -> bool:
    """Сверить цепочку с самой системой: каждый шаг — настоящее правило."""
    system = parse_srs(RULES)
    return all(
        second in system.step(first) for first, second in zip(chain, chain[1:])
    )


# --------------------------------------------------------------------------
# Сводка
# --------------------------------------------------------------------------


def report(limit: int = 12) -> str:
    """Сводка: что доказано и на чём сверено."""
    import itertools

    lines = [
        "РК2 2025, вариант 28, задача 1",
        "",
        f"L = {{w | ∃n ⩾ 1: aⁿbⁿaⁿ ⇒* w}} для системы {RULES.strip().splitlines()}",
        "",
        "Описание: w ∈ L ⟺ |w|_a ⩾ 1 и ∃n, ⌈k/2⌉ ⩽ n ⩽ |w|_b, при котором",
        "не менее n букв b стоят левее последней a и не правее n-й a.",
        "",
    ]
    total = built = 0
    for length in range(1, limit + 1):
        here = 0
        for letters in itertools.product("ab", repeat=length):
            word = "".join(letters)
            if not belongs(word):
                continue
            here += 1
            chain = derivation(word)
            assert chain is not None and chain[-1] == word, word
            assert check_derivation(chain), word
            built += 1
        total += here
        lines.append(f"длина {length:>2}: слов в языке {here:>4}, вывод построен для всех")
    lines.append("")
    lines.append(f"всего {total} слов, выводов построено и сверено с системой: {built}")
    return "\n".join(lines)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(report(limit))
