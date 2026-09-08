"""Кодировки и морфизмы: однозначная декодируемость и задержка.

Семинар 05.09.2026 (issue #44) целиком про этот класс задач, и в таксономии
его до сих пор не было: инъективность кодировки, гомоморфность, задержка
раскодирования, характеризация кодируемых строк.

Главное здесь — **алгоритм Сардинаса–Паттерсона**. Он решает вопрос
об однозначной декодируемости точно, а не перебором. Разница
принципиальная: перебор до длины `n` показывает, что коллизий нет
среди коротких слов, и ничего не говорит про длинные; алгоритм
завершается на конечном множестве «висящих хвостов» и даёт ответ.

Разбор задачи 4 семинара честно это оговаривал:

> Инъективен: проверено перебором до длины 9 — коллизий нет.
> (Это перебор, а не доказательство; уникальную декодируемость строго
> проверяет алгоритм Сардинаса–Паттерсона.)

Теперь проверяет.

Про **задержку**. Задача 4 просила «инъективный морфизм с максимальной
задержкой», и ответ `h(x) = 0`, `h(y) = 01`, `h(z) = 11` хорош тем, что
задержка у него не ограничена: чтобы отличить `x` от `y` в начале слова,
нужна чётность числа единиц, то есть чтение до конца. Оракул считает
задержку на срезе и печатает кривую роста — по ней и видно, растёт она
или выходит на полку. Доказательством неограниченности кривая не является,
как и всякий срез.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tfl.verdict import Verdict, proved, refuted, unknown

__all__ = ["Code", "Morphism", "Factorization", "parse_morphism"]


@dataclass(frozen=True)
class Factorization:
    """Два разных разбора одного слова — свидетель неоднозначности."""

    word: str
    left: tuple[str, ...]
    right: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"«{self.word}» = "
            + "·".join(self.left)
            + " = "
            + "·".join(self.right)
        )


@dataclass(frozen=True)
class Code:
    """Набор кодовых слов. Пустое слово запрещено — с ним разбор бессмыслен."""

    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(not word for word in self.words):
            raise ValueError("пустое слово кодовым быть не может")
        if len(set(self.words)) != len(self.words):
            raise ValueError("кодовые слова повторяются")

    def __len__(self) -> int:
        return len(self.words)

    def __str__(self) -> str:
        return "{" + ", ".join(self.words) + "}"

    def is_prefix_code(self) -> bool:
        """Ни одно кодовое слово не является префиксом другого.

        Достаточное условие однозначности — и самое дешёвое.
        """
        return not any(
            first != second and second.startswith(first)
            for first in self.words
            for second in self.words
        )

    def is_suffix_code(self) -> bool:
        return not any(
            first != second and second.endswith(first)
            for first in self.words
            for second in self.words
        )

    # ------------------------------------------------- Сардинас–Паттерсон

    def dangling_sets(self) -> list[frozenset[str]]:
        """Множества `S₁, S₂, …` алгоритма Сардинаса–Паттерсона.

        `S₁` — непустые хвосты, на которые одно кодовое слово длиннее
        другого. Дальше `Sₙ₊₁ = C⁻¹Sₙ ∪ Sₙ⁻¹C`: хвост либо укорачивается
        кодовым словом, либо сам укорачивает кодовое слово.

        Все элементы — суффиксы кодовых слов, а их конечное число, поэтому
        последовательность обязана зациклиться. Перечисление ведётся
        до повтора уже виденного множества.
        """
        first = {
            second[len(head) :]
            for head in self.words
            for second in self.words
            if head != second and second.startswith(head)
        }
        found = [frozenset(first)]
        seen = {frozenset(first)}
        while found[-1]:
            current = found[-1]
            following = {
                word[len(head) :]
                for head in self.words
                for word in current
                if word.startswith(head) and word != head
            } | {
                word[len(head) :]
                for head in current
                for word in self.words
                if word.startswith(head) and word != head
            }
            step = frozenset(following)
            found.append(step)
            if step in seen:
                break
            seen.add(step)
        return found

    def uniquely_decodable(self) -> Verdict:
        """Однозначно ли декодируется код. Ответ точный, а не переборный.

        Критерий Сардинаса–Паттерсона: код однозначен тогда и только тогда,
        когда ни одно `Sₙ` при `n ⩾ 1` не содержит кодового слова.

        При отрицательном ответе предъявляется **свидетель** — слово
        с двумя разборами. Он ищется отдельным обходом по тем же висящим
        хвостам: сам критерий говорит «да/нет», а в отчёт нужно слово.
        """
        for index, dangling in enumerate(self.dangling_sets(), start=1):
            common = dangling & set(self.words)
            if common:
                witness = self.ambiguity()
                shown = f": {witness}" if witness else ""
                return refuted(
                    f"код не однозначен: множество S{index} алгоритма "
                    f"Сардинаса–Паттерсона содержит кодовое слово "
                    f"«{sorted(common)[0]}»{shown}",
                    witness,
                )
        return proved(
            f"код однозначно декодируем: ни одно из {len(self.dangling_sets())} "
            f"множеств Сардинаса–Паттерсона не содержит кодового слова"
        )

    def ambiguity(self, max_nodes: int = 200_000) -> Factorization | None:
        """Слово с двумя разными разборами, если оно есть.

        Обход по состоянию «непокрытый хвост и чья сторона впереди» —
        тот же приём, что в `tfl/pcp.py`. Состояний конечное число
        (хвосты — суффиксы кодовых слов), поэтому обход завершается,
        и `None` здесь означает «двух разборов нет».
        """
        start: list[tuple[str, int, tuple[str, ...], tuple[str, ...]]] = []
        for left in self.words:
            for right in self.words:
                if left == right:
                    continue
                if right.startswith(left):
                    start.append((right[len(left) :], -1, (left,), (right,)))
        queue = deque(start)
        seen = {(rest, side) for rest, side, _, _ in start}
        nodes = 0
        while queue:
            rest, side, left_parse, right_parse = queue.popleft()
            for word in self.words:
                if side < 0:
                    # Правая сторона впереди на `rest`; левая догоняет.
                    if word.startswith(rest):
                        grown = word[len(rest) :]
                        mine, yours = (*left_parse, word), right_parse
                        ahead = 1
                    elif rest.startswith(word):
                        grown = rest[len(word) :]
                        mine, yours = (*left_parse, word), right_parse
                        ahead = -1
                    else:
                        continue
                else:
                    if word.startswith(rest):
                        grown = word[len(rest) :]
                        mine, yours = left_parse, (*right_parse, word)
                        ahead = -1
                    elif rest.startswith(word):
                        grown = rest[len(word) :]
                        mine, yours = left_parse, (*right_parse, word)
                        ahead = 1
                    else:
                        continue
                if not grown:
                    return Factorization("".join(mine), mine, yours)
                state = (grown, ahead)
                if state in seen:
                    continue
                nodes += 1
                if nodes >= max_nodes:
                    return None
                seen.add(state)
                queue.append((grown, ahead, mine, yours))
        return None

    # ---------------------------------------------------------- разборы

    def decodings(self, word: str, limit: int = 64) -> list[tuple[str, ...]]:
        """Все разборы слова на кодовые слова."""
        found: list[tuple[str, ...]] = []

        def walk(rest: str, parse: tuple[str, ...]) -> None:
            if len(found) >= limit:
                return
            if not rest:
                found.append(parse)
                return
            for code_word in self.words:
                if rest.startswith(code_word):
                    walk(rest[len(code_word) :], (*parse, code_word))

        walk(word, ())
        return found


# --------------------------------------------------------------------------
# Морфизмы
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Morphism:
    """Гомоморфизм букв в слова: `h(a) = 0`, `h(b) = 01`, …"""

    images: dict[str, str]

    def __str__(self) -> str:
        return "; ".join(
            f"{letter} ↦ {image or 'ε'}" for letter, image in sorted(self.images.items())
        )

    @property
    def source_alphabet(self) -> str:
        return "".join(sorted(self.images))

    def apply(self, word: str) -> str:
        return "".join(self.images[letter] for letter in word)

    def to_code(self) -> Code:
        """Образы букв как код. Морфизм инъективен ⟺ код однозначен.

        Именно это равносильность и делает алгоритм Сардинаса–Паттерсона
        применимым к морфизмам: разбор слова на образы букв — то же самое,
        что разбор на кодовые слова.
        """
        return Code(tuple(dict.fromkeys(self.images[letter] for letter in sorted(self.images))))

    def is_injective(self) -> Verdict:
        """Инъективен ли морфизм. Точно, через Сардинаса–Паттерсона."""
        if len(set(self.images.values())) != len(self.images):
            same = sorted(
                letter for letter in self.images
                if list(self.images.values()).count(self.images[letter]) > 1
            )
            return refuted(
                f"буквы {', '.join(same)} имеют один образ, инъективности нет"
            )
        verdict = self.to_code().uniquely_decodable()
        if verdict.value is True:
            return proved("морфизм инъективен: " + verdict.reason)
        return verdict

    # ------------------------------------------------------------ задержка

    def delay(self, max_len: int = 8) -> tuple[int, tuple[str, str] | None]:
        """Наибольший общий префикс образов слов с разными первыми буквами.

        Это и есть задержка раскодирования: сколько символов кода нужно
        прочесть, чтобы понять первую букву прообраза.
        """
        best, witness = 0, None
        words = _words_upto(self.source_alphabet, max_len)
        for first in words:
            for second in words:
                if not first or not second or first[0] == second[0]:
                    continue
                shared = _common_prefix(self.apply(first), self.apply(second))
                if shared > best:
                    best, witness = shared, (first, second)
        return best, witness

    def delay_growth(self, max_len: int = 8) -> list[int]:
        """Кривая задержки по длине прообраза.

        Читается так же, как кривая роста числа классов в `tfl/lang.py`:
        различать надо **полку и рост**, а не скорость. Вышла на полку —
        задержка, похоже, ограничена; растёт — похоже, нет. Ни то,
        ни другое срезом не доказывается.
        """
        return [self.delay(length)[0] for length in range(1, max_len + 1)]

    def bounded_delay(self, max_len: int = 8) -> Verdict:
        """Ограничена ли задержка — по кривой на срезе.

        Исход всегда `None`: и рост, и полка на конечном срезе остаются
        наблюдением. Вердикт печатает кривую, чтобы наблюдение было видно.
        """
        curve = self.delay_growth(max_len)
        shape = "растёт" if len(set(curve[-3:])) > 1 else "вышла на полку"
        return unknown(
            f"кривая задержки по длине прообраза: {curve} — {shape}. "
            f"Ни рост, ни полка на срезе до {max_len} ничего не доказывают",
            curve,
        )


def parse_morphism(text: str) -> Morphism:
    """Разобрать морфизм из записи `x -> 0; y -> 01; z -> 11`.

    >>> str(parse_morphism("x -> 0; y -> 01"))
    'x ↦ 0; y ↦ 01'
    """
    images: dict[str, str] = {}
    for piece in text.replace("\n", ";").split(";"):
        piece = piece.strip()
        if not piece:
            continue
        for arrow in ("->", "→", "↦", "="):
            if arrow in piece:
                letter, image = piece.split(arrow, 1)
                break
        else:
            raise ValueError(f"в правиле нет стрелки: {piece!r}")
        letter = letter.strip()
        if len(letter) != 1:
            raise ValueError(f"слева от стрелки должна быть одна буква: {piece!r}")
        images[letter] = image.strip().replace("ε", "")
    if not images:
        raise ValueError("морфизм пуст")
    return Morphism(images)


def _common_prefix(first: str, second: str) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def _words_upto(alphabet: str, limit: int) -> list[str]:
    found = [""]
    frontier = [""]
    for _ in range(limit):
        frontier = [word + letter for word in frontier for letter in alphabet]
        found.extend(frontier)
    return found
