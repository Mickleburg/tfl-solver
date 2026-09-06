"""Системы переписывания строк (SRS): переписывание, завершимость,
критические пары, пополнение по Кнуту–Бендиксу.

Обслуживает ЛР1 целиком, задачу 1 в РК2 («язык SRS над базисом») и третий
вопрос экзамена («всегда ли завершается переписывание»).

Что здесь принципиально: почти все свойства SRS неразрешимы в общем случае,
поэтому каждая функция честно разделяет три исхода — **доказано**,
**опровергнуто** и **не удалось выяснить в пределах бюджета**. Молчаливое
превращение третьего во второй — самый простой способ написать уверенно
выглядящий неверный отчёт.

Так, `terminates()` не возвращает `bool`: она возвращает вердикт с
обоснованием — либо найденный фундированный порядок, либо конкретный цикл
переписывания, либо признание, что бюджет исчерпан.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from tfl.words import iter_words

__all__ = [
    "Rule",
    "SRS",
    "Search",
    "parse_srs",
    "Verdict",
    "shortlex_key",
    "ARROWS",
]

# Стрелки, встречающиеся в условиях и чужих отчётах.
# Порядок важен: длинные стрелки проверяются первыми, иначе "->" найдётся
# внутри "-->" и разрежет правило не в том месте.
ARROWS = ("-->", "->", "→", "=>")
# Обозначения пустого слова во входных файлах.
EPSILON_TOKENS = {"ε", "eps", ".", "", "\\varepsilon", "λ"}


@dataclass(frozen=True, order=True)
class Rule:
    """Правило переписывания `lhs → rhs`."""

    lhs: str
    rhs: str

    def __post_init__(self) -> None:
        if not self.lhs:
            raise ValueError("левая часть правила не может быть пустой")

    def __str__(self) -> str:
        return f"{self.lhs} → {self.rhs or 'ε'}"

    @property
    def reversed(self) -> Rule:
        return Rule(self.rhs, self.lhs) if self.rhs else self

    def positions(self, word: str) -> Iterator[int]:
        """Все позиции вхождения левой части (редексы), включая наложения."""
        start = 0
        while True:
            idx = word.find(self.lhs, start)
            if idx < 0:
                return
            yield idx
            start = idx + 1

    def apply_at(self, word: str, pos: int) -> str:
        return word[:pos] + self.rhs + word[pos + len(self.lhs) :]


def shortlex_key(word: str, precedence: str) -> tuple[int, tuple[int, ...]]:
    """Ключ «армейского» (длина, затем лексикографика) порядка.

    Это тотальный фундированный порядок, согласованный с контекстом, поэтому
    убывание всех правил по нему **доказывает** завершимость. `precedence`
    задаёт старшинство букв, например `"abc"` означает a < b < c.
    """
    rank = {ch: i for i, ch in enumerate(precedence)}
    return (len(word), tuple(rank.get(ch, len(rank)) for ch in word))


@dataclass
class Verdict:
    """Результат проверки, которая может и не завершиться выводом.

    `value` — True (доказано), False (опровергнуто) или None (не выяснено
    в пределах бюджета). `witness` — то, что предъявляется в отчёте:
    цикл, критическая пара, порядок.
    """

    value: bool | None
    reason: str
    witness: object = None

    def __bool__(self) -> bool:  # pragma: no cover - защита от опечатки
        raise TypeError(
            "Verdict нельзя использовать как bool: у него три исхода. "
            "Проверяйте .value is True / is False / is None"
        )

    def __str__(self) -> str:
        mark = {True: "да", False: "нет", None: "не выяснено"}[self.value]
        return f"{mark}: {self.reason}"


@dataclass
class Search:
    """Результат обхода отношения достижимости.

    `exact` означает, что обход не обрезался ни по длине слова, ни по их
    числу, то есть множество вычислено целиком. Без этого флага пустое
    пересечение двух множеств ничего не доказывает.
    """

    words: set[str]
    exact: bool

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)


@dataclass
class SRS:
    """Система переписывания строк."""

    rules: tuple[Rule, ...]
    alphabet: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        if not self.alphabet:
            letters: set[str] = set()
            for rule in self.rules:
                letters |= set(rule.lhs) | set(rule.rhs)
            object.__setattr__(self, "alphabet", frozenset(letters))

    def __len__(self) -> int:
        return len(self.rules)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.rules)

    # ---------------------------------------------------------------- шаги

    def redexes(self, word: str) -> list[tuple[Rule, int]]:
        """Все применимые пары (правило, позиция)."""
        return [(rule, pos) for rule in self.rules for pos in rule.positions(word)]

    def step(self, word: str) -> set[str]:
        """Все слова, получаемые за один шаг переписывания."""
        return {rule.apply_at(word, pos) for rule, pos in self.redexes(word)}

    def is_normal_form(self, word: str) -> bool:
        # Осторожно: `any(rule.positions(w) ...)` здесь не работает — positions
        # возвращает генератор, а объект генератора истинен всегда.
        return not any(rule.lhs in word for rule in self.rules)

    def reachable(
        self, word: str, max_len: int = 24, max_words: int = 20000
    ) -> Search:
        """Слова, достижимые за любое число шагов.

        Ограничения по длине и числу слов обязательны: отношение достижимости
        в незавершимой системе бесконечно. Но тогда пустое пересечение двух
        таких множеств **не** доказывает несходимость — путь мог уходить через
        слова длиннее лимита.

        Поэтому результат помечается флагом `exact`: он выставляется, только
        если обход ни разу не упёрся в лимит и, значит, множество достижимых
        слов вычислено целиком. Вся разница между «не сходятся» и «не удалось
        свести» держится на этом флаге.
        """
        seen = {word}
        stack = [word]
        truncated = False
        while stack:
            if len(seen) >= max_words:
                truncated = True
                break
            for nxt in self.step(stack.pop()):
                if len(nxt) > max_len:
                    truncated = True
                    continue
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return Search(seen, not truncated)

    def normal_forms(
        self, word: str, max_len: int = 24, max_words: int = 20000
    ) -> Search:
        """Нормальные формы слова (в незавершимой системе может быть пусто)."""
        search = self.reachable(word, max_len, max_words)
        return Search({w for w in search.words if self.is_normal_form(w)}, search.exact)

    def random_chain(
        self, word: str, steps: int, rng: random.Random, max_len: int = 30
    ) -> list[str]:
        """Случайная цепочка переписываний — основа фазз-теста из ЛР1."""
        chain = [word]
        current = word
        for _ in range(steps):
            options = [
                (rule, pos)
                for rule, pos in self.redexes(current)
                if len(current) - len(rule.lhs) + len(rule.rhs) <= max_len
            ]
            if not options:
                break
            rule, pos = rng.choice(options)
            current = rule.apply_at(current, pos)
            chain.append(current)
        return chain

    # -------------------------------------------------------- завершимость

    def decreasing_under(self, precedence: str) -> Verdict:
        """Убывают ли все правила по «армейскому» порядку.

        Если да — система завершима, и это полноценное доказательство:
        шортлекс фундирован и согласован с контекстом.
        """
        bad = [
            rule
            for rule in self.rules
            if shortlex_key(rule.lhs, precedence) <= shortlex_key(rule.rhs, precedence)
        ]
        if bad:
            return Verdict(
                None,
                f"по порядку «{precedence}» не убывают правила: "
                + ", ".join(str(r) for r in bad)
                + ". Это не доказывает незавершимость — возможно, нужен другой порядок",
                witness=bad,
            )
        return Verdict(
            True,
            f"все правила убывают по армейскому порядку с приоритетом «{precedence}» "
            f"(длина, затем лексикографика) — порядок фундирован, значит система завершима",
            witness=precedence,
        )

    def find_cycle(self, max_len: int = 10, start_len: int = 8) -> list[str] | None:
        """Найти цикл переписывания `w →⁺ w` — свидетельство незавершимости.

        Перебираются стартовые слова до длины `start_len`; в обходе слова
        длиннее `max_len` отбрасываются. Ненайденный цикл ничего не доказывает:
        незавершимость бывает и без циклов (растущие слова).
        """
        letters = sorted(self.alphabet)
        for start in iter_words(letters, start_len):
            path: list[str] = []
            on_path: set[str] = set()
            explored: set[str] = set()

            def dfs(word: str) -> list[str] | None:
                if word in on_path:
                    return path[path.index(word) :] + [word]
                if word in explored or len(word) > max_len:
                    return None
                path.append(word)
                on_path.add(word)
                for nxt in sorted(self.step(word)):
                    found = dfs(nxt)
                    if found is not None:
                        return found
                path.pop()
                on_path.discard(word)
                explored.add(word)
                return None

            cycle = dfs(start)
            if cycle is not None:
                return cycle
        return None

    def terminates(
        self, precedence: str | None = None, max_len: int = 10, start_len: int = 7
    ) -> Verdict:
        """Вердикт о завершимости: порядок, цикл или «не выяснено».

        Сначала ищется цикл (дешёвое опровержение), затем проверяется убывание
        по армейскому порядку — по всем перестановкам алфавита, если приоритет
        не задан явно.
        """
        cycle = self.find_cycle(max_len=max_len, start_len=start_len)
        if cycle is not None:
            return Verdict(
                False,
                "найден цикл переписывания: " + " → ".join(cycle),
                witness=cycle,
            )

        import itertools

        candidates = (
            [precedence]
            if precedence
            else ["".join(p) for p in itertools.permutations(sorted(self.alphabet))]
        )
        for order in candidates:
            verdict = self.decreasing_under(order)
            if verdict.value is True:
                return verdict
        return Verdict(
            None,
            f"цикл длины ≤ {max_len} не найден, но и убывающего армейского порядка "
            f"нет ни при одном приоритете букв. Нужен более сильный порядок "
            f"(рекурсивный по путям, полиномиальная интерпретация) или другой аргумент",
        )

    # ------------------------------------------------------ конфлюэнтность

    def critical_pairs(self) -> list[tuple[str, str, str, Rule, Rule]]:
        """Критические пары: (критическое слово, левый редукт, правый редукт, r1, r2).

        Два источника наложений:

        * **пересечение** — непустой суффикс `l₁` совпадает с префиксом `l₂`;
        * **вложение** — `l₂` целиком входит в `l₁`.

        Мирные (не пересекающиеся) применения правил всегда сходятся, поэтому
        их рассматривать не нужно — это и есть смысл леммы о критических парах.
        """
        pairs: list[tuple[str, str, str, Rule, Rule]] = []
        for r1 in self.rules:
            for r2 in self.rules:
                l1, l2 = r1.lhs, r2.lhs

                # Вложение: l1 = u·l2·v
                for pos in r2.positions(l1):
                    if r1 == r2 and pos == 0:
                        continue  # тривиальное самоналожение
                    left = r1.rhs
                    right = l1[:pos] + r2.rhs + l1[pos + len(l2) :]
                    if left != right:
                        pairs.append((l1, left, right, r1, r2))

                # Пересечение: l1 = u·s, l2 = s·v, s ≠ ε, наложение собственное
                for k in range(1, min(len(l1), len(l2))):
                    if l1[len(l1) - k :] != l2[:k]:
                        continue
                    critical = l1 + l2[k:]
                    left = r1.rhs + l2[k:]
                    right = l1[: len(l1) - k] + r2.rhs
                    if left != right:
                        pairs.append((critical, left, right, r1, r2))
        # Убираем дубликаты, сохраняя порядок
        seen: set[tuple[str, str, str]] = set()
        unique = []
        for critical, left, right, r1, r2 in pairs:
            key = (critical, left, right)
            if key not in seen:
                seen.add(key)
                unique.append((critical, left, right, r1, r2))
        return unique

    def joinable(self, left: str, right: str, max_len: int = 20) -> Verdict:
        """Сходятся ли два слова к общему потомку.

        Три исхода. `True` — общий потомок найден. `False` — множества
        достижимых слов вычислены **целиком** и не пересекаются. `None` —
        пересечения не нашлось, но хотя бы один обход был обрезан лимитом,
        так что вывода нет.
        """
        if left == right:
            return Verdict(True, "слова совпадают")
        left_search = self.reachable(left, max_len)
        right_search = self.reachable(right, max_len)
        common = left_search.words & right_search.words
        if common:
            witness = min(common, key=lambda w: (len(w), w))
            return Verdict(True, f"общий потомок «{witness or 'ε'}»", witness=witness)
        if left_search.exact and right_search.exact:
            return Verdict(
                False,
                f"множества потомков вычислены полностью и не пересекаются",
            )
        return Verdict(
            None,
            f"общий потомок не найден, но обход обрезан лимитом длины {max_len}",
        )

    def locally_confluent(self, max_len: int = 20) -> Verdict:
        """Проверить локальную конфлюэнтность через критические пары."""
        pairs = self.critical_pairs()
        unresolved: list[tuple[str, str, str]] = []
        for critical, left, right, r1, r2 in pairs:
            verdict = self.joinable(left, right, max_len)
            if verdict.value is False:
                return Verdict(
                    False,
                    f"критическая пара не сходится: слово «{critical}» "
                    f"по правилу ({r1}) даёт «{left or 'ε'}», "
                    f"а по правилу ({r2}) — «{right or 'ε'}»; "
                    f"множества потомков вычислены полностью и не пересекаются",
                    witness=(critical, left, right, r1, r2),
                )
            if verdict.value is None:
                unresolved.append((critical, left, right))
        if unresolved:
            critical, left, right = unresolved[0]
            return Verdict(
                None,
                f"{len(unresolved)} из {len(pairs)} критических пар не удалось "
                f"свести в пределах длины {max_len}; первая из них — «{critical}» "
                f"с ветками «{left or 'ε'}» и «{right or 'ε'}». "
                f"Увеличьте max_len либо разбирайте вручную",
                witness=unresolved,
            )
        return Verdict(
            True,
            f"все {len(pairs)} критических пар сходятся",
        )

    # --------------------------------------------------------- Кнут–Бендикс

    def oriented(self, precedence: str) -> SRS:
        """Переориентировать все правила по убыванию армейского порядка."""
        rules: list[Rule] = []
        for rule in self.rules:
            lo, hi = rule.lhs, rule.rhs
            if shortlex_key(lo, precedence) == shortlex_key(hi, precedence):
                continue  # тождество, правило бесполезно
            if shortlex_key(lo, precedence) < shortlex_key(hi, precedence):
                lo, hi = hi, lo
            rules.append(Rule(lo, hi))
        return SRS(tuple(dict.fromkeys(rules)), self.alphabet)

    def complete(
        self, precedence: str, max_rules: int = 60, max_rounds: int = 40, max_len: int = 20
    ) -> tuple[SRS, Verdict]:
        """Пополнение по Кнуту–Бендиксу относительно армейского порядка.

        Возвращает пополненную систему и вердикт: удалось ли достичь
        конфлюэнтности или процедура упёрлась в лимит. Пополнение в принципе
        может не завершаться — это ожидаемый исход, а не ошибка.
        """
        current = self.oriented(precedence)
        for _ in range(max_rounds):
            added: list[Rule] = []
            for critical, left, right, _r1, _r2 in current.critical_pairs():
                if current.joinable(left, right, max_len).value is True:
                    continue
                lnf = min(
                    current.normal_forms(left, max_len).words or {left},
                    key=lambda w: shortlex_key(w, precedence),
                )
                rnf = min(
                    current.normal_forms(right, max_len).words or {right},
                    key=lambda w: shortlex_key(w, precedence),
                )
                if lnf == rnf:
                    continue
                lo, hi = (lnf, rnf)
                if shortlex_key(lo, precedence) < shortlex_key(hi, precedence):
                    lo, hi = hi, lo
                rule = Rule(lo, hi)
                if rule not in current.rules and rule not in added:
                    added.append(rule)
            if not added:
                return current.minimized(precedence, max_len), Verdict(
                    True, "пополнение завершено: все критические пары сходятся"
                )
            current = SRS(tuple(current.rules) + tuple(added), self.alphabet)
            if len(current) > max_rules:
                return current, Verdict(
                    None,
                    f"пополнение прервано: правил стало больше {max_rules}. "
                    f"Процедура Кнута–Бендикса может не завершаться",
                )
        return current, Verdict(
            None, f"пополнение прервано: исчерпан лимит в {max_rounds} раундов"
        )

    def minimized(self, precedence: str, max_len: int = 20) -> SRS:
        """Убрать правила, выводимые из остальных.

        Правило отбрасывается, если его левая и правая части всё ещё сходятся
        без него. Порядок обхода — от «тяжёлых» правил к «лёгким», чтобы
        в системе оставались короткие.
        """
        rules = list(
            sorted(
                dict.fromkeys(self.rules),
                key=lambda r: shortlex_key(r.lhs, precedence),
                reverse=True,
            )
        )
        i = 0
        while i < len(rules):
            trial = SRS(tuple(rules[:i] + rules[i + 1 :]), self.alphabet)
            if trial.rules and trial.joinable(rules[i].lhs, rules[i].rhs, max_len).value is True:
                rules.pop(i)
            else:
                i += 1
        return SRS(tuple(sorted(rules)), self.alphabet)

    # ------------------------------------------- классы эквивалентности

    def equivalence_classes(
        self, max_len: int = 8, slack: int = 2
    ) -> dict[str, str]:
        """Классы по отношению ↔* : слово → представитель класса.

        Правила применяются **в обе стороны** (как требует ЛР1). Обход идёт
        по словам до длины `max_len + slack`, а в результат попадают слова
        до `max_len`: запас нужен, чтобы не потерять связи, идущие в обход
        через более длинные слова. Полной гарантии он не даёт — при слишком
        малом запасе классы могут оказаться раздробленными.
        """
        letters = sorted(self.alphabet)
        limit = max_len + slack
        parent: dict[str, str] = {}

        def find(w: str) -> str:
            parent.setdefault(w, w)
            root = w
            while parent[root] != root:
                root = parent[root]
            while parent[w] != root:
                parent[w], w = root, parent[w]
            return root

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # представителем становится шортлекс-меньшее слово
            if (len(rb), rb) < (len(ra), ra):
                ra, rb = rb, ra
            parent[rb] = ra

        for word in iter_words(letters, limit):
            find(word)
            for nxt in self.step(word):
                if len(nxt) <= limit:
                    union(word, nxt)

        return {w: find(w) for w in iter_words(letters, max_len)}

    def class_count(self, max_len: int = 8, slack: int = 2) -> int:
        return len(set(self.equivalence_classes(max_len, slack).values()))

    # ---------------------------------------------------------- проверки

    def same_equivalence(
        self,
        other: SRS,
        max_len: int = 5,
        search_len: int = 10,
        max_words: int = 50000,
    ) -> Verdict:
        """Систематическая сверка отношений ↔* двух систем — пункт 4 ЛР1.

        Для каждого слова до длины `max_len` строится его класс в обеих
        системах (правила применяются в обе стороны) и классы сравниваются.

        Тонкость, из-за которой наивная проверка врёт: обратные правила
        удлиняют слова, поэтому обход почти всегда упирается в лимит, а
        слово, не найденное в обрезанном обходе, вполне может лежать чуть
        дальше границы. Поэтому вердикт «не эквивалентны» выдаётся, только
        если обход той стороны, которой слова не хватило, был **полным**.

        Короткие стартовые слова тут выгоднее длинных: чем короче слово, тем
        выше шанс, что его класс уместится в лимит целиком и вывод окажется
        доказательным.
        """
        letters = sorted(self.alphabet | other.alphabet)
        left, right = self.symmetric(), other.symmetric()
        inconclusive = 0
        for word in iter_words(letters, max_len):
            ls = left.reachable(word, search_len, max_words)
            rs = right.reachable(word, search_len, max_words)
            for side, (have, lack, text) in enumerate(
                (
                    (ls, rs, ("первой", "второй")),
                    (rs, ls, ("второй", "первой")),
                )
            ):
                extra = sorted(have.words - lack.words, key=lambda w: (len(w), w))
                if not extra:
                    continue
                if lack.exact:
                    return Verdict(
                        False,
                        f"«{word or 'ε'}» и «{extra[0] or 'ε'}» связаны в {text[0]} "
                        f"системе, а в {text[1]} класс «{word or 'ε'}» вычислен "
                        f"полностью и «{extra[0] or 'ε'}» в него не входит",
                        witness=(word, extra[0]),
                    )
                inconclusive += 1
        if inconclusive:
            return Verdict(
                None,
                f"доказательных расхождений нет, но {inconclusive} классов "
                f"не удалось вычислить целиком в пределах длины {search_len}",
            )
        return Verdict(
            True,
            f"классы всех слов до длины {max_len} вычислены полностью и совпали",
        )

    def fuzz_equivalence(
        self,
        other: SRS,
        trials: int = 500,
        chain_len: int = 6,
        word_len: int = 8,
        seed: int = 0,
        search_len: int = 14,
        max_words: int = 50000,
    ) -> Verdict:
        """Фазз-тест ровно по схеме задания: случайное слово ω, случайная
        цепочка переписываний в ω' по `self`, проверка связи ω и ω' в `other`.

        Слабее `same_equivalence` — длинные случайные слова почти гарантируют
        обрезанный обход, — но задание просит именно эту схему, поэтому она
        нужна для отчёта.
        """
        rng = random.Random(seed)
        letters = sorted(self.alphabet)
        symmetric = other.symmetric()
        inconclusive = 0
        for _ in range(trials):
            length = rng.randint(1, word_len)
            word = "".join(rng.choice(letters) for _ in range(length))
            target = self.random_chain(word, chain_len, rng)[-1]
            if target == word:
                continue
            search = symmetric.reachable(word, search_len, max_words)
            if target in search.words:
                continue
            if search.exact:
                return Verdict(
                    False,
                    f"«{word}» переписывается в «{target or 'ε'}» в первой системе, "
                    f"а во второй класс «{word}» вычислен полностью и не содержит его",
                    witness=(word, target),
                )
            inconclusive += 1
        return Verdict(
            None,
            f"{trials} случайных цепочек (seed={seed}) не дали доказательного "
            f"расхождения; из них {inconclusive} остались невыясненными",
        )

    def symmetric(self) -> SRS:
        """Система с правилами в обе стороны — для отношения ↔*."""
        rules = list(self.rules)
        for rule in self.rules:
            if rule.rhs:
                back = Rule(rule.rhs, rule.lhs)
                if back not in rules:
                    rules.append(back)
        return SRS(tuple(rules), self.alphabet)

    def check_invariant(
        self,
        invariant: Callable[[str], object],
        words: Iterable[str],
        monotone: bool = False,
    ) -> Verdict:
        """Метаморфное тестирование: сохраняется ли инвариант при переписывании.

        `monotone=True` допускает невозрастание вместо строгого равенства —
        задание ЛР1 разрешает и такие инварианты.
        """
        for word in words:
            before = invariant(word)
            for nxt in self.step(word):
                after = invariant(nxt)
                ok = (after <= before) if monotone else (after == before)  # type: ignore[operator]
                if not ok:
                    kind = "не убывает" if monotone else "не сохраняется"
                    return Verdict(
                        False,
                        f"инвариант {kind}: «{word}» → «{nxt or 'ε'}», "
                        f"значение {before} → {after}",
                        witness=(word, nxt, before, after),
                    )
        return Verdict(True, "инвариант выдержал все проверенные шаги")


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def parse_srs(text: str, alphabet: str = "") -> SRS:
    """Разобрать SRS из текста: по правилу на строку, `lhs -> rhs`.

    Понимает стрелки `->`, `→`, `-->`, `=>` и обозначения пустого слова
    `ε`, `eps`, `.`, пустую правую часть. Строки, начинающиеся с `#`,
    и пустые игнорируются.

    >>> srs = parse_srs("ab -> ba\\naaa -> a")
    >>> len(srs)
    2
    """
    rules: list[Rule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for arrow in ARROWS:
            if arrow in line:
                lhs, rhs = line.split(arrow, 1)
                break
        else:
            raise ValueError(f"в строке нет стрелки: {raw!r}")
        lhs, rhs = lhs.strip(), rhs.strip()
        if lhs in EPSILON_TOKENS:
            raise ValueError(f"левая часть не может быть пустой: {raw!r}")
        if rhs in EPSILON_TOKENS:
            rhs = ""
        rules.append(Rule(lhs, rhs))
    letters = set(alphabet)
    for rule in rules:
        letters |= set(rule.lhs) | set(rule.rhs)
    return SRS(tuple(rules), frozenset(letters))
