"""Приём задачи: что в ней дано, что спрашивают, на что она похожа.

Это шаг S0–S1 цикла из `docs/03-AGENT-PLAN.md`. На вход — текст задачи
как его прислал пользователь, на выход — разбор, по которому выбирается
рецепт и метод.

Что модуль делает и чего **не** делает.

Делает: находит в тексте формальные объекты (правила SRS, КС-грамматику,
атрибутную грамматику, регулярное выражение, описание языка в фигурных
скобках, счётчики подслов) и глаголы вопроса («проверить на регулярность»,
«построить минимальный», «завершима ли»), а затем ищет в корпусе похожие
условия с авторскими разборами.

Не делает: не выносит вердикт о классе. `classify` возвращает **ранжированный
список с уликами**, а не ответ; окончательное решение принимает тот, кто
читает задачу, и цена ошибки тут высока — неверно выбранный класс уводит
в неверный метод целиком.

Отдельно: **форма контроля (РК/экзамен/лаба) из текста задачи, как правило,
не выводится.** Одна и та же формулировка «проверить язык на регулярность»
встречается в разных работах. Поэтому форма берётся из пометки пользователя,
а не угадывается; без пометки она просто неизвестна, и это не мешает выбрать
метод. В курсе 2026 номер лабораторной ещё не определяет исторический рецепт.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

from tfl.hints import Hint, match as match_hints

__all__ = [
    "Evidence",
    "Candidate",
    "Analysis",
    "analyse",
    "classify",
    "find_features",
    "find_asks",
    "TaskIndex",
    "load_index",
    "FEATURES",
    "ASKS",
]

INDEX_PATH = pathlib.Path(__file__).resolve().parent.parent / "evals" / "tasks" / "index.jsonl"


# --------------------------------------------------------------------------
# Признаки: что в задаче дано
# --------------------------------------------------------------------------

#: Формальные объекты. Ключ — имя признака, значение — как его опознать.
FEATURES: dict[str, re.Pattern[str]] = {
    # `S′0.a := T.a + S′1.a` — присваивание атрибуту. Самый надёжный признак
    # во всём наборе, но `:=` обязано быть одиночным: в условиях лаб формат
    # входных данных задают в БНФ, где `::=` встречается на каждой строке.
    # Второй вариант ловит ссылку на атрибут (`S2.attr`, `T.free_a`, `S'.a`)
    # там, где присваивание записано через `=`, а не `:=` — так пишут
    # от руки в работах. Отрицательный просмотр назад отсекает
    # µ-выражения (`µY.bX` из экзаменационных билетов), где точка значит
    # совсем другое.
    "атрибутная грамматика": re.compile(
        r"(?<!:):=|(?<![µμ])[A-Z]['′]?[0-9]?\.[A-Za-z_][A-Za-z_0-9]*"
    ),
    # Правило переписывания: обе части — цепочки строчных букв.
    "правила SRS": re.compile(r"(?<![A-ZА-Я])[a-z][a-z0-9²³]*\s*(?:→|->)\s*[a-z]"),
    # Правило грамматики: слева заглавная буква (возможно со штрихом).
    "КС-грамматика": re.compile(r"[A-Z]['′]?\s*(?:→|->)"),
    "базис": re.compile(r"\bбазис"),
    # Описание языка множеством: `{w | условие}`.
    "описание языка множеством": re.compile(r"\{[^}]*[|∣][^}]*\}|Язык\s*\n?\s*[{⟨]"),
    "счётчики подслов": re.compile(r"\|\s*w[0-9₀-₉]*\s*\|\s*[a-z]|\|w\|"),
    # В извлечённом тексте фигурные скобки описания языка нередко теряются:
    # в LaTeX они набраны крупными разделителями и приходят управляющими
    # символами. Поэтому описание опознаётся ещё и по своей начинке.
    "принадлежность алфавиту": re.compile(r"∈\s*[{(\[]"),
    "конъюнкция условий": re.compile(r"&|∨|∃|∀"),
    # `(?<![A-Z])` отсекает «SLR», «LR», «LL»: там R стоит после заглавной,
    # и без этого всякое условие ЛР5 объявлялось задачей про реверс.
    "реверс": re.compile(r"(?<![A-Z])[a-zA-Zω]\s*R\b|wR|\bреверс|обращени"),
    "регулярное выражение": re.compile(r"[a-z()][*∗+]|\([^)]*\|[^)]*\)"),
    "язык пар": re.compile(r"x1\s*\n?\s*y1|xi,\s*yi|верхняя строка"),
    "атрибуты как условие": re.compile(r"\.[ab]\s*[><]\s*\w+\.[ab]"),
    "накачка": re.compile(r"накачк|лемм[аеы] о разраст"),
    "цена в баллах": re.compile(r"\(\d+\s*балл"),
    "числа в позиционной записи": re.compile(r"двоичн|троичн|десятичн|делящ|кратн"),
    "палиндром": re.compile(r"палиндром"),
    "скобочная последовательность": re.compile(r"скобоч|сбалансирован|скобк"),
}

#: Что именно спрашивают. Глагол вопроса определяет метод не меньше, чем объект.
ASKS: dict[str, re.Pattern[str]] = {
    "регулярность": re.compile(r"регуляр", re.I),  # «регулярен», «регулярна» тоже
    "КС-свойство": re.compile(r"\bКС\b|контекстно-свободн|контекстно свободн"),
    "детерминизм": re.compile(r"детерминиз|детерминирован|DPDA"),
    "LL-свойство": re.compile(r"\bLL\b|LL\(|LL-язык"),
    "минимальность": re.compile(r"минимальн"),
    "завершимость": re.compile(r"завершим|заверша\w*\s+ли|\bтерминир"),
    "замкнутость класса": re.compile(r"замкнут"),
    "беспрефиксность": re.compile(r"беспрефиксн|префиксн"),
    "построить распознаватель": re.compile(r"построить\s+(мин\w*\s+)?(ДКА|НКА|КА|автомат|PDA|грамматик|регул)"),
    # Семинары: кодировки, морфизмы, задержка раскодирования.
    "кодировка": re.compile(
        r"кодиру|кодировк|раскодир|декодир|морфизм|инъективн|задержк\w* раскод",
        re.I,
    ),
    # Правила с переменной: `aXb → bXa`.
    "образец с переменной": re.compile(r"образц|образец|[a-z][A-Z][a-z]\s*(->|→)"),
    # ЛР5 2023: разбор с гиперстеком, все пути сразу.
    "недетерминированный разбор": re.compile(
        r"графовидн\w* стек|древовидн\w* стек|гиперстек|Generic\s+(SLR|LL|LR)|"
        r"GLR|Томит|лес разбора|SPPF|запакованн\w* лес",
        re.I,
    ),
    # ЛР2 2024 и ЛР3 2023: угадывание автомата запросами к учителю.
    "активное обучение": re.compile(
        r"МАТ|минимально адекватн|угадыват|запрос\w* (об|о) (эквивалентност|включени)|"
        r"L\*|NL\*|активн\w+ обучени",
        re.I,
    ),
    "место в иерархии": re.compile(r"иерархи|класс языка|к какому классу"),
    "эквивалентность": re.compile(r"эквивалентн"),
    "проверка готового решения": re.compile(
        r"найти\s+(?:перв\w+\s+)?ошибк|указать\s+ошибк|"
        r"ошибк\w*\s+в\s+(?:готов\w+\s+)?(?:решени|доказательств)|"
        r"проверить\s+(?:готовое\s+)?решение|нейрослоп",
        re.I,
    ),
    "проанализировать": re.compile(r"проанализировать|исследовать|проверить|является ли|задаёт ли|задајт ли"),
    # В РК2 вопрос не задают глаголом: условие — именная группа
    # «Язык SRS …», «Язык {…}», «Язык, определяемый атрибутной грамматикой».
    # Спрашивают при этом всегда одно: описать язык и поместить его
    # в иерархию классов.
    "описать язык": re.compile(r"(?:^|\n)\s*(?:\d[.)]\s*)?Язык[\s,]", re.M),
}

# Глагол вопроса часто открывает предложение («Детерминирован ли язык?»),
# поэтому регистр решать не должен. Флаг дописывается здесь, а не в каждой
# строке: так его не забудешь у очередной записи.
ASKS = {
    name: re.compile(pattern.pattern, pattern.flags | re.I)
    for name, pattern in ASKS.items()
}


@dataclass(frozen=True)
class Evidence:
    """Найденная улика: имя признака и кусок текста, где он найден."""

    name: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.name}: «{self.excerpt}»"


def _found(text: str, table: dict[str, re.Pattern[str]]) -> list[Evidence]:
    out = []
    for name, pattern in table.items():
        match = pattern.search(text)
        if match:
            excerpt = match.group(0).strip().replace("\n", " ")
            out.append(Evidence(name, excerpt[:60]))
    return out


def find_features(text: str) -> list[Evidence]:
    """Формальные объекты, встречающиеся в тексте задачи."""
    return _found(text, FEATURES)


def find_asks(text: str) -> list[Evidence]:
    """Вопросы, которые задача задаёт."""
    return _found(text, ASKS)


# --------------------------------------------------------------------------
# Классы задач
# --------------------------------------------------------------------------

#: Вес признака для класса. Отрицательный вес — признак против класса.
#: Веса подобраны по размеченной части индекса (`evals/tasks/index.jsonl`),
#: точность измеряется в `tests/test_intake.py`, а не заявляется.
WEIGHTS: dict[str, dict[str, float]] = {
    "RK2-C": {"атрибутная грамматика": 10, "атрибуты как условие": 3, "КС-грамматика": 1},
    "RK2-A": {"правила SRS": 4, "базис": 6, "КС-грамматика": 2,
              "атрибутная грамматика": -10},
    "RK2-B": {"описание языка множеством": 4, "счётчики подслов": 3, "реверс": 2,
              "принадлежность алфавиту": 3, "конъюнкция условий": 2,
              "место в иерархии": 2, "атрибутная грамматика": -10, "базис": -4},
    "RK1-A": {"язык пар": 8, "числа в позиционной записи": 4, "регулярность": 1,
              "атрибутная грамматика": -10},
    "RK1-B": {"счётчики подслов": 4, "регулярность": 2, "описание языка множеством": 1,
              "принадлежность алфавиту": 2, "конъюнкция условий": 1,
              "атрибутная грамматика": -10, "базис": -4},
    "RK1-C": {"палиндром": 5, "скобочная последовательность": 4,
              "регулярное выражение": 2, "регулярность": 2,
              "атрибутная грамматика": -10},
    "LAB-1": {"правила SRS": 4, "завершимость": 5, "базис": -2},
    "LAB-2": {"регулярное выражение": 3, "минимальность": 3,
              "построить распознаватель": 2},
    "LAB-3": {"детерминизм": 5, "беспрефиксность": 4, "КС-грамматика": 2,
              "LL-свойство": 2},
    "LAB-4": {"КС-грамматика": 1, "КС-свойство": 2},
    "LAB-5": {"недетерминированный разбор": 9, "КС-грамматика": 1},
    "MAT": {"активное обучение": 9, "построить распознаватель": 1},
    "CODE": {"кодировка": 8, "образец с переменной": 6},
    "EXAM-1": {"регулярность": 2, "КС-свойство": 2, "описание языка множеством": 1},
    "EXAM-2": {"LL-свойство": 4, "детерминизм": 2, "КС-грамматика": 2},
    "EXAM-3": {"замкнутость класса": 6, "завершимость": 3},
    "EXAM-ERROR": {"проверка готового решения": 12},
    "PHARMA": {"цена в баллах": 8},
}

#: Какие классы возможны при данной пометке пользователя.
CURRENT_LAB_CLASSES = ("LAB-1", "LAB-2", "LAB-3", "LAB-4", "LAB-5", "MAT")
CURRENT_RK_CLASSES = (
    "RK1-A", "RK1-B", "RK1-C", "RK2-A", "RK2-B", "RK2-C"
)
BY_FORM: dict[str, tuple[str, ...]] = {
    "РК": CURRENT_RK_CLASSES,
    "РК1": ("RK1-A", "RK1-B", "RK1-C"),
    "РК2": ("RK2-A", "RK2-B", "RK2-C"),
    "экзамен": ("EXAM-1", "EXAM-2", "EXAM-3", "EXAM-ERROR"),
    "экзамен-ошибка": ("EXAM-ERROR",),
    "ЛР": CURRENT_LAB_CLASSES,
    "ЛР1": CURRENT_LAB_CLASSES,
    "ЛР2": CURRENT_LAB_CLASSES,
    "ЛР3": CURRENT_LAB_CLASSES,
    "ЛР4": CURRENT_LAB_CLASSES,
    "ЛР5": ("LAB-5",),
    "МАТ": ("MAT",),
    "семинар": ("CODE",),
    "Аптека": ("PHARMA",),
}


@dataclass
class Candidate:
    """Класс-кандидат с уликами, по которым он предложен."""

    code: str
    score: float
    evidence: tuple[Evidence, ...] = ()

    def __str__(self) -> str:
        marks = ", ".join(e.name for e in self.evidence)
        return f"{self.code} ({self.score:.0f}): {marks}" if marks else f"{self.code} ({self.score:.0f})"


def classify(text: str, hint: str | None = None, limit: int = 3) -> list[Candidate]:
    """Ранжированный список классов-кандидатов с уликами.

    `hint` — пометка пользователя о форме контроля (`«РК»`, `«ЛР»`,
    `«экзамен»`). Она сужает набор классов, но не подменяет разбор. Номера
    лабораторных 2026 не привязаны к старым содержательным кодам, поэтому
    `ЛР1`…`ЛР4` оставляют доступными все исторические lab-рецепты.

    Возвращается именно список: пустой результат означает «признаков
    не хватило», а один кандидат с большим отрывом — совсем не то же
    самое, что два с равным счётом.
    """
    marks = {e.name: e for e in find_features(text) + find_asks(text)}
    allowed = BY_FORM.get(hint or "", tuple(WEIGHTS))

    scored: list[Candidate] = []
    for code in allowed:
        weights = WEIGHTS.get(code, {})
        score = sum(w for name, w in weights.items() if name in marks)
        if score <= 0:
            continue
        used = tuple(
            marks[name] for name, w in sorted(weights.items(), key=lambda kv: -kv[1])
            if name in marks and w > 0
        )
        scored.append(Candidate(code, score, used))

    scored.sort(key=lambda c: (-c.score, c.code))
    return scored[:limit]


# --------------------------------------------------------------------------
# Поиск похожих задач
# --------------------------------------------------------------------------

TOKEN = re.compile(r"[а-яёa-z]+|\d+|[|&∗*+∈≥≤→↔¬∨∧]")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class TaskIndex:
    """Индекс условий из корпуса с поиском по косинусной близости.

    Мера — TF-IDF по словам и математическим значкам. Ничего умнее не нужно:
    задача не в семантическом поиске, а в том, чтобы найти вариант того же
    жанра, к которому есть авторский разбор.
    """

    records: list[dict]
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _vectors: list[dict[str, float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        documents = [tokenize(r["text"]) for r in self.records]
        seen: Counter[str] = Counter()
        for tokens in documents:
            seen.update(set(tokens))
        total = len(documents) or 1
        self._idf = {t: math.log(total / (1 + n)) + 1 for t, n in seen.items()}
        self._vectors = [self._vector(tokens) for tokens in documents]

    def __len__(self) -> int:
        return len(self.records)

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        raw = {t: (1 + math.log(n)) * self._idf.get(t, 1.0) for t, n in counts.items()}
        norm = math.sqrt(sum(v * v for v in raw.values())) or 1.0
        return {t: v / norm for t, v in raw.items()}

    def similar(
        self,
        text: str,
        limit: int = 5,
        form: str | None = None,
        exclude: str | None = None,
    ) -> list[tuple[float, dict]]:
        """Похожие условия, от самого близкого. `exclude` — id, который пропустить."""
        query = self._vector(tokenize(text))
        scored: list[tuple[float, dict]] = []
        for vector, record in zip(self._vectors, self.records):
            if form and record["form"] != form:
                continue
            if exclude and record["id"] == exclude:
                continue
            shared = set(query) & set(vector)
            if not shared:
                continue
            scored.append((sum(query[t] * vector[t] for t in shared), record))
        scored.sort(key=lambda pair: -pair[0])
        return scored[:limit]


def load_index(path: pathlib.Path | None = None) -> TaskIndex:
    """Загрузить индекс, собранный `tools/build_task_index.py`."""
    source = path or INDEX_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"нет {source}; соберите индекс: py -3 tools/build_task_index.py"
        )
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return TaskIndex(records)


# --------------------------------------------------------------------------
# Сводный разбор
# --------------------------------------------------------------------------


@dataclass
class Analysis:
    """Всё, что удалось понять о задаче до начала решения."""

    text: str
    hint: str | None
    features: tuple[Evidence, ...]
    asks: tuple[Evidence, ...]
    candidates: tuple[Candidate, ...]
    similar: tuple[tuple[float, dict], ...] = ()
    hints: tuple[Hint, ...] = ()

    @property
    def confident(self) -> bool:
        """Есть ли отрыв у лидера.

        Без отрыва выбирать класс автоматически нельзя: разница между
        `RK1-B` и `RK2-B` — это разница между «доказать нерегулярность»
        и «определить место в иерархии», и метод у них разный.
        """
        if not self.candidates:
            return False
        if len(self.candidates) == 1:
            return True
        return self.candidates[0].score >= self.candidates[1].score + 3

    def report(self) -> str:
        lines = ["## Разбор задачи", ""]
        lines.append(f"Пометка пользователя: {self.hint or '—'}")
        lines.append("")
        if self.hint in {"ЛР1", "ЛР2", "ЛР3", "ЛР4"}:
            lines.append(
                "> В курсе 2026 номер лабораторной не выбирает старый рецепт: "
                "маршрут определяется полным условием."
            )
            lines.append("")
        if self.hint in {"ЛР5", "Аптека"}:
            lines.append(
                "> Эта форма архивная для курса 2026; материал используется "
                "только как источник методов и регрессий."
            )
            lines.append("")
        lines.append("**Что дано:** " + (", ".join(e.name for e in self.features) or "не опознано"))
        lines.append("")
        lines.append("**Что спрашивают:** " + (", ".join(e.name for e in self.asks) or "не опознано"))
        lines.append("")
        lines.append("**Кандидаты:**")
        lines.append("")
        if not self.candidates:
            lines.append("* признаков не хватило — класс определяйте вручную")
        for candidate in self.candidates:
            lines.append(f"* {candidate}")
        lines.append("")
        if not self.confident and self.candidates:
            lines.append("> Отрыва у лидера нет: класс не определён, "
                         "выбирать метод по этому списку нельзя.")
            lines.append("")
        if self.hints:
            lines.append("**Структурные образцы** (гипотезы, не вердикты):")
            lines.append("")
            for hint in self.hints:
                lines.append(f"* {hint}")
                if hint.note:
                    lines.append(f"  * оговорка: {hint.note}")
            lines.append("")
        if self.similar:
            lines.append("**Похожие условия из корпуса:**")
            lines.append("")
            for score, record in self.similar:
                head = record["text"].replace("\n", " ")[:90]
                lines.append(f"* `{record['id']}` ({score:.2f}, {record['form']}) — {head}…")
            lines.append("")
        return "\n".join(lines)


def analyse(
    text: str, hint: str | None = None, index: TaskIndex | None = None, limit: int = 5
) -> Analysis:
    """Полный приём задачи: признаки, вопросы, кандидаты, похожие условия."""
    similar: tuple[tuple[float, dict], ...] = ()
    if index is not None:
        similar = tuple(index.similar(text, limit=limit))
    return Analysis(
        text=text,
        hint=hint,
        features=tuple(find_features(text)),
        asks=tuple(find_asks(text)),
        candidates=tuple(classify(text, hint)),
        similar=similar,
        hints=tuple(match_hints(text)),
    )
