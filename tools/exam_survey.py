"""Что на самом деле спрашивают в экзаменационных билетах.

В `evals/tasks/index.jsonl` вопросы билета размечены **позиционно**:
первый — `EXAM-1`, второй — `EXAM-2`, третий — `EXAM-3`. Разметка честно
помечена `label: "позиционно"`, и этот скрипт проверяет, насколько она
соответствует содержанию.

Классификация здесь — по тексту вопроса, а не по номеру. Правила
упорядочены: сначала опознаются формулировки, которые ни с чем не путаются
(«Построить синтаксический моноид», «Всегда ли завершается переписывание»),
и только потом — общие («Проверить язык на регулярность»). Обратный порядок
ломается сразу: теоретический вопрос «Верно ли, что конкатенация двух
LL(1)-языков всегда детерминирована?» содержит и `LL`, и «детерминирова»,
и по одним лишь ключевым словам неотличим от разбора конкретной грамматики.

    py -3 tools/exam_survey.py            # сводка: тема × позиция
    py -3 tools/exam_survey.py --list КС  # выписать вопросы одной темы
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = ROOT / "evals" / "tasks" / "index.jsonl"

#: Тема → как её опознать. **Порядок значим**, см. модульную строку.
RULES: list[tuple[str, re.Pattern[str]]] = [
    # 1. Конструктивные задания: узнаются по объекту, который просят построить.
    ("моноид", re.compile(r"синтаксическ\w+ моноид|моноид\w* (?:трансформац|перехо)")),
    ("конъюнктивная грамматика", re.compile(r"конъюнктивн")),  # в одном билете опечатка «грамматиу»
    ("ПКА", re.compile(r"переключающ\w+ автомат|ПКА|И-недетерминизм")),
    ("µ-выражение", re.compile(r"µ-?выражени|мю-выражени")),
    ("проблема Поста", re.compile(r"соответстви\w+ Поста|ПКП")),
    ("длина накачки", re.compile(r"длин\w+ накачки")),
    ("атрибутная грамматика", re.compile(r"атрибутн\w+ грамматик")),
    ("таблица классов", re.compile(r"таблиц\w+ классов")),
    ("НКА для пересечения", re.compile(r"НКА (?:для|слов)")),
    ("к академической", re.compile(r"академическ")),
    ("морфизм", re.compile(r"морфизм|прообраз")),
    # 2. Завершимость SRS: ловится до теоретических «всегда ли», иначе
    #    «Всегда ли завершается переписывание» уедет в замкнутость.
    ("завершимость SRS", re.compile(r"заверш\w+ (?:перепис|по SRS)|SRS на заверш|на заверш\w+:")),
    # 3. Теоретические вопросы о классах: узнаются по форме вопроса,
    #    а не по терминам внутри — термы там те же, что в разборе задачи.
    ("замкнутость классов", re.compile(
        r"^(?:Верно ли|Может ли|Всегда ли|Если |Пусть |Итерацией|Левое частное|Обязательно ли)")),
    # 4. Конструктивные задания про распознаватель — после теоретических,
    #    потому что «Может ли … ДКА» это вопрос, а не построение.
    ("построить ДКА", re.compile(r"[Пп]остроить (?:минимальный )?ДКА|выглядит (?:минимальный )?ДКА")),
    ("построить регулярку", re.compile(r"[Пп]остроить регулярное выражение|дополнение к регулярному")),
    # 5. Аналитические вопросы о свойстве предъявленного языка.
    ("LL/LR-свойство", re.compile(r"\bLL\b|LL\(|LL-|\bLR\b|LR\(")),
    ("детерминированность", re.compile(r"детерминиз|детерминирован")),
    ("КС-свойство", re.compile(r"\bКС\b|контекстн\w+[ -]свобод|контекстную свободу")),
    ("регулярность", re.compile(r"регулярн|регулярен")),
]


def topic(text: str) -> str:
    """Тема вопроса по его тексту. `?` — правила не сработали."""
    flat = " ".join(text.split())
    for name, pattern in RULES:
        if pattern.search(flat):
            return name
    return "?"


def load() -> list[dict]:
    if not INDEX.exists():
        sys.exit(f"нет индекса {INDEX}; сначала py -3 tools/build_task_index.py")
    rows = [json.loads(line) for line in INDEX.open(encoding="utf-8")]
    return [r for r in rows if r.get("form") == "экзамен"]


def survey(rows: list[dict]) -> None:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        table[topic(row["text"])][row["number"]] += 1

    order = sorted(table, key=lambda t: -sum(table[t].values()))
    width = max(len(t) for t in order)
    print(f"{'тема':<{width}} | вопрос 1 | вопрос 2 | вопрос 3 | всего")
    print("-" * (width + 42))
    for name in order:
        counts = table[name]
        total = sum(counts.values())
        print(f"{name:<{width}} | {counts[1]:^8} | {counts[2]:^8} | {counts[3]:^8} | {total:^5}")

    # Насколько позиция предсказывает тему: берём для каждой позиции самую
    # частую тему и считаем, сколько вопросов в неё не попало.
    print()
    for number in (1, 2, 3):
        column = collections.Counter(topic(r["text"]) for r in rows if r["number"] == number)
        total = sum(column.values())
        top, hits = column.most_common(1)[0]
        print(f"вопрос {number}: тем {len(column)}, самая частая «{top}» — "
              f"{hits} из {total} ({hits / total:.0%})")
    unknown = [r for r in rows if topic(r["text"]) == "?"]
    print(f"\nне классифицировано: {len(unknown)} из {len(rows)}")
    for row in unknown:
        print("   ", row["id"], " ".join(row["text"].split())[:90])


def listing(rows: list[dict], needle: str) -> None:
    for row in rows:
        name = topic(row["text"])
        if needle.lower() in name.lower():
            print(f"--- {row['id']} [{name}]")
            print(" ".join(row["text"].split()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", metavar="ТЕМА", help="выписать вопросы одной темы")
    args = parser.parse_args()
    rows = load()
    if args.list:
        listing(rows, args.list)
    else:
        survey(rows)


if __name__ == "__main__":
    main()
