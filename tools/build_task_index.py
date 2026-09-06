#!/usr/bin/env python3
"""Сборка размеченного индекса задач из корпуса.

    python tools/build_task_index.py

Читает `corpus/txt/`, режет документы на отдельные задачи и складывает
в `evals/tasks/index.jsonl`. Индекс нужен для двух вещей: искать похожие
задачи с авторскими формулировками (шаг S1 цикла) и оценивать
классификатор на настоящих условиях, а не на выдуманных.

**Разметка выводится из структуры документа, а не угадывается.** Форма
контроля известна из имени файла, а подкласс — из номера задачи в
варианте, но только там, где эта связь проверена:

* РК2, 2023–2025 — задача 1 всегда язык SRS/грамматики, 2 — словесное
  описание языка, 3 — атрибутная грамматика (в каждом третьем пункте
  есть `:=`, проверено на всех вариантах трёх лет);
* экзамен — **разметка позиционная, не содержательная.** Номер вопроса
  в билете задаёт слот, но не жанр: третьим вопросом идёт и теорема
  о замкнутости, и «построить синтаксический моноид», и «построить НКА».
  Такие записи помечены `label: "позиционно"`, и мерить на них точность
  классификатора нельзя — это измерение шума;
* РК1 — **связь не подтвердилась**: в 2025 счётные ограничения стоят
  и первым пунктом (вариант 3), и вторым (вариант 1). Такие задачи
  попадают в индекс с формой контроля, но без подкласса. Выдумывать
  разметку ради красивой цифры точности нельзя — на ней потом будет
  меряться классификатор.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "txt"
OUT = ROOT / "evals" / "tasks" / "index.jsonl"

PAGE = re.compile(r"^-{5} \[page \d+\] -{5}$", re.M)
VARIANT = re.compile(r"^Вариант\s+(\d+)", re.M)
ITEM = re.compile(r"^(\d+)\.\s", re.M)
TICKET = re.compile(r"ЭКЗАМЕНАЦИОННЫЙ БИЛЕТ\s*[ќ№]?\s*(\d+)")
SECTION = re.compile(r"^([A-FА-Е])\.\s+(.+)$", re.M)
POINTS = re.compile(r"\((\d+)\s*балл")

# Хвост экзаменационного билета — реквизиты кафедры, к задаче отношения не имеют.
FOOTER = re.compile(r"Билет рассмотрен.*", re.S)
HEADER_LINE = re.compile(r"^по дисциплине.*$", re.M)


def normalize(text: str) -> str:
    """Убрать разметку страниц и склеить переносы.

    В извлечённом тексте перенос слова остаётся дефисом на конце строки
    (`рас-\\nпознающий`). Без склейки любое сравнение текстов ломается
    на словах, которые просто не туда перенесли.
    """
    text = PAGE.sub("", text)
    text = re.sub(r"-\n(?=[а-яё])", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_items(block: str) -> dict[int, str]:
    """Разрезать блок по нумерованным пунктам `1.`, `2.`, …"""
    parts = ITEM.split(block)[1:]
    return {
        int(parts[i]): normalize(parts[i + 1]).strip()
        for i in range(0, len(parts) - 1, 2)
    }


def split_variants(text: str) -> list[tuple[int, dict[int, str]]]:
    """Разрезать документ на варианты, каждый — на нумерованные задачи."""
    numbers = [int(m.group(1)) for m in VARIANT.finditer(text)]
    blocks = VARIANT.split(text)[2::2]
    return [(n, split_items(b)) for n, b in zip(numbers, blocks)]


# --------------------------------------------------------------------------
# Разметка
# --------------------------------------------------------------------------

SLUG = {"РК1": "rk1", "РК2": "rk2"}
RK2_BY_POSITION = {1: "RK2-A", 2: "RK2-B", 3: "RK2-C"}
EXAM_BY_POSITION = {1: "EXAM-1", 2: "EXAM-2", 3: "EXAM-3"}


def rk_records(path: pathlib.Path, form: str, year: int) -> list[dict]:
    """Задачи РК. Подкласс ставится только там, где связь с номером проверена."""
    text = normalize(path.read_text(encoding="utf-8"))
    trusted = form == "РК2" and year >= 2023
    out = []
    for variant, items in split_variants(text):
        for number, body in items.items():
            if not body or number > 4:
                continue
            out.append(
                {
                    "id": f"{SLUG[form]}-{year}-v{variant:02d}-t{number}",
                    "source": path.name,
                    "year": year,
                    "form": form,
                    "variant": variant,
                    "number": number,
                    "code": RK2_BY_POSITION.get(number) if trusted else None,
                    "label": "проверено" if trusted else None,
                    "text": body,
                }
            )
    return out


def exam_records(path: pathlib.Path, year: int) -> list[dict]:
    """Билеты. В одном файле их несколько — по билету на страницу."""
    text = normalize(path.read_text(encoding="utf-8"))
    marks = list(TICKET.finditer(text))
    out = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        ticket = int(mark.group(1))
        body = FOOTER.sub("", text[mark.end() : end])
        body = HEADER_LINE.sub("", body.strip())
        for number, task in split_items(body).items():
            if task and number <= 3:
                out.append(
                    {
                        "id": f"exam-{year}-b{ticket:02d}-q{number}",
                        "source": path.name,
                        "year": year,
                        "form": "экзамен",
                        "variant": ticket,
                        "number": number,
                        "code": EXAM_BY_POSITION.get(number),
                        "label": "позиционно",
                        "text": task,
                    }
                )
    return out


def pharma_records(path: pathlib.Path, year: int) -> list[dict]:
    """Задачи «Аптеки»: разделы A–F, у каждой задачи своя цена в баллах."""
    text = normalize(path.read_text(encoding="utf-8"))
    out = []
    marks = list(SECTION.finditer(text))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        letter, title = mark.group(1), mark.group(2).strip()
        for number, body in split_items(text[mark.end() : end]).items():
            points = POINTS.search(body)
            out.append(
                {
                    "id": f"pharma-{year}-{letter}{number}",
                    "source": path.name,
                    "year": year,
                    "form": "Аптека",
                    "section": f"{letter}. {title}",
                    "variant": None,
                    "number": number,
                    "points": int(points.group(1)) if points else None,
                    "code": "PHARMA",
                    "label": "проверено",
                    "text": body,
                }
            )
    return out


def lab_records(path: pathlib.Path, year: int, lab: int) -> list[dict]:
    """Условие лабораторной целиком: оно одно на всех, варианты — только данные."""
    text = normalize(path.read_text(encoding="utf-8"))
    head = re.split(r"^Индивидуальные варианты|^\d+\s*$", text, flags=re.M)[0]
    return [
        {
            "id": f"lab{lab}-{year}",
            "source": path.name,
            "year": year,
            "form": f"ЛР{lab}",
            "variant": None,
            "number": None,
            "code": f"LAB-{lab}" if lab <= 4 else None,
            "label": "проверено" if lab <= 4 else None,
            "text": head.strip()[:4000],
        }
    ]


def collect() -> list[dict]:
    records: list[dict] = []
    for path in sorted(CORPUS.glob("*.txt")):
        name = path.name
        year_match = re.search(r"(20\d\d)", name)
        year = int(year_match.group(1)) if year_match else 2025

        if re.search(r"rk1_tfl(_20\d\d)?\.txt$|rk1_tfl_homework", name):
            records += rk_records(path, "РК1", year)
        elif re.search(r"rk2_tfl(_20\d\d)?\.txt$", name):
            records += rk_records(path, "РК2", year)
        elif "TFL_exam" in name:
            records += exam_records(path, year)
        elif "Pharma" in name or "Big_Pharma" in name:
            records += pharma_records(path, year)
        elif (lab := re.search(r"lab_tfl(?:_20\d\d)?_(\d)\.txt$", name)) :
            records += lab_records(path, year, int(lab.group(1)))
    return [r for r in records if len(r["text"]) > 20]


def main() -> None:
    records = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_form: dict[str, int] = {}
    checked = positional = 0
    for record in records:
        by_form[record["form"]] = by_form.get(record["form"], 0) + 1
        checked += record.get("label") == "проверено"
        positional += record.get("label") == "позиционно"
    print(f"записей: {len(records)}, разметка проверенная: {checked}, "
          f"позиционная: {positional}")
    for form, count in sorted(by_form.items()):
        print(f"  {form:10s} {count}")
    print(f"индекс: {OUT}")


if __name__ == "__main__":
    main()
