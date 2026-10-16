"""Регрессии версионируемого текстового зеркала официальных материалов."""

from __future__ import annotations

import csv
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_lecture_4_2026_is_indexed_and_has_all_reviewed_pages():
    with (ROOT / "corpus/INDEX.tsv").open(encoding="utf-8", newline="") as source:
        matching = [row for row in csv.DictReader(source, delimiter="\t")
                    if row["txt"] == "FormalLanguageTheory_Lecture4_Trace.txt"]
    assert len(matching) == 1
    assert matching[0]["kind"] == "TEXT" and matching[0]["pages"] == "11"
    mirror = (ROOT / "corpus/txt" / matching[0]["txt"]).read_text(encoding="utf-8")
    assert int(matching[0]["chars"]) == len(mirror)
    assert mirror.count("----- [page ") == 11
    for topic in ("Алгебра Клини", "ACI-упрощение", "Автоматное отношение",
                  "Прямое произведение", "Асинхронная композиция",
                  "Символические автоматы", "Древесные автоматы"):
        assert topic in mirror
    notes = (ROOT / "docs/LECTURE-4-NOTES.md").read_text(encoding="utf-8")
    assert "полную конкатенацию" in notes and "{X^nY^n" in notes


def test_lecture_2_2026_is_indexed_and_contains_its_key_topics():
    with (ROOT / "corpus" / "INDEX.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))

    matching = [
        row for row in rows
        if row["txt"] == "FormalLanguageTheory_Lecture2_Trace.txt"
    ]
    assert matching == [
        {
            "kind": "TEXT",
            "pages": "14",
            "chars": "5479",
            "txt": "FormalLanguageTheory_Lecture2_Trace.txt",
            "pdf": "references\\teacher\\FormalLanguageTheory\\Lecture2_Trace.pdf",
        }
    ]

    mirror = (
        ROOT / "corpus" / "txt" / "FormalLanguageTheory_Lecture2_Trace.txt"
    ).read_text(encoding="utf-8")
    assert mirror.count("----- [page ") == 14
    for topic in (
        "Автомат Мили",
        "Графы Кэли полугруппы",
        "Трансформационный моноид конечного автомата",
        "Синтаксический моноид языка",
        "Теорема Майхилла",
        "Нероде",
        "Префиксное переписывание",
    ):
        assert topic in mirror


def test_lecture_3_2026_is_indexed_and_contains_its_key_topics():
    with (ROOT / "corpus" / "INDEX.tsv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))

    matching = [
        row for row in rows
        if row["txt"] == "FormalLanguageTheory_Lecture3_Trace.txt"
    ]
    assert matching == [
        {
            "kind": "TEXT",
            "pages": "20",
            "chars": "9468",
            "txt": "FormalLanguageTheory_Lecture3_Trace.txt",
            "pdf": "references\\teacher\\FormalLanguageTheory\\Lecture3_Trace.pdf",
        }
    ]

    mirror = (
        ROOT / "corpus" / "txt" / "FormalLanguageTheory_Lecture3_Trace.txt"
    ).read_text(encoding="utf-8")
    assert mirror.count("----- [page ") == 20
    for topic in (
        "Префиксная грамматика",
        "Отношения Грина",
        "Лемма Грина о сдвигах",
        "Лемма Ардена",
        "Производная Брзозовски",
        "Автомат Брзозовски",
        "Алгебра Клини",
        "ACI-упрощение",
    ):
        assert topic in mirror
