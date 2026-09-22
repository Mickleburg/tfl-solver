"""Регрессии версионируемого текстового зеркала официальных материалов."""

from __future__ import annotations

import csv
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent


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
