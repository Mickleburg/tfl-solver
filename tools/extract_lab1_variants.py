#!/usr/bin/env python3
"""Извлечение 28 вариантов ЛР1 2025 из текстового зеркала PDF.

    python tools/extract_lab1_variants.py

Варианты в PDF свёрстаны в две колонки, причём номер варианта стоит
вертикально по центру своего блока — в `pdftotext -layout` часть правил
оказывается выше собственного номера и «прилипает» к предыдущему варианту.
PyMuPDF восстанавливает порядок чтения правильно, поэтому парсер работает
по `corpus/txt/`, а не по сырому PDF.

Формат зеркала после номера варианта — плоская последовательность троек:

    20.
    cb
    →
    ba
    ...

Контроль: вариант 20 сверен вручную с `boomhaa-tfl-labs/lab1/report_lab1.md`,
и результат извлечения обязан совпасть с ним до символа.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "corpus" / "txt" / "FormalLanguageTheory_lab_tfl_2025_1.txt"
OUTDIR = ROOT / "evals" / "lab1_2025"
EXPECTED_VARIANTS = 28

VARIANT_RE = re.compile(r"^(\d{1,2})\.$")
ARROW = "→"
EPSILON = "ε"


def parse_variants(text: str) -> dict[int, list[tuple[str, str]]]:
    """Разобрать секцию индивидуальных вариантов в правила."""
    body = text.split("Индивидуальные варианты", 1)[-1]
    tokens = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.startswith("-----")
    ]

    variants: dict[int, list[tuple[str, str]]] = {}
    current: int | None = None
    pending: list[str] = []

    for token in tokens:
        match = VARIANT_RE.match(token)
        # Номер страницы — тоже голое число, но без точки, так что не спутать.
        if match and int(match.group(1)) <= EXPECTED_VARIANTS:
            current = int(match.group(1))
            variants.setdefault(current, [])
            pending = []
            continue
        if current is None:
            continue  # преамбула до первого варианта
        pending.append(token)
        # Тройка «lhs → rhs» собрана
        if len(pending) == 3 and pending[1] == ARROW:
            lhs, _, rhs = pending
            variants[current].append((lhs, "" if rhs == EPSILON else rhs))
            pending = []
        elif len(pending) > 3:
            # Рассинхронизация: сдвигаем окно, чтобы не потерять поток
            pending = pending[-2:]

    return variants


def write_fixtures(variants: dict[int, list[tuple[str, str]]]) -> list[pathlib.Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    written = []
    for number, rules in sorted(variants.items()):
        lines = [
            f"# ЛР1 2025, вариант {number} (FormalLanguageTheory/lab_tfl_2025_1.pdf)",
            f"# Извлечено tools/extract_lab1_variants.py, правил: {len(rules)}",
        ]
        lines += [f"{lhs} -> {rhs or EPSILON}" for lhs, rhs in rules]
        target = OUTDIR / f"variant-{number:02d}.srs"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"нет текстового зеркала: {SOURCE}\nзапустите tools/extract_corpus.py")
    variants = parse_variants(SOURCE.read_text(encoding="utf-8"))

    missing = [n for n in range(1, EXPECTED_VARIANTS + 1) if not variants.get(n)]
    if missing:
        print(f"ВНИМАНИЕ: не извлечены варианты {missing}", file=sys.stderr)

    written = write_fixtures(variants)
    print(f"вариантов: {len(written)} -> {OUTDIR}")
    for number, rules in sorted(variants.items()):
        letters = sorted({c for lhs, rhs in rules for c in lhs + rhs})
        print(f"  {number:>2}: правил {len(rules):>2}, алфавит {''.join(letters)}")


if __name__ == "__main__":
    main()
