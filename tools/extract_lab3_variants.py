#!/usr/bin/env python3
"""Извлечение вариантов ЛР3 2025 из текстового зеркала PDF.

Варианты чередуются: нечётные — КС-грамматики, чётные — словесные описания
языка. Грамматики сохраняются в `evals/lab3_2025/variant-NN.cfg` и годятся
как вход для `tfl.cfg.parse_cfg`; описания складываются в один файл
`languages.md` — их всё равно предстоит формализовать вручную.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "corpus" / "txt" / "FormalLanguageTheory_2025_lab_tfl_2025_3.txt"
OUTDIR = ROOT / "evals" / "lab3_2025"
MARKER = re.compile(r"^(\d{1,2})\.\s*(.*)$")
ARROW = "→"


def main() -> None:
    body = SOURCE.read_text(encoding="utf-8").split("Индивидуальные варианты", 1)[-1]
    grammars: dict[int, list[str]] = {}
    languages: dict[int, str] = {}
    current: int | None = None

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("-----"):
            continue
        match = MARKER.match(line)
        if match and int(match.group(1)) <= 28:
            current = int(match.group(1))
            tail = match.group(2).strip()
            if tail:
                languages[current] = tail  # словесное описание на той же строке
                current = None
            else:
                grammars[current] = []
            continue
        if current is None:
            if languages:  # продолжение описания, перенесённое на след. строку
                last = max(languages)
                languages[last] += " " + line
            continue
        if ARROW in line:
            grammars[current].append(line)
        else:
            current = None  # номер страницы или мусор — блок закончился

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for number, rules in sorted(grammars.items()):
        header = [
            f"# ЛР3 2025, вариант {number} (FormalLanguageTheory/lab_tfl_2025_3.pdf)",
            f"# Извлечено tools/extract_lab3_variants.py, правил: {len(rules)}",
        ]
        (OUTDIR / f"variant-{number:02d}.cfg").write_text(
            "\n".join(header + rules) + "\n", encoding="utf-8"
        )
    lines = ["# ЛР3 2025: варианты со словесным описанием языка", ""]
    for number, text in sorted(languages.items()):
        lines.append(f"{number}. {text}")
    (OUTDIR / "languages.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"грамматик: {len(grammars)}, описаний языка: {len(languages)} -> {OUTDIR}")
    for number, rules in sorted(grammars.items()):
        print(f"  {number:>2}: правил {len(rules)}")


if __name__ == "__main__":
    main()
