#!/usr/bin/env python3
"""Приём задачи: разбор условия, кандидаты в классы, похожие задачи из корпуса.

    python tools/intake.py --text "Язык SRS с правилами aa →aba, базис (ab)*"
    python tools/intake.py --file task.txt --hint РК2
    cat task.txt | python tools/intake.py --hint экзамен

Пометка `--hint` (`РК1`, `РК2`, `ЛР1`…`ЛР4`, `экзамен`, `Аптека`) сужает
набор классов. Она не обязательна: без неё разбор всё равно покажет, какие
объекты в задаче и что спрашивают, — а именно это определяет метод.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tfl.intake import BY_FORM, analyse, load_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="текст задачи прямо в командной строке")
    source.add_argument("--file", help="файл с текстом задачи")
    parser.add_argument("--hint", choices=sorted(BY_FORM), help="форма контроля")
    parser.add_argument("--limit", type=int, default=5, help="сколько похожих задач показать")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = pathlib.Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise SystemExit("пустой текст задачи")

    print(analyse(text, args.hint, load_index(), args.limit).report())


if __name__ == "__main__":
    main()
