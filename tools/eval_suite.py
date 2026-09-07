"""Прогон набора для оценки: что оракул закрывает, а что нет.

    python tools/eval_suite.py                 # таблица по классам
    python tools/eval_suite.py --class RK2-B   # только один класс
    python tools/eval_suite.py --report        # записать evals/coverage_report.md

Смысл — не в зелёных галочках. Каждое утверждение вида «класс закрыт»
из `docs/OPEN-GAPS.md` здесь становится исполняемой проверкой с одним
из трёх исходов, и доля `вручную` показывает, где агент по-прежнему
только помощник.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.suite import CASES, MANUAL, PARTIAL, SOLVED  # noqa: E402

MARK = {SOLVED: "+", PARTIAL: "~", MANUAL: "-", "ошибка": "!"}


def run(selected=None):
    """Прогнать набор. Возвращает список `(case, статус, деталь, секунды)`."""
    results = []
    for case in CASES:
        if selected and case.task_class != selected and case.id != selected:
            continue
        start = time.time()
        status, detail = case.evaluate()
        results.append((case, status, detail, time.time() - start))
    return results


def summary(results) -> dict[str, dict[str, int]]:
    """Сводка по классам: сколько чего в каждом."""
    table: dict[str, dict[str, int]] = {}
    for case, status, _detail, _spent in results:
        row = table.setdefault(case.task_class, {SOLVED: 0, PARTIAL: 0, MANUAL: 0, "ошибка": 0})
        row[status] = row.get(status, 0) + 1
    return table


def _print(results) -> None:
    for case, status, detail, spent in results:
        expected = "" if status == case.expected else f"  ⚠ ожидалось «{case.expected}»"
        print(f"{MARK.get(status, '?')} [{case.task_class:6}] {case.id:24} {status:9} "
              f"{spent:5.2f}s{expected}")
        print(f"      {case.statement}")
        print(f"      → {detail}")
    print()
    table = summary(results)
    width = max(len(name) for name in table) if table else 6
    print(f"{'класс':<{width}}  решено  частично  вручную  ошибок")
    for name in sorted(table):
        row = table[name]
        print(
            f"{name:<{width}}  {row[SOLVED]:^6}  {row[PARTIAL]:^8}  "
            f"{row[MANUAL]:^7}  {row['ошибка']:^6}"
        )
    total = {key: sum(row[key] for row in table.values()) for key in (SOLVED, PARTIAL, MANUAL, "ошибка")}
    print(
        f"{'всего':<{width}}  {total[SOLVED]:^6}  {total[PARTIAL]:^8}  "
        f"{total[MANUAL]:^7}  {total['ошибка']:^6}"
    )


def markdown(results) -> str:
    """Отчёт в том виде, в каком он лежит в `evals/coverage_report.md`."""
    table = summary(results)
    lines = [
        "# Покрытие: куда приходит оракул по каждому классу задач",
        "",
        "Считается `python tools/eval_suite.py --report`. Исходы:",
        "",
        "* **решено** — оракул выдаёт ответ на вопрос условия;",
        "* **частично** — необходимое условие, свидетель либо проверка",
        "  предъявленного объекта, но не ответ целиком;",
        "* **вручную** — исполняемой части нет.",
        "",
        "Граница между «решено» и «частично» проведена по **вопросу условия**,",
        "а не по объёму отработавшего кода: проверка предъявленной интерпретации —",
        "это «частично», потому что придумать её всё равно надо самому.",
        "",
        "| класс | решено | частично | вручную | ошибок |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(table):
        row = table[name]
        lines.append(
            f"| `{name}` | {row[SOLVED]} | {row[PARTIAL]} | {row[MANUAL]} | {row['ошибка']} |"
        )
    total = {
        key: sum(row[key] for row in table.values())
        for key in (SOLVED, PARTIAL, MANUAL, "ошибка")
    }
    lines.append(
        f"| **всего** | **{total[SOLVED]}** | **{total[PARTIAL]}** | "
        f"**{total[MANUAL]}** | **{total['ошибка']}** |"
    )

    lines += ["", "## По задачам", ""]
    for case, status, detail, spent in results:
        points = f", {case.points} балл(ов)" if case.points else ""
        lines += [
            f"### `{case.id}` — {case.statement}",
            "",
            f"*{case.task_class}, {case.source}{points}.* **{status}** ({spent:.2f} с)",
            "",
            f"> {detail}",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="selected", help="класс задач либо id случая")
    parser.add_argument(
        "--report",
        nargs="?",
        const="evals/coverage_report.md",
        help="записать отчёт в файл",
    )
    arguments = parser.parse_args()

    results = run(arguments.selected)
    _print(results)
    mismatched = [case.id for case, status, _, _ in results if status != case.expected]
    if mismatched:
        print(f"\nразошлись с ожиданием: {', '.join(mismatched)}")
    if arguments.report:
        Path(arguments.report).write_text(markdown(results), encoding="utf-8")
        print(f"\nотчёт записан: {arguments.report}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
