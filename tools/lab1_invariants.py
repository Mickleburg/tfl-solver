"""Замер: у скольких вариантов ЛР1 есть инвариант, видящий порядок букв.

Условие ЛР1 просит «как минимум два разных инварианта», а за нетривиальные —
те, что «нельзя увидеть по SRS за пять минут внимательного рассматривания» —
обещает **до 2 дополнительных баллов**. Счётные инварианты вида
`Σ cₓ·|w|ₓ mod m` видно сразу; вопрос в том, часто ли находится что-то
сверх них.

Нетривиальность здесь понимается механически и проверяется точно:
инвариант **не сводится к счёту букв** тогда и только тогда, когда
различает два слова с одинаковым набором букв.

Перебор тяжёлый (сотни тысяч сопоставлений на вариант), поэтому он вынесен
сюда, а не в набор для оценки: `evals/suite.py` держит быструю проверку
на разобранном примере, а число по всем вариантам берётся отсюда.

Полная таблица считается **минутами**: на трёхбуквенном алфавите один
только перебор матриц `2×2` над `Z₃` — это 531 441 сопоставление, и таких
вариантов больше десятка. Разбор одного варианта — секунды.

```bash
export PYTHONPATH=.
py -3 tools/lab1_invariants.py                 # таблица по всем вариантам
py -3 tools/lab1_invariants.py --variant 20    # подробно по одному
py -3 tools/lab1_invariants.py --cap 4         # быстрее, оценка грубее
```
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import time

from tfl.srs import SRS, parse_srs

#: Что перебираем: имя колонки и как позвать поиск.
FAMILIES = (
    ("T₂", lambda system, limit, cap: system.monoid_invariants(2, limit, cap)),
    ("T₃", lambda system, limit, cap: system.monoid_invariants(3, limit, cap)),
    ("M₂(Z₂)", lambda system, limit, cap: system.matrix_invariants(2, 2, limit, cap)),
    ("M₂(Z₃)", lambda system, limit, cap: system.matrix_invariants(2, 3, limit, cap)),
)


def search(system: SRS, limit: int, cap: int) -> dict[str, list | None]:
    """Все семейства для одной системы. `None` — перебор не влез в бюджет."""
    found: dict[str, list | None] = {}
    for name, call in FAMILIES:
        try:
            found[name] = call(system, limit, cap)
        except ValueError:
            found[name] = None
    return found


def order_sensitive(groups: dict[str, list | None], alphabet: str, cap: int) -> list:
    """Инварианты, различающие слова с одинаковым набором букв."""
    return [
        invariant
        for group in groups.values()
        if group
        for invariant in group
        if invariant.counting_witness(alphabet, cap)
    ]


def table(limit: int = 30, cap: int = 5) -> str:
    names = [name for name, _ in FAMILIES]
    lines = [
        "| вариант | букв | счётные | " + " | ".join(names) + " | видит порядок |",
        "|---" * (len(names) + 4) + "|",
    ]
    total = beyond = 0
    for path in sorted(glob.glob("evals/lab1_2025/*.srs")):
        system = parse_srs(pathlib.Path(path).read_text(encoding="utf-8"))
        alphabet = "".join(sorted(system.alphabet))
        counting = sum(len(system.linear_invariants(m)) for m in (2, 3, 5, 7))
        groups = search(system, limit, cap)
        sharp = order_sensitive(groups, alphabet, cap)
        total += 1
        beyond += 1 if sharp else 0
        cells = [
            "—" if groups[name] is None else str(len(groups[name])) for name in names
        ]
        lines.append(
            f"| {path[-6:-4]} | {len(alphabet)} | {counting} | "
            + " | ".join(cells)
            + f" | {'да' if sharp else 'нет'} |"
        )
    lines.append("")
    lines.append(
        f"Вариантов {total}, с инвариантом, видящим порядок букв: **{beyond}**. "
        f"Параметры: до {limit} инвариантов на семейство, оценка на словах длины ⩽ {cap}."
    )
    return "\n".join(lines)


def detail(number: int, limit: int, cap: int) -> str:
    path = pathlib.Path(f"evals/lab1_2025/variant-{number:02d}.srs")
    system = parse_srs(path.read_text(encoding="utf-8"))
    alphabet = "".join(sorted(system.alphabet))
    lines = [f"# ЛР1 2025, вариант {number}", "", f"Алфавит: `{alphabet}`", ""]
    for modulus in (2, 3, 5, 7):
        found = system.linear_invariants(modulus)
        if found:
            lines.append(f"* счётные по модулю {modulus}: {found}")
    groups = search(system, limit, cap)
    for name, group in groups.items():
        if group is None:
            lines.append(f"* {name}: перебор не влез в бюджет")
            continue
        if not group:
            lines.append(f"* {name}: инвариантов нет")
            continue
        lines.append(f"* {name}: найдено {len(group)}, лучший —")
        best = group[0]
        lines.append(f"  * `{best}`")
        witness = best.counting_witness(alphabet, cap)
        lines.append(
            f"  * различает `{witness[0]}` и `{witness[1]}` — набор букв один"
            if witness
            else "  * от счётного на этом срезе неотличим"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, help="разобрать один вариант подробно")
    parser.add_argument("--limit", type=int, default=30, help="сколько инвариантов брать")
    parser.add_argument("--cap", type=int, default=5, help="длина слов для оценки")
    args = parser.parse_args()

    started = time.time()
    if args.variant is not None:
        print(detail(args.variant, args.limit, args.cap))
    else:
        print(table(args.limit, args.cap))
    print(f"\n_посчитано за {time.time() - started:.0f} с_", file=sys.stderr)


if __name__ == "__main__":
    main()
