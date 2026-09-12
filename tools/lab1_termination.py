"""Замер: чем закрывается завершимость на 28 вариантах ЛР1 2025.

Порядок попыток тот же, в каком их стоит делать руками, и каждая
следующая дороже предыдущей:

1. **петля** `w →⁺ u·w·v` — опровержение, и оно дешёвое;
2. **армейский порядок** — работает, пока правила не удлиняют слово;
3. **линейная интерпретация** `x ↦ mx + c` — берёт часть удлиняющих;
4. **пары зависимостей** — правила достаточно уронить нестрого,
   строго — по одной паре на циклическую компоненту (`tfl/deppair.py`);
5. **матричная интерпретация** — многомерная мера (`tfl/matrix.py`);
6. **удаление правил** — все правила роняются нестрого, строго хотя бы
   одно, оно выбрасывается, и всё повторяется на остатке
   (`tfl/removal.py`; шаги берутся и матричные, и арктические);
7. **ограничение совпадениями** — регулярный язык слов с высотами
   вместо меры (`tfl/matchbound.py`), пробуется и на зеркале системы.

Отдельная строка — **полный перебор по длинам**. У не удлиняющей системы
длина слова не растёт, поэтому бесконечный вывод обязан застрять на одной
длине и дать цикл; слов этой длины конечное число, значит «циклов нет» —
доказанный факт, а не «не нашли». Вердикт всё равно неполный: он про слова
до потолка длины, а не про все.

Шаги 1–3 сидят внутри `SRS.terminates`, остальные требуют Z3 (кроме 7)
и потому вызываются отдельно.

```bash
export PYTHONPATH=.
py -3 tools/lab1_termination.py              # вся таблица
py -3 tools/lab1_termination.py --variant 8  # разобрать один вариант
py -3 tools/lab1_termination.py --dimension 2 --ceiling 2
```

Прогон всей таблицы идёт **десятками минут**: на неудачных вариантах Z3
упирается в отведённый ему потолок времени по нескольку раз.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import time

from tfl.srs import SRS, parse_srs


def classify(system: SRS, dimension: int, ceiling: int, timeout: int):
    """Чем закрывается система: метод, пояснение и само доказательство."""
    basic = system.terminates()
    if basic.value is False:
        return "петля", " → ".join(basic.witness), None
    if basic.value is True:
        return "порядок/интерпретация", basic.reason, None

    pairs = system.prove_by_dependency_pairs(1, ceiling, timeout)
    if pairs.value is True:
        return "пары зависимостей", pairs.reason, pairs
    for wider in range(2, dimension + 1):
        wide = system.prove_by_dependency_pairs(wider, ceiling, timeout)
        if wide.value is True:
            return f"пары зависимостей, d={wider}", wide.reason, wide

    matrix = system.find_matrix_interpretation(2, 3, timeout)
    if matrix.value is True:
        return "матричная интерпретация", matrix.reason, None

    removal = system.prove_by_removal((1, 2, 3), 3, timeout)
    if removal.value is True:
        return "удаление правил", removal.reason, removal

    for bound in (1, 2, 3):
        match = system.prove_by_match_bound(bound, 250, 40, timeout_s=20)
        if match.value is True:
            return "ограничение совпадениями", match.reason, match

    # Ничего не доказано целиком — но у не удлиняющей системы можно
    # доказать хотя бы кусок, и это лучше пустого «не выяснено».
    if not system.is_lengthening():
        partial = system.terminates_by_exhaustion(12)
        if partial.value is False:
            return "петля", " → ".join(partial.witness), None
        return "перебор по длинам", partial.reason, None
    return "не выяснено", pairs.reason, None


def table(dimension: int, ceiling: int, timeout: int) -> str:
    lines = [
        "| # | правил | алфавит | пар | компонент | чем закрыто |",
        "|---|---|---|---|---|---|",
    ]
    tally: dict[str, int] = {}
    for path in sorted(glob.glob("evals/lab1_2025/*.srs")):
        number = int(path[-6:-4])
        system = parse_srs(pathlib.Path(path).read_text(encoding="utf-8"))
        pairs = system.dependency_pairs()
        from tfl.deppair import cycles, defined_symbols, estimated_graph

        graph = estimated_graph(pairs, defined_symbols(system))
        started = time.time()
        method, _, _ = classify(system, dimension, ceiling, timeout)
        tally[method] = tally.get(method, 0) + 1
        print(f"  вариант {number:2d}: {method} ({time.time() - started:.0f} с)",
              file=sys.stderr, flush=True)
        lines.append(
            f"| {number} | {len(system.rules)} | `{''.join(sorted(system.alphabet))}` "
            f"| {len(pairs)} | {len(cycles(graph))} | {method} |"
        )
    lines.append("")
    for method, count in sorted(tally.items(), key=lambda item: -item[1]):
        lines.append(f"* {method}: **{count}**")
    return "\n".join(lines)


def detail(number: int, dimension: int, ceiling: int, timeout: int) -> str:
    from tfl.deppair import cycles, defined_symbols, estimated_graph

    path = pathlib.Path(f"evals/lab1_2025/variant-{number:02d}.srs")
    system = parse_srs(path.read_text(encoding="utf-8"))
    pairs = system.dependency_pairs()
    graph = estimated_graph(pairs, defined_symbols(system))
    parts = cycles(graph)
    lines = [
        f"# ЛР1 2025, вариант {number}",
        "",
        f"Правила: {'; '.join(str(rule) for rule in system.rules)}",
        "",
        f"Определённые буквы: `{''.join(sorted(defined_symbols(system)))}`",
        f"Пар зависимостей: {len(pairs)}, циклических компонент: {len(parts)} "
        f"(размеры {[len(part) for part in parts]})",
        f"Длину слова система {'удлиняет' if system.is_lengthening() else 'не удлиняет'}",
        "",
    ]
    method, reason, verdict = classify(system, dimension, ceiling, timeout)
    lines.append(f"**Чем закрыто:** {method} — {reason}")
    witness = getattr(verdict, "witness", None)
    if witness is not None and hasattr(witness, "markdown"):
        lines.append("")
        lines.append(witness.markdown())
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, help="разобрать один вариант")
    parser.add_argument("--dimension", type=int, default=2, help="размерность пары")
    parser.add_argument("--ceiling", type=int, default=4, help="потолок элементов")
    parser.add_argument("--timeout", type=int, default=20_000, help="мс на компоненту")
    args = parser.parse_args()

    started = time.time()
    if args.variant is not None:
        print(detail(args.variant, args.dimension, args.ceiling, args.timeout))
    else:
        print(table(args.dimension, args.ceiling, args.timeout))
    print(f"\n_посчитано за {time.time() - started:.0f} с_", file=sys.stderr)


if __name__ == "__main__":
    main()
