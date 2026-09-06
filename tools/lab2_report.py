#!/usr/bin/env python3
"""Генератор отчёта по ЛР2 (регулярка → ДКА / НКА / ПКА / расширенная регулярка).

    python tools/lab2_report.py 15
    python tools/lab2_report.py --regex "b*((ab*a)*(aabb|(babb)*))*" --out out/

Автоматизируется только механическая часть — та, где есть исполняемый оракул:
минимальный ДКА, таблица классов, малый НКА и оценка его минимальности,
протокол проверок. Пункты, требующие содержательной догадки (переключающийся
автомат и расширенное регулярное выражение), генератор не выдумывает: он
оставляет размеченные заготовки и явно пишет, что осталось сделать руками.

Отчёт, в котором недоказанное выдано за доказанное, хуже отсутствующего —
поэтому все непроверенные места помечаются `[НЕ ПРОВЕРЕНО]`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tfl import regex as rx
from tfl.automata import DFA, NFA, dfa_of, disagreements, thompson
from tfl.glushkov import glushkov, small_nfa
from tfl.myhill import class_table, extended_fooling_set, fooling_set
from tfl.words import count_words, iter_words

ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANTS = ROOT / "evals" / "lab2_2025_variants.txt"
CHECK_LEN = 12  # до какой длины перебираются слова в проверках L1


def load_variant(number: int) -> str:
    for line in VARIANTS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        num, pattern = line.split("\t", 1)
        if int(num) == number:
            return pattern.strip()
    raise SystemExit(f"вариант {number} не найден в {VARIANTS}")


def transition_table(dfa: DFA) -> str:
    letters = sorted(dfa.alphabet)
    lines = ["| состояние | " + " | ".join(letters) + " | финальное |",
             "|---" * (len(letters) + 2) + "|"]
    for state in sorted(dfa.states, key=lambda s: (isinstance(s, str), s)):
        cells = []
        for ch in letters:
            target = dfa.delta.get((state, ch))
            cells.append("—" if target is None else str(target))
        mark = "да" if state in dfa.finals else ""
        start = " (старт)" if state == dfa.start else ""
        lines.append(f"| {state}{start} | " + " | ".join(cells) + f" | {mark} |")
    return "\n".join(lines)


def nfa_transition_table(nfa: NFA) -> str:
    letters = sorted(nfa.alphabet)
    order = {st: i for i, st in enumerate(sorted(nfa.states, key=repr))}
    lines = ["| состояние | " + " | ".join(letters) + " | финальное |",
             "|---" * (len(letters) + 2) + "|"]
    for state in sorted(nfa.states, key=repr):
        cells = []
        for ch in letters:
            targets = sorted(order[t] for t in nfa.delta.get((state, ch), ()))
            cells.append("∅" if not targets else "{" + ",".join(map(str, targets)) + "}")
        mark = "да" if state in nfa.finals else ""
        start = " (старт)" if state == nfa.start else ""
        lines.append(
            f"| {order[state]}{start} | " + " | ".join(cells) + f" | {mark} |"
        )
    return "\n".join(lines)


def build_report(pattern: str, title: str, outdir: pathlib.Path) -> str:
    node = rx.parse(pattern)
    alphabet = sorted(node.alphabet())

    dfa = dfa_of(node)
    table = class_table(dfa)
    positional = glushkov(node)
    nfa = small_nfa(node)
    sym = fooling_set(dfa)
    tri = extended_fooling_set(dfa)
    best = max(sym, tri, key=lambda fs: fs.bound)

    # --- проверки уровня L1 ------------------------------------------------
    thompson_dfa = thompson(node).determinize().minimize()
    constructions_agree = disagreements(
        thompson_dfa.accepts, dfa.accepts, alphabet, CHECK_LEN, limit=1
    )
    nfa_agrees = disagreements(nfa.accepts, dfa.accepts, alphabet, CHECK_LEN, limit=1)
    pos_agrees = disagreements(
        positional.accepts, dfa.accepts, alphabet, CHECK_LEN, limit=1
    )
    checked_words = count_words(alphabet, CHECK_LEN)

    nfa_minimal = best.bound == len(nfa.states)
    examples = [w for w in iter_words(alphabet, 8) if dfa.accepts(w)][:8]

    (outdir / "dfa.dot").write_text(dfa.to_dot("min_dfa"), encoding="utf-8")
    (outdir / "nfa.dot").write_text(nfa.to_dot("nfa"), encoding="utf-8")
    (outdir / "positional.dot").write_text(
        positional.to_dot("positional"), encoding="utf-8"
    )

    md: list[str] = []
    add = md.append

    add(f"# {title}")
    add("")
    add(f"Регулярное выражение: `{pattern}`")
    add("")
    add(f"Алфавит: {{{', '.join(alphabet)}}}. Примеры слов языка: "
        + ", ".join(f"`{w or 'ε'}`" for w in examples) + ".")
    add("")
    add("> Отчёт сгенерирован `tools/lab2_report.py`. Пункты, помеченные")
    add("> `[НЕ ПРОВЕРЕНО]` и `[ВРУЧНУЮ]`, требуют доработки человеком.")
    add("")

    # -- 1. ДКА -------------------------------------------------------------
    add("## 1. Минимальный ДКА")
    add("")
    add(f"Состояний: **{len(dfa)}** (без состояния-ловушки). "
        f"Классов Майхилла–Нероуда с учётом ловушки: **{dfa.class_count()}**.")
    add("")
    add("Граф — `dfa.dot`.")
    add("")
    add(transition_table(dfa))
    add("")
    add("### Таблица классов эквивалентности")
    add("")
    add(f"Строки — кратчайшие представители классов, столбцы — "
        f"различающие суффиксы ({len(table.suffixes)} шт.). "
        f"Все строки попарно различны, поэтому склеить никакие два класса "
        f"нельзя и автомат минимален.")
    add("")
    add(table.to_markdown())
    add("")

    # -- 2. НКА -------------------------------------------------------------
    add("## 2. Возможно малый НКА")
    add("")
    add(f"Построение: позиционный автомат (Глушкова) на "
        f"{len(positional.states)} состояний, затем склейка состояний "
        f"с совпадающим правым языком → **{len(nfa.states)}** состояний. "
        f"Для сравнения, автомат Томпсона дал бы "
        f"{len(thompson(node).states)} состояний.")
    add("")
    add("Граф — `nfa.dot` (позиционный автомат до склейки — `positional.dot`).")
    add("")
    add(nfa_transition_table(nfa))
    add("")
    add("### Оценка минимальности снизу")
    add("")
    add(f"Обманывающее множество ({'треугольная' if best.kind == 'triangular' else 'симметричная'} "
        f"версия теоремы Глайстера–Шаллита) размера **{best.bound}**: "
        f"γ_i ω_i ∈ L, а γ_j ω_i ∉ L при j < i. Значит, ни один НКА для этого "
        f"языка не может иметь меньше {best.bound} состояний.")
    add("")
    add(best.to_markdown())
    add("")
    if nfa_minimal:
        add(f"Оценка снизу ({best.bound}) совпала с числом состояний "
            f"построенного автомата — **НКА минимален**.")
    else:
        add(f"Оценка снизу ({best.bound}) меньше числа состояний "
            f"({len(nfa.states)}), так что минимальность не доказана: "
            f"истинный минимум лежит между этими числами. "
            f"[ВРУЧНУЮ] попробовать сжать автомат дальше либо усилить оценку.")
    add("")

    # -- 3. ПКА -------------------------------------------------------------
    add("## 3. Переключающийся (конъюнктивный) автомат")
    add("")
    add("[ВРУЧНУЮ] **Инвариант** генератор не выдумывает: надо заметить "
        "свойство языка, которое проверяется отдельно от основной структуры. "
        "Преподаватель различает два вида — обычные (проверяются для всей "
        "строки сразу) и рекурсивные (для всех суффиксов после определённого "
        "сочетания букв).")
    add("")
    add("Когда инвариант найден, автомат строится готовой конструкцией "
        "из `tfl/afa.py` — руками его собирать не нужно:")
    add("")
    add("```python")
    add("from tfl.afa import conjunction, with_lookahead, disagreements")
    add("from tfl.automata import dfa_of")
    add("")
    add("# обычный инвариант: И-ветвление в стартовой вершине")
    add('A = conjunction([dfa_of("<условие 1>", alphabet),')
    add('                 dfa_of("<условие 2>", alphabet)], alphabet)')
    add("")
    add("# рекурсивный инвариант: проверка всего остатка слова")
    add('A = with_lookahead(reader=dfa_of("<основная структура>", alphabet),')
    add('                   checker=dfa_of("<условие на остаток>", alphabet))')
    add("")
    add(f'print(A.summary())                       # размеры ПКА / НКА / ДКА')
    add(f'disagreements(A, dfa_of({pattern!r}, alphabet), max_len={CHECK_LEN})')
    add("```")
    add("")
    add("Подробно — `docs/recipes/LAB-2.md`, §4. Произведение автоматов "
        "строить **не надо**: пересечение даёт `|Q₁|·|Q₂|` состояний, "
        "а ПКА на том же условии — `|Q₁| + |Q₂| + 1`, ради этого "
        "конструкция и нужна.")
    add("")

    # -- 4. Расширенная регулярка -------------------------------------------
    add("## 4. Расширенное регулярное выражение")
    add("")
    add("[ВРУЧНУЮ] Требуются маркеры `^` и `$`. Разрешены `.`, `τ+`, `τ?`, "
        "опережающие `(?=…)` / `(?!…)` и ретроспективные `(?<=…)` / `(?<!…)` "
        "проверки, классы символов `[…]`.")
    add("")
    add("Если упрощение сводится к академической регулярке, его можно "
        "проверить машинно:")
    add("")
    add("```python")
    add("from tfl.automata import dfa_of, counterexample")
    add(f'counterexample(dfa_of({pattern!r}), dfa_of("<упрощение>"))  # None = равны')
    add("```")
    add("")

    # -- 5. Протокол проверок ----------------------------------------------
    add("## 5. Протокол проверок (уровень L1)")
    add("")
    add(f"Все слова алфавита {{{', '.join(alphabet)}}} до длины {CHECK_LEN} "
        f"включительно — {checked_words} слов на каждую строку таблицы.")
    add("")
    add("| Что проверялось | Чем | Результат |")
    add("|---|---|---|")
    add(f"| Томпсон+подмножества против производных Брзозовского | `counterexample` | "
        f"{'расхождений нет' if not constructions_agree else '**РАСХОЖДЕНИЕ: ' + constructions_agree[0] + '**'} |")
    add(f"| Позиционный автомат против ДКА | `disagreements` | "
        f"{'расхождений нет' if not pos_agrees else '**РАСХОЖДЕНИЕ: ' + pos_agrees[0] + '**'} |")
    add(f"| Редуцированный НКА против ДКА | `disagreements` | "
        f"{'расхождений нет' if not nfa_agrees else '**РАСХОЖДЕНИЕ: ' + nfa_agrees[0] + '**'} |")
    add(f"| Таблица классов различает все классы | `ClassTable.is_minimal_proof` | "
        f"{'да' if table.is_minimal_proof else '**нет**'} |")
    add(f"| Обманывающее множество корректно | `FoolingSet.verify` | "
        f"{'да' if best.verify(dfa.accepts) else '**нет**'} |")
    add("| ПКА против ДКА | `tfl.afa.disagreements` | "
        "[НЕ ПРОВЕРЕНО] пункт 3 не выполнен: инвариант не предъявлен |")
    add("| Расширенная регулярка против ДКА | — | [НЕ ПРОВЕРЕНО] пункт 4 не выполнен |")
    add("")
    add("Проверка на конечном срезе не заменяет доказательства, но любое "
        "расхождение здесь означает ошибку наверняка.")
    add("")
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", type=int, help="номер варианта ЛР2 2025")
    parser.add_argument("--regex", help="произвольная регулярка вместо варианта")
    parser.add_argument("--out", default=None, help="каталог для отчёта")
    args = parser.parse_args()

    if args.regex:
        pattern, title = args.regex, "ЛР2: регулярное выражение"
        name = "custom"
    elif args.variant is not None:
        pattern = load_variant(args.variant)
        title = f"ЛР2, вариант {args.variant}"
        name = f"variant-{args.variant:02d}"
    else:
        parser.error("укажите номер варианта или --regex")

    outdir = pathlib.Path(args.out) if args.out else ROOT / "reports" / f"lab2-{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report(pattern, title, outdir)
    target = outdir / "report.md"
    target.write_text(report, encoding="utf-8")
    print(f"отчёт: {target}")
    print(f"графы: {outdir / 'dfa.dot'}, {outdir / 'nfa.dot'}, {outdir / 'positional.dot'}")


if __name__ == "__main__":
    main()
