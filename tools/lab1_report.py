#!/usr/bin/env python3
"""Генератор отчёта по ЛР1 (системы переписывания строк).

    python tools/lab1_report.py 20
    python tools/lab1_report.py --file my.srs --out out/

Автоматизируется всё, что имеет исполняемый оракул: завершимость, критические
пары, пополнение по Кнуту–Бендиксу, классы эквивалентности, поиск линейных
инвариантов и сверка исходной системы с построенной.

Каждый вывод сопровождается свидетелем — циклом, критической парой,
фундированным порядком. Там, где бюджет исчерпан и вывода нет, так и
написано: «не выяснено». Это не отговорка, а единственный честный ответ
для свойств, неразрешимых в общем случае.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tfl.srs import SRS, Verdict, parse_srs

ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANTS = ROOT / "evals" / "lab1_2025"

MARK = {True: "✅ да", False: "❌ нет", None: "⚠️ не выяснено"}


def rules_block(srs: SRS) -> str:
    return "```text\n" + "\n".join(str(r) for r in srs.rules) + "\n```"


def class_growth(srs: SRS, lengths: tuple[int, ...], search_len: int) -> list[tuple[int, int, bool]]:
    """Число классов эквивалентности среди слов растущей длины.

    Конечность числа классов на всём Σ* бесконечным перебором не проверить,
    но выход счётчика на плато — содержательный довод в пользу конечности,
    а безостановочный рост — против.
    """
    out = []
    for n in lengths:
        classes = srs.equivalence_classes(max_len=n, slack=search_len - n)
        out.append((n, len(set(classes.values())), True))
    return out


def build_report(srs: SRS, title: str, precedence: str, budget: dict) -> str:
    md: list[str] = []
    add = md.append

    add(f"# {title}")
    add("")
    add(f"Алфавит: {{{', '.join(sorted(srs.alphabet))}}}, правил: {len(srs)}.")
    add("")
    add(rules_block(srs))
    add("")
    add("> Отчёт сгенерирован `tools/lab1_report.py`. Свойства SRS неразрешимы")
    add("> в общем случае, поэтому часть выводов помечена «не выяснено» —")
    add("> это означает исчерпание бюджета поиска, а не отсутствие ответа.")
    add("")

    # -- 1. Завершимость ----------------------------------------------------
    add("## 1. Завершимость")
    add("")
    termination = srs.terminates(
        max_len=budget["cycle_len"], context=budget["context"]
    )
    add(f"**{MARK[termination.value]}** — {termination.reason}")
    add("")
    if termination.value is False:
        cycle = termination.witness
        add("Цикл переписывания:")
        add("")
        add("```text")
        add(" → ".join(cycle))
        add("```")
        add("")
        add("Слово возвращается к себе, значит переписывание можно продолжать "
            "бесконечно и нормальной формы у него нет.")
    elif termination.value is None:
        add("Ни цикла, ни убывающего армейского порядка. Дальше — вручную: "
            "рекурсивный по путям порядок (LPO), полиномиальная интерпретация "
            "или SMT-модель (см. `2023/SMTLIB2.pdf`).")
    add("")

    monotone = srs.length_is_monotone()
    add(f"Длина слова не возрастает: **{MARK[monotone.value]}** — {monotone.reason}")
    add("")

    # -- 2. Классы эквивалентности ------------------------------------------
    add("## 2. Классы эквивалентности по нормальной форме")
    add("")
    add("Правила применяются в обе стороны, как требует задание. Классы "
        "считаются среди слов растущей длины: выход на плато — довод "
        "в пользу конечности числа классов, продолжающийся рост — против.")
    add("")
    growth = class_growth(srs, budget["class_lengths"], budget["class_search"])
    add("| слов до длины | классов |")
    add("|---|---|")
    for n, count, _ in growth:
        add(f"| {n} | {count} |")
    add("")
    counts = [c for _, c, _ in growth]
    if len(set(counts)) == 1:
        add(f"Счётчик стоит на {counts[0]} — похоже, классов конечное число.")
    else:
        add("Счётчик растёт, то есть на проверенном отрезке классы не "
            "стабилизировались. [ВРУЧНУЮ] требуется содержательный довод.")
    add("")

    # -- 3. Локальная конфлюэнтность ----------------------------------------
    add("## 3. Локальная конфлюэнтность")
    add("")
    pairs = srs.critical_pairs()
    add(f"Критических пар: **{len(pairs)}**. Рассматриваются только наложения "
        f"левых частей — непересекающиеся применения правил сходятся всегда.")
    add("")
    confluence = srs.locally_confluent(max_len=budget["join_len"])
    add(f"**{MARK[confluence.value]}** — {confluence.reason}")
    add("")
    if confluence.value is False:
        critical, left, right, r1, r2 = confluence.witness
        add("Контрпример:")
        add("")
        add("```text")
        add(f"{critical}")
        add(f"  ├─ по правилу {r1}  →  {left or 'ε'}")
        add(f"  └─ по правилу {r2}  →  {right or 'ε'}")
        add("```")
        add("")
        add("Множества потомков обеих веток вычислены полностью и не "
            "пересекаются, поэтому свести их нельзя.")
    add("")

    # -- 4. Пополнение ------------------------------------------------------
    add("## 4. Пополнение по Кнуту–Бендиксу")
    add("")
    add(f"Порядок — армейский с приоритетом букв `{precedence}` "
        f"(длина, затем лексикографика).")
    add("")
    completed, completion = srs.complete(
        precedence,
        max_rules=budget["max_rules"],
        max_rounds=budget["max_rounds"],
        max_len=budget["join_len"],
    )
    add(f"**{MARK[completion.value]}** — {completion.reason}")
    add("")
    add(f"Полученная система `T′` ({len(completed)} правил):")
    add("")
    add(rules_block(completed))
    add("")
    if completion.value is True:
        after = completed.locally_confluent(max_len=budget["join_len"])
        add(f"Проверка результата: локальная конфлюэнтность `T′` — "
            f"{MARK[after.value]}.")
    else:
        add("Процедура Кнута–Бендикса может не завершаться, и здесь она "
            "уперлась в лимит. Система выше — промежуточная, конфлюэнтной "
            "её считать нельзя.")
    add("")

    # -- 5. Сверка T и T' ---------------------------------------------------
    add("## 5. Сохраняет ли `T′` классы эквивалентности")
    add("")
    systematic = srs.same_equivalence(
        completed, max_len=budget["eq_len"], search_len=budget["eq_search"]
    )
    fuzz = srs.fuzz_equivalence(completed, trials=budget["fuzz_trials"], seed=0)
    add(f"* Систематически (все слова до длины {budget['eq_len']}): "
        f"**{MARK[systematic.value]}** — {systematic.reason}")
    add(f"* Фаззом по схеме задания ({budget['fuzz_trials']} цепочек, seed=0): "
        f"**{MARK[fuzz.value]}** — {fuzz.reason}")
    add("")
    add("Обратные правила удлиняют слова, поэтому классы почти всегда "
        "не помещаются в бюджет обхода целиком. Вывод «не эквивалентны» "
        "делается, только если класс той стороны, которой слова не хватило, "
        "вычислен полностью.")
    add("")

    # -- 6. Инварианты ------------------------------------------------------
    add("## 6. Инварианты для метаморфного тестирования")
    add("")
    add("Ищутся инварианты вида `Σ cₓ·|w|ₓ mod m`. Такой инвариант сохраняется "
        "тогда и только тогда, когда для каждого правила "
        "`Σ cₓ·(|rhs|ₓ − |lhs|ₓ) ≡ 0 (mod m)`; это однородная линейная система, "
        "и её решения находятся точно, а не подбором.")
    add("")
    found: list[tuple[int, dict[str, int]]] = []
    for modulus in (2, 3, 5, 7):
        for vector in srs.linear_invariants(modulus):
            found.append((modulus, vector))
    if found:
        add("| модуль | инвариант | перепроверка на словах |")
        add("|---|---|---|")
        for modulus, vector in found:
            terms = " + ".join(
                (f"{c}·|w|_{x}" if c != 1 else f"|w|_{x}") for x, c in vector.items()
            )
            check = srs.check_invariant(
                lambda w, v=vector, m=modulus: sum(c * w.count(x) for x, c in v.items()) % m,
                budget["invariant_words"],
            )
            add(f"| {modulus} | `{terms} ≡ const` | {MARK[check.value]} |")
        add("")
        add("Инварианты доказаны для всех слов сразу — линейной алгеброй над "
            "GF(m). Колонка перепроверки прогоняет их независимым кодом "
            "по конкретным словам: две независимые реализации должны сойтись.")
    else:
        add("Линейных инвариантов по модулям 2, 3, 5, 7 нет.")
        add("")
        add("[ВРУЧНУЮ] Задание просит минимум два инварианта, так что нужны "
            "нелинейные: моноидные гомоморфизмы в матрицы, порядковые меры, "
            "число вхождений подслов. За нетривиальные — до +2 баллов.")
    add("")
    if monotone.value is True:
        add("Дополнительно годится монотонный инвариант: длина слова "
            "не возрастает при переписывании.")
        add("")

    # -- 7. Протокол --------------------------------------------------------
    add("## 7. Протокол проверок (уровень L1)")
    add("")
    add("| Что проверялось | Бюджет | Результат |")
    add("|---|---|---|")
    add(f"| Поиск цикла переписывания | слова до длины {budget['cycle_len']}, "
        f"контекст {budget['context']} | {MARK[termination.value]} |")
    add(f"| Сходимость критических пар | потомки до длины {budget['join_len']} | "
        f"{MARK[confluence.value]} |")
    add(f"| Пополнение | ≤{budget['max_rules']} правил, "
        f"≤{budget['max_rounds']} раундов | {MARK[completion.value]} |")
    add(f"| Совпадение классов `T` и `T′` | слова до длины {budget['eq_len']} | "
        f"{MARK[systematic.value]} |")
    add(f"| Фазз-эквивалентность | {budget['fuzz_trials']} цепочек | "
        f"{MARK[fuzz.value]} |")
    add(f"| Линейные инварианты | модули 2, 3, 5, 7 | найдено {len(found)} |")
    add("")
    return "\n".join(md)


DEFAULT_BUDGET = {
    "cycle_len": 9,
    "context": 1,
    "join_len": 11,
    "max_rules": 40,
    "max_rounds": 8,
    "class_lengths": (4, 5, 6),
    "class_search": 8,
    "eq_len": 3,
    "eq_search": 7,
    "fuzz_trials": 200,
    "invariant_words": ["ab", "aab", "abab", "bba", "abcabc", "aabbcc"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", type=int, help="номер варианта ЛР1 2025")
    parser.add_argument("--file", help="путь к .srs вместо варианта")
    parser.add_argument("--precedence", help="приоритет букв для армейского порядка")
    parser.add_argument("--out", help="каталог для отчёта")
    args = parser.parse_args()

    if args.file:
        path = pathlib.Path(args.file)
        title, name = f"ЛР1: {path.name}", path.stem
    elif args.variant is not None:
        path = VARIANTS / f"variant-{args.variant:02d}.srs"
        title, name = f"ЛР1, вариант {args.variant}", f"variant-{args.variant:02d}"
        if not path.exists():
            raise SystemExit(f"нет файла {path}; запустите tools/extract_lab1_variants.py")
    else:
        parser.error("укажите номер варианта или --file")

    srs = parse_srs(path.read_text(encoding="utf-8"))
    precedence = args.precedence or "".join(sorted(srs.alphabet))

    outdir = pathlib.Path(args.out) if args.out else ROOT / "reports" / f"lab1-{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report(srs, title, precedence, DEFAULT_BUDGET)
    target = outdir / "report.md"
    target.write_text(report, encoding="utf-8")
    print(f"отчёт: {target}")


if __name__ == "__main__":
    main()
