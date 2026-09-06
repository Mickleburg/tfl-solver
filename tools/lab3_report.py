#!/usr/bin/env python3
"""Генератор отчёта по ЛР3 (КС-грамматики, PDA, регулярные аппроксимации).

    python tools/lab3_report.py 1
    python tools/lab3_report.py --file my.cfg --pda my.pda --out out/

Что делает генератор и чего не делает — граница проходит по наличию оракула.

Механизируется: свойства грамматики (чистка, First/Follow, LL(1), LR(0)/SLR(1)),
перечисление языка, беспрефиксность, обе регулярные аппроксимации сверху,
пересечение по Бар-Хиллелу и проверка, что язык при этом не изменился,
приведение к НФХ и сверка трёх независимых распознавателей.

Не механизируется и помечается `[ВРУЧНУЮ]`: сам вывод о детерминизме языка,
построение DPDA, обоснование каждой оставшейся развилки накачкой. Готовый
автомат можно передать через `--pda` — тогда он проверяется: структурный
детерминизм и пословная сверка с грамматикой.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tfl.approx import intersect, ll1_automaton, lr0_automaton, surplus, to_dfa
from tfl.cfg import CFG, parse_cfg
from tfl.parse import cyk, equivalent_up_to, language, prefix_free, recognize
from tfl.pda import PDA, from_cfg, parse_pda
from tfl.words import iter_words

ROOT = pathlib.Path(__file__).resolve().parent.parent
VARIANTS = ROOT / "evals" / "lab3_2025"

MARK = {True: "✅ да", False: "❌ нет", None: "⚠️ не выяснено"}
MANUAL = "`[ВРУЧНУЮ]`"


def fence(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text}\n```"


def show(words, limit: int = 12) -> str:
    """Слова через запятую, пустое слово — как ε."""
    chosen = list(words)[:limit]
    tail = " …" if len(list(words)) > limit else ""
    return ", ".join(f"`{w}`" if w else "`ε`" for w in chosen) + tail if chosen else "—"


def sorted_words(words) -> list[str]:
    return sorted(words, key=lambda w: (len(w), w))


# --------------------------------------------------------------------------
# Разделы
# --------------------------------------------------------------------------


def section_grammar(add, grammar: CFG, alphabet: str) -> None:
    add("## 1. Грамматика")
    add("")
    add(f"Алфавит: $\\{{{', '.join(sorted(grammar.terminals))}\\}}$, "
        f"нетерминалы: ${', '.join(sorted(grammar.nonterminals))}$, "
        f"стартовый: ${grammar.start}$, правил: {len(grammar)}.")
    add("")
    add(fence(str(grammar)))
    add("")

    cleaned = grammar.clean()
    removed = sorted(set(grammar.nonterminals) - set(cleaned.nonterminals))
    if removed:
        add(f"Чистка убрала бесполезные нетерминалы: ${', '.join(removed)}$. "
            "Язык при этом не меняется.")
    else:
        add("Бесполезных нетерминалов нет: грамматика уже приведённая.")
    add("")

    words = sorted_words(language(grammar, 7))
    add(f"Слова языка длины не больше 7 ({len(words)} шт.): {show(words, 20)}")
    add("")


def section_language_properties(add, grammar: CFG) -> None:
    add("## 2. Свойства грамматики")
    add("")

    nullable = sorted(grammar.nullable())
    recursive = sorted(grammar.left_recursive())
    add(f"* Аннулируемые нетерминалы: ${', '.join(nullable)}$" if nullable
        else "* Аннулируемых нетерминалов нет.")
    add(f"* Левая рекурсия: ${', '.join(recursive)}$ — LL(k)-разбор невозможен "
        "ни при каком $k$ без переписывания грамматики." if recursive
        else "* Левой рекурсии нет.")
    add("")

    first, follow = grammar.first_k(1), grammar.follow_k(1)
    add("| Нетерминал | $First_1$ | $Follow_1$ |")
    add("|---|---|---|")
    for n in sorted(grammar.nonterminals):
        f = ", ".join(sorted("".join(t) or "ε" for t in first[n]))
        g = ", ".join(sorted("".join(t) or "ε" for t in follow.get(n, set())))
        add(f"| ${n}$ | {f} | {g} |")
    add("")

    ll_conflicts = grammar.ll1_table()[1]
    add(f"**LL(1)**: {MARK[not ll_conflicts]}")
    if ll_conflicts:
        add("")
        add(fence("\n".join(str(c) for c in ll_conflicts[:8])))
    add("")
    add(f"**LR(0)**: {MARK[grammar.is_lr0()]}, **SLR(1)**: {MARK[grammar.is_slr1()]}. "
        f"Состояний в каноническом наборе LR(0): {len(grammar.lr0_states()[0])}.")
    slr = grammar.slr1_conflicts()
    if slr:
        add("")
        add(fence("\n".join(str(c) for c in slr[:8])))
    add("")
    add("> Всё перечисленное — свойства **грамматики**, а не языка. Из того, "
        "что грамматика не LL(1), не следует, что язык не LL: достаточно "
        "бывает переписать правила. Обратный переход неразрешим в общем случае.")
    add("")


def section_prefix(add, grammar: CFG, max_len: int) -> None:
    add("## 3. Беспрефиксность")
    add("")
    verdict = prefix_free(grammar, max_len)
    add(f"**Беспрефиксен**: {MARK[verdict.value]} — {verdict.reason}")
    if verdict.value is False:
        short, long = verdict.witness
        add("")
        add(f"Свидетель: `{short or 'ε'}` — собственное начало `{long}`, оба в языке.")
        add("")
        add("Следствие: допуск по пустому стеку невозможен, детерминированный "
            "распознаватель обязан принимать по финальному состоянию "
            "(по пустому стеку DPDA распознаёт ровно беспрефиксные языки).")
    else:
        add("")
        add(f"Контрпримеров до длины {max_len} нет. Это **не доказательство**: "
            "беспрефиксность КС-языка неразрешима. Доказательство "
            f"нужно провести содержательно. {MANUAL}")
    add("")


def section_pda(add, grammar: CFG, pda: PDA | None, alphabet: str, max_len: int) -> None:
    add("## 4. Магазинный автомат")
    add("")
    add(f"**Детерминирован ли язык** — {MANUAL}. Вопрос содержательный: "
        "по грамматике он не решается, потому что недетерминированность "
        "грамматики ничего не говорит о языке.")
    add("")

    top_down = from_cfg(grammar)
    add("### 4.1. Нисходящий автомат по грамматике")
    add("")
    add("Строится механически: стек хранит неразобранный остаток "
        "сентенциальной формы, раскрытие нетерминала — ε-переход, чтение "
        "терминала — снятие с вершины. Допуск по пустому стеку. "
        "Он заведомо недетерминирован и служит эталоном, а не ответом.")
    add("")
    add(fence(str(top_down)))
    add("")
    add(f"Развилок: {len(top_down.nondeterminism())}.")
    add("")

    if pda is None:
        add("### 4.2. Ваш автомат")
        add("")
        add(f"{MANUAL} Постройте DPDA (для детерминированного языка) или PDA "
            "без избыточного недетерминизма и передайте его через `--pda`; "
            "генератор проверит структурный детерминизм и сверит со словами "
            "языка. Формат — см. `tfl.pda.parse_pda`.")
        add("")
        add("Для каждой оставшейся развилки задание требует пару слов "
            "с общим длинным префиксом, на которых поведение стека существенно "
            "расходится; обоснование развилки накачкой даёт +1 балл.")
        add("")
        return

    add("### 4.2. Проверка предъявленного автомата")
    add("")
    add(fence(str(pda)))
    add("")
    forks = pda.nondeterminism()
    add(f"**Детерминирован (структурно)**: {MARK[not forks]}")
    if forks:
        add("")
        add("Развилки (критерий из лекции 9: если есть ε-переход, то он "
            "единственный и других переходов из состояния нет):")
        add("")
        add(fence("\n\n".join(str(c) for c in forks)))
        add("")
        add(f"Каждую нужно либо устранить, либо обосновать накачкой. {MANUAL}")
    add("")

    wrong: list[str] = []
    undecided: list[str] = []
    for word in iter_words(alphabet, max_len):
        verdict = pda.accepts(word)
        if verdict.value is None:
            undecided.append(word)
        elif verdict.value != recognize(grammar, word):
            wrong.append(word)
    add(f"**Сверка с грамматикой** на всех словах длины ≤ {max_len}: "
        f"расхождений {len(wrong)}, без вердикта {len(undecided)}.")
    if wrong:
        add("")
        add(f"Расхождения: {show(sorted_words(wrong))} — автомат распознаёт **не** этот язык.")
    if undecided:
        add("")
        add(f"Без вердикта (бюджет симуляции исчерпан): {show(sorted_words(undecided))}. "
            "Эти слова проверку не проходили и не проваливали.")
    add("")


def section_approximations(add, grammar: CFG, outdir: pathlib.Path,
                           alphabet: str, max_len: int) -> None:
    add("## 5. Регулярные аппроксимации сверху и пересечение")
    add("")
    add("Обе аппроксимации получаются одним приёмом — **берём магазинный "
        "распознаватель и забываем стек**. Возврат из нетерминала перестаёт "
        "быть привязан к месту вызова, поэтому язык может только расшириться. "
        "Для LR(0)-случая это конструкция Перейры–Райта из лекции 9.")
    add("")

    for name, title, build in [
        ("lr0", "LR(0)-автомат (позиционный)", lr0_automaton),
        ("ll1", "LL(1)-автомат", ll1_automaton),
    ]:
        nfa = build(grammar)
        dfa = to_dfa(nfa)
        minimal = dfa.minimize()
        (outdir / f"{name}.dot").write_text(dfa.to_dot(name.upper()), encoding="utf-8")

        add(f"### 5.{1 if name == 'lr0' else 2}. {title}")
        add("")
        add(f"Состояний: {len(dfa)} после детерминизации, {len(minimal)} после "
            f"минимизации. Граф — `{name}.dot`.")
        add("")

        missed = [w for w in sorted_words(language(grammar, max_len)) if not dfa.accepts(w)]
        add(f"**Надмножество** (обязано быть): {MARK[not missed]}")
        if missed:
            add("")
            add(f"Не покрыты: {show(missed)} — ошибка построения автомата.")
        add("")

        extra = surplus(dfa, grammar, alphabet, max_len, 10)
        add(f"Лишние слова, которые автомат принял, забыв стек: {show(extra)}")
        add("")

        crossed = intersect(grammar, dfa)
        (outdir / f"intersect-{name}.cfg").write_text(str(crossed) + "\n", encoding="utf-8")
        lost, gained = equivalent_up_to(grammar, crossed, max_len)
        add(f"Пересечение по Бар-Хиллелу: {len(crossed)} правил "
            f"(нетерминалы вида $\\langle p, A, q \\rangle$), файл `intersect-{name}.cfg`.")
        add("")
        add(f"**Языки до и после пересечения совпадают** на словах длины ≤ {max_len}: "
            f"{MARK[not lost and not gained]}")
        if lost:
            add("")
            add(f"Пересечение потеряло: {show(lost)}.")
        if gained:
            add("")
            add(f"Пересечение добавило: {show(gained)} — такого быть не может, ошибка.")
        add("")


def section_recognizers(add, grammar: CFG, alphabet: str, max_len: int) -> None:
    add("## 6. Сверка распознавателей")
    add("")
    add("Требование задания «провести автоматическое тестирование "
        "предполагаемой эквивалентности построенных распознавателей» "
        "выполняется тремя независимыми путями: алгоритм Эрли по исходной "
        "грамматике (сверху вниз), алгоритм Кока–Янгера–Касами по нормальной "
        "форме Хомского (снизу вверх) и перечисление языка неподвижной точкой.")
    add("")

    normal = grammar.chomsky_normal_form()
    add(f"Нормальная форма Хомского: {len(normal)} правил, форма проверена: "
        f"{MARK[normal.is_chomsky_normal_form()]}")
    add("")

    inside = language(grammar, max_len)
    rows = []
    for word in iter_words(alphabet, max_len):
        earley = recognize(grammar, word)
        table = cyk(grammar, word, normal)
        enumerated = word in inside
        if not (earley == table == enumerated):
            rows.append((word, earley, table, enumerated))
    add(f"Слов проверено: {sum(1 for _ in iter_words(alphabet, max_len))}, "
        f"расхождений: {len(rows)}.")
    if rows:
        add("")
        add("| Слово | Эрли | CYK | перечисление |")
        add("|---|---|---|---|")
        for word, a, b, c in rows[:10]:
            add(f"| `{word or 'ε'}` | {a} | {b} | {c} |")
    add("")


def section_manual(add, pda: PDA | None) -> None:
    add("## 7. Что осталось сделать вручную")
    add("")
    add("| Пункт задания | Почему не автоматизируется |")
    add("|---|---|")
    add("| Вывод о детерминизме языка | Свойство языка, а не грамматики; "
        "эквивалентность КС-языков неразрешима |")
    if pda is None:
        add("| Построение DPDA/PDA | Требует содержательной догадки о том, "
            "что держать в стеке; проверить готовый автомат генератор умеет |")
    add("| Обоснование развилок накачкой (+1 балл) | Нужна пара слов с общим "
        "длинным префиксом и разбор поведения стека |")
    add("| Доказательство беспрефиксности | Неразрешима; перебор даёт только "
        "контрпримеры |")
    add("| LL-свойство языка (+1 балл) | Неразрешимо; генератор отвечает "
        "только про конкретную грамматику |")
    add("")


# --------------------------------------------------------------------------


def build_report(grammar: CFG, title: str, pda: PDA | None,
                 outdir: pathlib.Path, budget: dict) -> str:
    md: list[str] = []
    add = md.append
    alphabet = "".join(sorted(grammar.terminals)) or "ab"

    add(f"# {title}")
    add("")
    add("> Отчёт сгенерирован `tools/lab3_report.py`. Всё, что помечено "
        f"{MANUAL}, оракула не имеет и требует содержательного рассуждения. "
        "Проверки на словах ограниченной длины опровергают гипотезы, "
        "но не доказывают их.")
    add("")

    section_grammar(add, grammar, alphabet)
    section_language_properties(add, grammar)
    section_prefix(add, grammar, budget["prefix_len"])
    section_pda(add, grammar, pda, alphabet, budget["pda_len"])
    section_approximations(add, grammar, outdir, alphabet, budget["approx_len"])
    section_recognizers(add, grammar, alphabet, budget["cross_len"])
    section_manual(add, pda)

    return "\n".join(md) + "\n"


DEFAULT_BUDGET = {
    "prefix_len": 10,
    "pda_len": 6,
    "approx_len": 7,
    "cross_len": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", type=int, help="номер варианта ЛР3 2025")
    parser.add_argument("--file", help="путь к .cfg вместо варианта")
    parser.add_argument("--pda", help="путь к .pda: построенный вами автомат для проверки")
    parser.add_argument("--out", help="каталог для отчёта")
    args = parser.parse_args()

    if args.file:
        path = pathlib.Path(args.file)
        title, name = f"ЛР3: {path.name}", path.stem
    elif args.variant is not None:
        path = VARIANTS / f"variant-{args.variant:02d}.cfg"
        title, name = f"ЛР3, вариант {args.variant}", f"variant-{args.variant:02d}"
        if not path.exists():
            raise SystemExit(
                f"нет файла {path}. Грамматики даны только в нечётных вариантах; "
                "в чётных язык задан словесно — см. evals/lab3_2025/languages.md, "
                "грамматику по описанию нужно построить самому и передать --file"
            )
    else:
        parser.error("укажите номер варианта или --file")

    grammar = parse_cfg(path.read_text(encoding="utf-8"))
    pda = parse_pda(pathlib.Path(args.pda).read_text(encoding="utf-8")) if args.pda else None

    outdir = pathlib.Path(args.out) if args.out else ROOT / "reports" / f"lab3-{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report(grammar, title, pda, outdir, DEFAULT_BUDGET)
    target = outdir / "report.md"
    target.write_text(report, encoding="utf-8")
    print(f"отчёт: {target}")


if __name__ == "__main__":
    main()
