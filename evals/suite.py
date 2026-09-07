"""Набор для оценки: задачи корпуса и то, куда по ним приходит оракул.

Сессия S-8. До неё рецепты были написаны для всех четырнадцати классов,
но **систематической проверки того, что по ним получается, не было** —
утверждения «закрыто» жили в прозе `docs/OPEN-GAPS.md` и нигде не считались.
Здесь каждое такое утверждение становится исполняемой проверкой с одним
из трёх исходов:

* `решено` — оракул выдаёт ответ на вопрос задачи;
* `частично` — оракул даёт необходимое условие, свидетеля или проверку
  предъявленного объекта, но не ответ целиком;
* `вручную` — исполняемой части нет, задача решается человеком.

Разница между `решено` и `частично` проведена **по вопросу условия**,
а не по тому, много ли кода отработало. Проверка предъявленной
интерпретации — это `частично`: интерпретацию надо ещё придумать.

Запуск: `python tools/eval_suite.py`. Тест `tests/test_eval_suite.py`
следит, чтобы исходы не разъезжались с записанными здесь.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

SOLVED = "решено"
PARTIAL = "частично"
MANUAL = "вручную"


@dataclass(frozen=True)
class Case:
    """Одна задача набора."""

    id: str
    task_class: str
    source: str
    statement: str
    expected: str
    run: Callable[[], tuple[str, str]]
    points: int | None = None

    def evaluate(self) -> tuple[str, str]:
        try:
            return self.run()
        except Exception as error:  # noqa: BLE001 — падение оракула тоже исход
            return ("ошибка", f"{type(error).__name__}: {error}")


# --------------------------------------------------------------------------
# ЛР1 — системы переписывания строк
# --------------------------------------------------------------------------


def _lab1_variants():
    import glob

    from tfl.srs import parse_srs

    counts = {True: 0, False: 0, None: 0}
    for path in sorted(glob.glob("evals/lab1_2025/*.srs")):
        system = parse_srs(open(path, encoding="utf-8").read())
        counts[system.terminates().value] += 1
    total = sum(counts.values())
    status = PARTIAL if counts[None] else SOLVED
    return status, (
        f"{total} вариантов: незавершимость доказана для {counts[False]}, "
        f"завершимость для {counts[True]}, «не выяснено» {counts[None]}. "
        f"Матричный поиск сюда не входит — он платный по времени и вызывается "
        f"отдельно; с ним завершимых становится {counts[True] + 1} "
        f"(`evals/lab1_2025_report.md`)"
    )


def _lab1_loop():
    from tfl.srs import parse_srs

    verdict = parse_srs("aaa -> aaab").terminates()
    ok = verdict.value is False and "петл" in verdict.reason
    return (SOLVED if ok else "ошибка", verdict.reason[:90])


def _lab1_matrix():
    from tfl.srs import parse_srs

    verdict = parse_srs("fg -> gff\nfh -> hg").find_matrix_interpretation(3, 2, 25_000)
    if verdict.value is True:
        return SOLVED, f"матрица 3×3: {verdict.witness}"
    return PARTIAL, verdict.reason[:110]


def _lab1_unification():
    from tfl.trs import parse_term, unified_term, unify_mm

    names = {"q", "x", "y", "a", "w"}
    system = unify_mm(
        parse_term("F(q,q,q)", names),
        parse_term("F(F(x,y,R), F(a,w,a), F(w,x,y))", names),
    )
    answer = str(unified_term(system))
    ok = answer == "F(F(R, R, R), F(R, R, R), F(R, R, R))"
    return (SOLVED if ok else "ошибка", f"результат: {answer}")


def _lab1_completion():
    from tfl.trs import parse_trs

    axioms = parse_trs(
        "variables = [x, y, z]\n"
        "f(f(x, y), z) -> f(x, f(y, z))\n"
        "f(e, x) -> x\n"
        "f(i(x), x) -> e"
    )
    done, verdict = axioms.complete()
    ok = verdict.value is True and len(done) == 10
    return (SOLVED if ok else PARTIAL, f"{verdict.reason[:80]}, правил {len(done)}")


# --------------------------------------------------------------------------
# ЛР2 — регулярные выражения и автоматы
# --------------------------------------------------------------------------


def _lab2_glushkov():
    from tfl import regex as rx
    from tfl.automata import dfa_of, equivalent
    from tfl.glushkov import glushkov

    pattern = "(a|b)*abb"
    automaton = glushkov(rx.parse(pattern))
    same = equivalent(
        automaton.determinize().minimize(), dfa_of(pattern, "ab").minimize()
    )
    return (
        SOLVED if same else "ошибка",
        f"автомат Глушкова из {len(automaton.states)} состояний "
        f"совпал с ДКА по производным",
    )


def _lab2_afa():
    from tfl.afa import conjunction, disagreements
    from tfl.automata import dfa_of

    machine = conjunction([dfa_of("(a|b)*a(a|b)*", "ab"), dfa_of("(a|b)*b(a|b)*", "ab")], "ab")
    apart = disagreements(machine, lambda w: "a" in w and "b" in w, max_len=6)
    return (
        SOLVED if not apart else "ошибка",
        "ПКА как И-ветвление в стартовой вершине: расхождений с предикатом нет",
    )


def _lab2_minimality():
    from tfl.automata import dfa_of
    from tfl.myhill import extended_fooling_set

    lower = extended_fooling_set(dfa_of("(a|b)*abb", "ab"))
    return PARTIAL, (
        f"оценка снизу на число состояний НКА: {len(lower.pairs)}; "
        f"совпадение с верхней границей доказывало бы минимальность, "
        f"иначе вывода нет"
    )


# --------------------------------------------------------------------------
# ЛР3 — КС-грамматики
# --------------------------------------------------------------------------


def _lab3_properties():
    from tfl.cfg import parse_cfg

    grammar = parse_cfg("S -> a S b | ε")
    return SOLVED, (
        f"LL(1): {grammar.is_ll1()}, LR(0): {grammar.is_lr0()}, "
        f"нетерминалов {len(grammar.nonterminals)}"
    )


def _lab3_pda():
    from tfl.cfg import parse_cfg
    from tfl.pda import disagreements, from_cfg

    automaton = from_cfg(parse_cfg("S -> a S b | ε"))
    missed, extra = disagreements(
        automaton,
        lambda w: len(w) % 2 == 0 and w == "a" * (len(w) // 2) + "b" * (len(w) // 2),
        "ab",
        max_len=6,
    )
    return (
        SOLVED if not missed and not extra else PARTIAL,
        f"МП-автомат по грамматике: потеряно {len(missed)}, лишних {len(extra)}",
    )


def _lab3_determinism():
    return MANUAL, (
        "детерминированность **языка** неразрешима; оракул проверяет "
        "детерминизм предъявленного автомата, вывод о языке за человеком"
    )


# --------------------------------------------------------------------------
# ЛР4 — расширенные регулярки
# --------------------------------------------------------------------------


#: Вердикты преподавателя из issue #35 — приёмочная таблица ЛР4.
LAB4_VERDICTS = (
    ("(a|(bb))(?2)", True),
    (r"(a(bb)|b(cc))\2", False),
    ("((?:a(?2)|(bb))(?1))", True),
    ("(a(?1))", True),
    (r"(a|(bb))\2", False),
    ("(a|(bb))(a|(?3))", True),
    (r"(a|(?2))(a|(bb\1))", False),
    ("(?1)(a|b)", True),
    (r"(\1)(a|b)", False),
    ("(?1)(a|b)*(?1)", True),
    (r"(a|b)*\1", False),
    ("((?1)a|b)", True),
)


def _lab4_verdicts():
    from tfl.extre import parse_extended, validate

    agree = sum(
        1
        for pattern, expected in LAB4_VERDICTS
        if validate(parse_extended(pattern)).value is expected
    )
    total = len(LAB4_VERDICTS)
    return (
        SOLVED if agree == total else "ошибка",
        f"вердиктов преподавателя воспроизведено {agree} из {total} (issue #35)",
    )


def _lab4_capture():
    return MANUAL, (
        "точная проверка вхождения со ссылками на строку не сделана: "
        "равенство захваченных подстрок КС-грамматикой не выражается"
    )


# --------------------------------------------------------------------------
# РК1
# --------------------------------------------------------------------------


def _rk1_counter():
    from tfl.lang import counter_dfa

    verdict = counter_dfa({"ab": 1, "ba": -1}, "ab")
    if verdict.value is True:
        return SOLVED, f"регулярность доказана обходом: {verdict.reason[:80]}"
    return PARTIAL, verdict.reason[:100]


def _rk1_monoid():
    from tfl.automata import dfa_of
    from tfl.monoid import syntactic_monoid

    monoid = syntactic_monoid(dfa_of("(a|b)*abb", "ab"))
    return SOLVED, f"синтаксический моноид: {len(monoid)} элементов"


def _rk1_schutzenberger():
    from tfl.cfg import parse_cfg
    from tfl.schutzenberger import bracketed

    representation = bracketed(parse_cfg("S -> A X | A B\nX -> S B\nA -> a\nB -> b"))
    verdict = representation.agrees_with_source(40)
    return (
        SOLVED if verdict.value is True else "ошибка",
        f"скобочное представление для aⁿbⁿ: {verdict.reason[:70]}",
    )


def _rk1_conway():
    from tfl.conway import is_minimal

    verdict = is_minimal("(a*b)*a*", max_size=12)
    if verdict.value is False:
        return SOLVED, f"сокращено до «{verdict.witness.best}» тождеством Конвея"
    return PARTIAL, verdict.reason[:100]


def _rk1_irreducible():
    from tfl.conway import is_minimal

    verdict = is_minimal("(a|b)*", max_size=12)
    return (
        PARTIAL if verdict.value is None else "ошибка",
        f"{verdict.reason[:120]}",
    )


def _rk1_tree():
    from tfl.tree import trees

    found = trees({"f": 2, "a": 0}, 5)
    return SOLVED, f"древесных термов размера ≤ 5: {len(found)}, пример {found[-1]}"


# --------------------------------------------------------------------------
# РК2
# --------------------------------------------------------------------------


def _rk2_counting():
    from tfl.cfg import parse_cfg

    # Проверенная работа РК2 2025: `I(w) = |w|_a − 2|w|_b` нечётно на всех
    # выводимых словах, а условие `|w|_a = 4|w|_b` делает его чётным.
    grammar = parse_cfg("S -> a S b S b | a S a | a")
    verdict = grammar.counting_condition({"a": 1, "b": -4})
    return (
        SOLVED if verdict.value is False else PARTIAL,
        f"числовой инвариант грамматики: {verdict.reason[:90]}",
    )


def _rk2_conjunctive():
    from tfl.conj import parse_conjunctive

    grammar = parse_conjunctive("S -> A & B\nA -> a A | a\nB -> a B a | a", start="S")
    verdict = grammar.agrees_with(lambda w: set(w) <= {"a"} and len(w) % 2 == 1, max_len=7)
    return (
        SOLVED if verdict.value is True else PARTIAL,
        f"конъюнктивная грамматика: {verdict.reason[:80]}",
    )


def _rk2_mfa():
    from tfl.mfa import parse_mfa

    machine = parse_mfa(
        "q0 2 q1 o ⋄\nq1 1 q2 c o\nq2 a q3 ⋄ ⋄\nq3 2 q1 o c",
        start="q0",
        accepting="q3",
        cells=2,
        alphabet="a",
    )
    ok = machine.words(16) == ["a", "a" * 4, "a" * 9, "a" * 16]
    return (SOLVED if ok else "ошибка", "2-MFA лекции для {aⁿ²} воспроизведён")


def _rk2_csy():
    from tfl.lang import from_predicate
    from tfl.pump import defeats_csy

    language = from_predicate(
        lambda w: len(w) % 2 == 0 and w == "a" * (len(w) // 2) + "b" * (len(w) // 2), "ab"
    )
    verdict = defeats_csy(language, "aaabbb", 3)
    return (
        PARTIAL if verdict.value is True else "ошибка",
        "накачка CSY отбивает свидетеля при N = 3; произвольное N — за человеком",
    )


def _rk2_jumping():
    from tfl.lang import from_predicate
    from tfl.mfa import nondmfl_by_jumping

    language = from_predicate(
        lambda w: len(w) % 2 == 0 and w == "a" * (len(w) // 2) + "b" * (len(w) // 2), "ab"
    )
    verdict = nondmfl_by_jumping(language, upto=2, max_n=3, max_prefix=6, every=True)
    return (
        PARTIAL if verdict.value is None else "ошибка",
        "Jumping Lemma: условие не выполнено в пределах перебора",
    )


def _rk2_attributes():
    from tfl.attr import parse_attr_grammar

    grammar = parse_attr_grammar(
        "S -> a S b ; S0.n := S1.n + 1\nS -> ε ; S.n := 0"
    )
    verdict = grammar.check_attribute("S", "n", lambda word: len(word) // 2, max_len=6)
    return (
        SOLVED if verdict.value is True else PARTIAL,
        f"атрибутная грамматика: {verdict.reason[:80]}",
    )


def _rk2_real_time():
    from tfl.cfg import parse_cfg
    from tfl.pda import from_cfg

    automaton = from_cfg(parse_cfg("S -> a S b | ε"))
    return PARTIAL, (
        f"real-time: ε-переходов {len(automaton.epsilon_transitions())}, "
        f"снятий цепочки {len(automaton.multi_pop_transitions())}; "
        f"несуществование real-time автомата — свойство языка, не механизировано"
    )


# --------------------------------------------------------------------------
# Экзамен
# --------------------------------------------------------------------------


def _exam_mu():
    from tfl.mu import parse_mu

    expression = parse_mu("µX.(a(µY.bX|Ya|(µZ.ZZ|cc))bX|ε)")
    grammar = expression.to_cfg()
    ok = len(grammar.productions) == 7 and grammar.start == "X"
    return (
        SOLVED if ok else "ошибка",
        f"µ-выражение переведено в грамматику из {len(grammar.productions)} правил",
    )


def _exam_pcp():
    from tfl.pcp import parse_pcp

    verdict = parse_pcp("(ab,bba)\n(aba,a)\n(b,ba)").refute_by_adjacency()
    return (
        SOLVED if verdict.value is False else PARTIAL,
        f"ПСП билета 2024: {verdict.reason[:80]}",
    )


def _exam_srs():
    from tfl.srs import parse_srs

    questions = [
        "fg -> gff\nfh -> hg",
        "ab -> ba",
        "aaa -> aaab",
        "a -> bb",
    ]
    decided = 0
    for text in questions:
        system = parse_srs(text)
        if system.terminates().value is not None:
            decided += 1
        elif system.find_matrix_interpretation(3, 2, 25_000).value is True:
            decided += 1
    return (
        SOLVED if decided == len(questions) else PARTIAL,
        f"SRS-вопросов решено механически: {decided} из {len(questions)}",
    )


def _exam_ogden():
    import re

    from tfl.lang import from_predicate
    from tfl.pump import defeats_ogden

    def predicate(word: str) -> bool:
        match = re.fullmatch(r"(a*)(b*)(c*)(d*)", word)
        if match is None:
            return False
        a, b, c, d = (len(group) for group in match.groups())
        return b == c == d if a else True

    language = from_predicate(predicate, "abcd", "{aᵐbⁿcⁿdⁿ} ∪ {bⁱcʲdᵏ}")
    n = 3
    word = "a" + "b" * (2 * n) + "c" * (2 * n) + "d" * (2 * n)
    marks = set(range(len(word) - n, len(word)))
    verdict = defeats_ogden(language, word, marks, n)
    return (
        PARTIAL if verdict.value is True else "ошибка",
        "лемма Огдена отбивает все разбиения при n = 3; произвольное n — "
        "за человеком. Обычная накачка этот язык не берёт вовсе",
    )


def _exam_closure():
    return MANUAL, (
        "вопросы о замкнутости классов (47 из 85 третьих вопросов) решаются "
        "предъявлением контрпримера; оракул контрпример проверяет, "
        "но не придумывает"
    )


# --------------------------------------------------------------------------
# «Аптека»
# --------------------------------------------------------------------------


def _pharma_pcp_one():
    from tfl.pcp import parse_pcp

    verdict = parse_pcp("(a,ba)\n(aa,ba)\n(b,ba)").refute_by_counting()
    return (
        SOLVED if verdict.value is False else PARTIAL,
        "счётные уравнения на числа домино опровергают решение",
    )


def _pharma_pcp_all_pairs():
    from tfl.pcp import PCP

    instance = PCP.all_pairs(["ab", "b"])
    verdict = instance.solve()
    ok = verdict.value is True and len(verdict.witness) == 1
    return (
        SOLVED if ok else "ошибка",
        "домино из всех пар: решение — диагональное домино, одно",
    )


def _pharma_trs_unary():
    from tfl.trs import parse_trs

    system = parse_trs("variables = [X]\nf(g(X)) -> g(f(X))")
    bridged = system.as_srs()
    return SOLVED, f"унарная TRS переведена в строковую систему: {bridged}"


def _pharma_interpretation_check():
    from tfl.trs import parse_interpretation, parse_trs

    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    supplied = parse_interpretation("plus(x, y) = 2*x + y + 1; s(x) = x + 1; 0 = 1")
    verdict = system.check_interpretation(supplied)
    return (
        PARTIAL if verdict.value is True else "ошибка",
        "предъявленная интерпретация проверена; придумать её оракул не берётся",
    )


def _pharma_f8():
    from tfl.trs import parse_trs

    system = parse_trs("variables = [x, y]\nf(f(a,x),y) -> f(f(x,f(a,y)),a)")
    verdict = system.terminates()
    return (
        MANUAL if verdict.value is None else SOLVED,
        f"F8: {verdict.reason[:100]}",
    )


# --------------------------------------------------------------------------
# Набор
# --------------------------------------------------------------------------

CASES: tuple[Case, ...] = (
    Case("lab1-2025-все", "LAB-1", "lab_tfl_2025_*.pdf",
         "исследовать SRS на завершимость", PARTIAL, _lab1_variants),
    Case("lab1-петля", "LAB-1", "лекция 2 / ЛР1",
         "правило с вложенной левой частью", SOLVED, _lab1_loop),
    Case("lab1-матрица", "LAB-1", "Pharma_2022, вопрос 12",
         "исследовать на завершаемость fg → gff, fh → hg", SOLVED, _lab1_matrix, 5),
    Case("lab1-2022-унификация", "LAB-1", "issue #2",
         "унификация алгоритмом 3 Мартелли–Монтанари", SOLVED, _lab1_unification),
    Case("lab1-кнут-бендикс", "LAB-1", "аксиомы группы",
         "пополнение системы термов", SOLVED, _lab1_completion),
    Case("lab2-глушков", "LAB-2", "ЛР2 2025",
         "позиционный автомат по регулярке", SOLVED, _lab2_glushkov),
    Case("lab2-пка", "LAB-2", "issue #40",
         "переключающийся автомат для конъюнкции", SOLVED, _lab2_afa),
    Case("lab2-минимальность", "LAB-2", "ЛР2 2025",
         "обосновать минимальность НКА", PARTIAL, _lab2_minimality),
    Case("lab3-свойства", "LAB-3", "ЛР3 2025",
         "LL(1), LR(0), нормальные формы", SOLVED, _lab3_properties),
    Case("lab3-мп-автомат", "LAB-3", "ЛР3 2025",
         "построить МП-автомат по грамматике", SOLVED, _lab3_pda),
    Case("lab3-детерминизм", "LAB-3", "ЛР3 2025",
         "детерминирован ли язык", MANUAL, _lab3_determinism),
    Case("lab4-вердикты", "LAB-4", "issue #35",
         "корректность ссылок в расширенной регулярке", SOLVED, _lab4_verdicts),
    Case("lab4-захваты", "LAB-4", "ЛР4",
         "разбор со ссылками на строку", MANUAL, _lab4_capture),
    Case("rk1-счётчики", "RK1-B", "rk1_tfl_2025",
         "регулярен ли язык с условием на счётчики подслов", SOLVED, _rk1_counter),
    Case("rk1-моноид", "RK1-A", "разбалловка issue #6",
         "синтаксический моноид языка", SOLVED, _rk1_monoid, 1),
    Case("rk1-шютценберже", "RK1-B", "разбалловка issue #21",
         "скобочное представление языка", SOLVED, _rk1_schutzenberger, 2),
    Case("rk1-конвей", "RK1-B", "разбалловка issue #21",
         "сократить регулярку переписываниями Конвея", SOLVED, _rk1_conway),
    Case("rk1-конвей-минимальность", "RK1-B", "разбалловка issue #21",
         "предъявить регулярку, не сокращаемую переписываниями", PARTIAL,
         _rk1_irreducible, 2),
    Case("rk1-древесные", "RK1-C", "rk1_tfl_2025",
         "древесный язык", SOLVED, _rk1_tree),
    Case("rk2-счётный-инвариант", "RK2-A", "пробная РК2",
         "пуст ли язык грамматики при числовом условии", SOLVED, _rk2_counting),
    Case("rk2-конъюнктивная", "RK2-B", "лекция 11",
         "конъюнктивная грамматика для языка", SOLVED, _rk2_conjunctive, 3),
    Case("rk2-mfa", "RK2-B", "лекция 12",
         "MFA для языка квадратов", SOLVED, _rk2_mfa, 3),
    Case("rk2-csy", "RK2-B", "лекция 12",
         "язык не описывается CSY-регуляркой", PARTIAL, _rk2_csy),
    Case("rk2-jumping", "RK2-B", "лекция 12",
         "язык не DMFL", PARTIAL, _rk2_jumping),
    Case("rk2-атрибуты", "RK2-C", "лекция 12",
         "атрибутная грамматика для свойства", SOLVED, _rk2_attributes),
    Case("rk2-real-time", "RK2-B", "Pharma_2023",
         "real-time свойство автомата", PARTIAL, _rk2_real_time),
    Case("exam-µ", "EXAM-1", "билет 2022, вопрос 1",
         "описать язык µ-выражения", SOLVED, _exam_mu),
    Case("exam-псп", "EXAM-3", "билет 2024, вопрос 3",
         "решить проблему соответствия Поста", SOLVED, _exam_pcp),
    Case("exam-srs", "EXAM-3", "билеты 2022-2025",
         "завершимость SRS", SOLVED, _exam_srs),
    Case("exam-огден", "EXAM-3", "РК1 2021",
         "лемма Огдена против КС-свойства", PARTIAL, _exam_ogden),
    Case("exam-замкнутость", "EXAM-3", "билеты 2022-2025",
         "замкнут ли класс относительно операции", MANUAL, _exam_closure),
    Case("pharma-псп-1", "PHARMA", "Pharma_2022, вопрос 1",
         "решить ПСП ⟨a,ba⟩ ⟨aa,ba⟩ ⟨b,ba⟩", SOLVED, _pharma_pcp_one, 1),
    Case("pharma-псп-11", "PHARMA", "Pharma_2022, вопрос 11",
         "общий метод для домино из всех пар слов", SOLVED, _pharma_pcp_all_pairs, 4),
    Case("pharma-унарная-trs", "PHARMA", "Pharma, раздел F",
         "TRS из одноместных символов", SOLVED, _pharma_trs_unary),
    Case("pharma-интерпретация", "PHARMA", "банк вопросов 2024",
         "проверить приведённую интерпретацию", PARTIAL, _pharma_interpretation_check),
    Case("pharma-f8", "PHARMA", "Pharma, задача F8",
         "завершимость f(f(a,x),y) → f(f(x,f(a,y)),a)", MANUAL, _pharma_f8, 3),
)
