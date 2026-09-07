"""Проблема соответствия Поста: поиск решения и счётные опровержения.

Приёмочные точки — разобранный преподавателем пример из ЛР0 2023
(с выписанными уравнениями на числа домино), два экземпляра «Аптеки» 2022
и вопрос 3 экзаменационного билета 2024. Контроль в другую сторону —
классический решаемый экземпляр Хопкрофта: метод обязан не отсечь его.
"""

from __future__ import annotations

from tfl.pcp import PCP, Domino, parse_pcp

# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


def test_dominoes_are_read_in_both_notations():
    assert parse_pcp("(ab,a)\n⟨aa, b⟩").dominoes == (Domino("ab", "a"), Domino("aa", "b"))


def test_empty_word_is_written_as_epsilon_or_nothing():
    assert parse_pcp("(a,ε)\n(b,)").dominoes == (Domino("a", ""), Domino("b", ""))


def test_printing_marks_the_empty_word():
    assert str(parse_pcp("(a,)")) == "⟨a, ε⟩"


# --------------------------------------------------------------------------
# Поиск решения
# --------------------------------------------------------------------------


def test_the_classic_solvable_instance():
    """`⟨1,101⟩ ⟨10,00⟩ ⟨011,11⟩` — учебный пример с решением 1 3 2 3."""
    verdict = parse_pcp("(1,101)\n(10,00)\n(011,11)").solve()
    assert verdict.value is True
    assert verdict.witness == (0, 2, 1, 2)
    assert "101110011" in verdict.reason


def test_a_solution_is_checked_not_taken_on_faith():
    instance = parse_pcp("(1,101)\n(10,00)\n(011,11)")
    assert instance.is_solution(instance.solve().witness)
    assert not instance.is_solution((0, 1, 2))
    assert not instance.is_solution(())


def test_a_diagonal_domino_solves_the_task_alone():
    """Вопрос 11 «Аптеки» 2022: домино — **все** пары слов множества.

    Общий метод тут состоит из одного наблюдения: среди всех пар есть
    диагональные `⟨wᵢ, wᵢ⟩`, а такое домино само по себе решение. Значит,
    для множеств такого вида задача разрешима всегда и решается за один шаг.
    """
    instance = PCP.all_pairs(["ab", "b"])
    verdict = instance.solve()
    assert verdict.value is True
    assert len(verdict.witness) == 1
    assert instance.is_solution(verdict.witness)


def test_search_never_returns_a_refutation():
    """Из «не нашли» не следует «нет»: задача неразрешима.

    Оракул обязан вернуть «не выяснено» и сказать, до какой длины
    решение исключено, — это и есть то, что можно писать в отчёт.
    """
    verdict = parse_pcp("(a,ba)\n(aa,ba)\n(b,ba)").solve(max_dominoes=12)
    assert verdict.value is None
    assert "домино" in verdict.reason


# --------------------------------------------------------------------------
# Счётные уравнения — метод преподавателя из ЛР0 2023
# --------------------------------------------------------------------------


def test_the_worked_example_gives_exactly_the_equations_of_the_statement():
    """> для пары домино выше в каждом решении должны выполняться равенства:
    > (a) `2·M₂ = M₃`  (b) `M₁ + 2·M₃ = M₂`
    """
    instance = parse_pcp("(ab,a)\n(aa,b)\n(bb,a)")
    printed = instance.equations_markdown()
    assert "$2 M_2 = M_3$" in printed
    assert "$M_1 + 2 M_3 = M_2$" in printed


def test_the_worked_example_is_refuted_by_counting():
    """> В целых положительных значениях Mi решений этой системы
    > не существует, поэтому ПСП решения не имеет.
    """
    verdict = parse_pcp("(ab,a)\n(aa,b)\n(bb,a)").refute_by_counting()
    assert verdict.value is False
    assert "не имеет неотрицательных ненулевых решений" in verdict.reason


def test_both_pharma_instances_fall_to_counting():
    """«Аптека» 2022, вопросы 1 и 2 — по баллу каждый."""
    for text in ("(a,ba)\n(aa,ba)\n(b,ba)", "(a,ba)\n(b,bab)\n(a,ε)"):
        assert parse_pcp(text).refute_by_counting().value is False


def test_counting_does_not_refute_a_solvable_instance():
    """Контроль: у решаемого экземпляра уравнения обязаны быть совместны."""
    verdict = parse_pcp("(1,101)\n(10,00)\n(011,11)").refute_by_counting()
    assert verdict.value is None
    assert "не запрещают" in verdict.reason


def test_nonnegative_is_the_right_condition_not_positive():
    """Решение не обязано использовать все домино, и это меняет вывод.

    Читать «в целых **положительных** значениях» буквально нельзя:
    у системы могут отсутствовать положительные решения при наличии
    неотрицательных, и тогда опровержение по строгому условию было бы
    неверным. Здесь так и есть: уравнение по букве `b` заставляет `M₂ = 0`,
    положительных решений нет вовсе, а решение ПСП есть — первое домино.
    """
    instance = parse_pcp("(a,a)\n(b,bb)")
    verdict = instance.refute_by_counting()
    assert verdict.value is None
    assert "положительных" in verdict.reason
    assert instance.solve().value is True


# --------------------------------------------------------------------------
# Усиление уравнениями на пары букв
# --------------------------------------------------------------------------


def test_adjacency_equations_narrow_the_endpoints():
    """Вопрос 3 билета 2024: счётные уравнения молчат, парные — говорят.

    `⟨ab,bba⟩ ⟨aba,a⟩ ⟨b,ba⟩` уравнения на буквы не опровергают: они дают
    `M₁ = M₂ = M₃`. Уравнения на пары букв вместе с балансом соседств
    отсеивают восемь вариантов из девяти: первым и последним может быть
    только домино 2.
    """
    instance = parse_pcp("(ab,bba)\n(aba,a)\n(b,ba)")
    assert instance.refute_by_counting().value is None

    verdict = instance.refute_by_adjacency()
    assert verdict.value is None
    assert verdict.witness == [(2, 2)]


def test_adjacency_keeps_the_endpoints_of_a_real_solution():
    """Контроль на решаемом экземпляре: настоящие крайние домино уцелели.

    Решение Хопкрофта — `1 3 2 3`, значит пара «первое…последнее» это `1…3`.
    Метод обязан её оставить, иначе он неверен.
    """
    verdict = parse_pcp("(1,101)\n(10,00)\n(011,11)").refute_by_adjacency()
    assert verdict.value is None
    assert (1, 3) in verdict.witness


def test_adjacency_refutes_what_counting_refutes():
    for text in ("(ab,a)\n(aa,b)\n(bb,a)", "(a,ba)\n(aa,ba)\n(b,ba)"):
        assert parse_pcp(text).refute_by_adjacency().value is False


def test_an_empty_word_disables_the_adjacency_method():
    """Стыковая биграмма у пустого слова не определена, и метод честно молчит."""
    verdict = parse_pcp("(a,ba)\n(b,bab)\n(a,ε)").refute_by_adjacency()
    assert verdict.value is None
    assert "пустое слово" in verdict.reason


# --------------------------------------------------------------------------
# SMT-модель по условию ЛР0 2023
# --------------------------------------------------------------------------


def test_smtlib_model_declares_both_families_of_variables():
    model = parse_pcp("(ab,a)\n(aa,b)\n(bb,a)").smtlib_model()
    assert "(declare-const M1 Int)" in model
    assert "(declare-const N2_3 Int)" in model
    assert "(check-sat)" in model
    assert model.count("(assert") > 10
