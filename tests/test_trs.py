"""Системы переписывания термов: унификация, переписывание, интерпретации.

Приёмочные примеры: разобранная преподавателем унификация из issue #2
и задачи «Аптеки» из раздела с TRS.
"""

from __future__ import annotations

import pytest

from tfl.trs import (
    lpo_greater,
    Poly,
    TRS,
    Rule,
    app,
    match,
    parse_interpretation,
    parse_term,
    parse_trs,
    substitute,
    unify,
    var,
)
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Переменные объявляются, а не угадываются
# --------------------------------------------------------------------------


def test_two_opposite_conventions_live_in_the_corpus():
    """В issue #2 переменные строчные, в «Аптеке» — заглавные.

    Один и тот же текст читается по-разному, и разница не косметическая:
    от неё зависит, что вообще является правилом.
    """
    lowercase = parse_term("F(q, q, q)", {"q"})
    assert lowercase.variables() == {"q"}
    assert lowercase.symbols() == {"F": 3}

    uppercase = parse_term("w(w(s, X), Y)", {"X", "Y"})
    assert uppercase.variables() == {"X", "Y"}
    assert uppercase.symbols() == {"w": 2, "s": 0}


def test_variable_cannot_take_arguments():
    with pytest.raises(ValueError, match="не может иметь аргументов"):
        parse_term("x(a)", {"x"})


def test_numeric_constants_are_symbols():
    """`plus(0, y) → y` — обычная запись, и `0` тут константа."""
    term = parse_term("plus(0, y)", {"y"})
    assert term.symbols() == {"plus": 2, "0": 0}


# --------------------------------------------------------------------------
# Унификация: разобранный пример преподавателя
# --------------------------------------------------------------------------


def test_unification_reproduces_the_teachers_example():
    """Пара `F(q,q,q)` и `F(F(x,y,R), F(a,w,a), F(w,x,y))` из issue #2.

    Ответ преподавателя записан мультиуравнениями: `{q} := F(x,y,a)`,
    `{x,a,w,y} = R`. В обычной записи это `q ↦ F(R,R,R)`, а `x`, `y`,
    `a`, `w` — все `R`.
    """
    names = {"q", "x", "y", "a", "w"}
    left = parse_term("F(q,q,q)", names)
    right = parse_term("F(F(x,y,R), F(a,w,a), F(w,x,y))", names)

    mgu = unify(left, right)
    assert mgu is not None
    assert str(mgu["q"]) == "F(R, R, R)"
    assert {str(mgu[n]) for n in ("x", "y", "a", "w")} == {"R"}
    assert substitute(left, mgu) == substitute(right, mgu)


def test_occurs_check():
    """Без проверки вхождения `x` и `F(x)` «унифицируются» в бесконечный терм."""
    assert unify(parse_term("x", {"x"}), parse_term("F(x)", {"x"})) is None


def test_unification_fails_on_different_heads():
    assert unify(parse_term("F(a)", set()), parse_term("G(a)", set())) is None


def test_matching_is_one_way():
    """Сопоставление подставляет только в образец — терм не трогается.

    Унификация здесь была бы ошибкой: она позволила бы «доопределить»
    переписываемый терм и применить правило там, где оно не применимо.
    """
    pattern = parse_term("f(a, x)", {"x"})
    assert match(pattern, parse_term("f(a, b)", set())) == {"x": parse_term("b", set())}
    assert match(pattern, parse_term("f(y, b)", {"y"})) is None
    assert unify(pattern, parse_term("f(y, b)", {"x", "y"})) is not None


def test_nonlinear_pattern_requires_equal_subterms():
    """`f(x, x)` подходит только к терму с двумя одинаковыми аргументами."""
    pattern = parse_term("f(x, x)", {"x"})
    assert match(pattern, parse_term("f(a, a)", set())) is not None
    assert match(pattern, parse_term("f(a, b)", set())) is None


# --------------------------------------------------------------------------
# Правила и переписывание
# --------------------------------------------------------------------------


def test_free_variables_on_the_right_are_rejected():
    """`f(x) → g(y)` не задаёт переписывания: что подставлять вместо `y`?"""
    with pytest.raises(ValueError, match="свободные переменные"):
        parse_trs("variables = [x, y]\nf(x) -> g(y)")


def test_rewriting_applies_at_every_position():
    system = parse_trs("variables = [x]\nf(x) -> g(x)")
    term = parse_term("f(f(a))", set())
    got = {str(t) for t in system.step(term)}
    assert got == {"g(f(a))", "f(g(a))"}


def test_normal_form():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    term = parse_term("plus(s(s(0)), s(0))", set())
    assert str(system.normal_form(term)) == "s(s(s(0)))"


# --------------------------------------------------------------------------
# Завершимость
# --------------------------------------------------------------------------


def test_loop_is_found_when_the_term_grows():
    """`f(x) → f(f(x))` не возвращает терм к себе — растёт, но содержит его."""
    verdict = parse_trs("variables = [x]\nf(x) -> f(f(x))").terminates()
    assert verdict.value is False
    assert "найдена петля" in verdict.reason


def test_cycle_is_called_a_cycle():
    verdict = parse_trs("variables = []\na -> a").terminates()
    assert verdict.value is False
    assert "найден цикл" in verdict.reason


def test_peano_addition_terminates_two_ways():
    """Сложение по Пеано завершимо, и это видно двумя независимыми способами.

    Общий вердикт приходит от LPO: путевой порядок пробуется раньше
    интерпретаций, потому что даёт сертификат в одну строчку — старшинство
    символов. Интерпретация находится тоже, и её проверяем отдельно.
    """
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    verdict = system.terminates()
    assert verdict.value is True
    assert "путевом порядке" in verdict.reason

    found = system.find_interpretation()
    assert found.value is True
    assert "интерпретация" in found.reason


def test_a_supplied_interpretation_can_be_checked():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    good = parse_interpretation("plus(x, y) = 2*x + y + 1; s(x) = x + 1; 0 = 1")
    assert system.check_interpretation(good).value is True

    bad = parse_interpretation("plus(x, y) = x + y; s(x) = x + 1; 0 = 1")
    verdict = system.check_interpretation(bad)
    assert verdict.value is False
    assert "не убывает" in verdict.reason


def test_question_bank_interpretation_does_not_decrease():
    """Банк вопросов 2024: `f(x) => g(x,x)` при `f(x) := x+1; g(x,y) := x + y*2 + 1`.

    Интерпретация в том виде, в каком она выписана в банке, правило
    **не уменьшает**: `[f(x)] = x + 1`, а `[g(x,x)] = 3x + 1`. Сама система
    при этом завершима — правая часть не содержит `f`, — и оракул находит
    для неё другую интерпретацию.
    """
    system = parse_trs("variables = [x]\nf(x) -> g(x, x)")
    quoted = parse_interpretation("f(x) := x+1; g(x,y) := x + y*2 + 1")
    verdict = system.check_interpretation(quoted)
    assert verdict.value is False
    assert "[f(x)] = x + 1" in verdict.reason

    assert system.terminates().value is True


# --------------------------------------------------------------------------
# Полиномы
# --------------------------------------------------------------------------


def test_polynomial_arithmetic():
    x, y = Poly.of("x"), Poly.of("y")
    assert str(x * y + x + Poly.number(2)) == "x*y + x + 2"
    assert str((x + Poly.number(1)) * (x + Poly.number(1))) == "x^2 + 2*x + 1"
    assert (x * y).constant() == 0


def test_domination_needs_a_positive_constant_and_no_negative_coefficients():
    x = Poly.of("x")
    assert (x + Poly.number(2)).dominates(x + Poly.number(1))
    assert not (x + Poly.number(2)).dominates(x + Poly.number(2))
    assert not (x + Poly.number(5)).dominates(x * Poly.number(2))


# --------------------------------------------------------------------------
# Мост в строковые системы
# --------------------------------------------------------------------------


def test_unary_system_becomes_a_string_system():
    """Все символы одноместные — значит это переписывание строк.

    Так устроена задача F6 «Аптеки» (2022, 2023 и 2024 годов): семь правил
    над `E`, `Q`, `q`, `W`, все унарные, и вся строковая машинерия
    к ней применима без изменений.
    """
    system = parse_trs("variables = [X]\nE(Q(q(X))) -> Q(W(X))\nq(E(X)) -> W(X)")
    assert system.is_unary()
    strings = system.as_srs()
    assert [str(rule) for rule in strings.rules] == ["EQq → QW", "qE → W"]


def test_bridge_preserves_rewriting():
    """Шаг в термах и шаг в строках дают одно и то же — это и проверяется."""
    system = parse_trs("variables = [X]\nE(Q(X)) -> Q(E(E(X)))\nq(E(X)) -> W(X)")
    strings = system.as_srs()

    def to_term(word: str):
        term = var("X")
        for letter in reversed(word):
            term = app(letter, term)
        return term

    for word in iter_words("EQqW", 5):
        by_terms = {str(t) for t in system.step(to_term(word))}
        by_strings = {str(to_term(w)) for w in strings.step(word)}
        assert by_terms == by_strings, word


def test_binary_system_has_no_bridge():
    system = parse_trs("variables = [x, y]\nf(x, y) -> f(y, x)")
    assert not system.is_unary()
    with pytest.raises(ValueError, match="только для унарных"):
        system.as_srs()


# --------------------------------------------------------------------------
# Мартелли–Монтанари, алгоритм 3
# --------------------------------------------------------------------------


def test_common_part_and_frontier_from_the_lab_statement():
    """Разобранный пример из условия ЛР1 2022.

    > У мультиуравнения `{x₁, x₂} = (f(g(x₃), h(x₄, g(x₅))), f(x₄, h(g(g(x₆)), x₇)))`
    > общая часть — это `f(x₄, h(x₄, x₇))`, граница — это
    > `{{x₄} = (g(x₃), g(g(x₆))), {x₇} = g(x₅)}`.
    """
    from tfl.trs import common_part

    names = {f"x{i}" for i in range(1, 9)}
    terms = (
        parse_term("f(g(x3), h(x4, g(x5)))", names),
        parse_term("f(x4, h(g(g(x6)), x7))", names),
    )
    part, frontier = common_part(terms)
    assert str(part) == "f(x4, h(x4, x7))"

    got = {
        (tuple(sorted(e.variables)), tuple(sorted(str(t) for t in e.terms)))
        for e in frontier
    }
    assert got == {
        (("x4",), ("g(g(x6))", "g(x3)")),
        (("x7",), ("g(x5)",)),
    }


def test_common_part_stops_at_a_variable():
    """Если в позиции стоит переменная, конструктор в общую часть не идёт."""
    from tfl.trs import common_part

    part, frontier = common_part(
        (parse_term("f(g(a), g(b))", set()), parse_term("f(x, g(b))", {"x"}))
    )
    assert str(part) == "f(x, g(b))"
    assert [str(e) for e in frontier] == ["{x} = (g(a))"]


def test_no_common_part_when_heads_clash():
    from tfl.trs import common_part

    part, frontier = common_part((parse_term("F(a)", set()), parse_term("G(a)", set())))
    assert part is None and frontier == []


def test_algorithm_three_on_the_teachers_example():
    """Тот же пример из issue #2, но проверяется форма ответа, а не только он.

    Преподаватель записывает результат как «терм `F(q,q,q)` с подстановками
    `{q} := F(x,y,a)`, `{x,a,w,y} = R`». Представитель внутри слитого класса
    переменных может быть выбран иначе — важно, что классы те же
    и подстановка та же.
    """
    from tfl.trs import substitution_of, unified_term, unify_mm

    names = {"q", "x", "y", "a", "w"}
    left = parse_term("F(q,q,q)", names)
    right = parse_term("F(F(x,y,R), F(a,w,a), F(w,x,y))", names)

    system = unify_mm(left, right)
    assert system is not None
    classes = {tuple(sorted(e.variables)) for e in system}
    assert ("a", "w", "x", "y") in classes
    assert ("q",) in classes

    binding = substitution_of(system)
    assert str(binding["q"]) == "F(R, R, R)"
    assert {str(binding[n]) for n in ("x", "y", "a", "w")} == {"R"}
    assert substitute(left, binding) == substitute(right, binding)
    assert str(unified_term(system)) == "F(F(R, R, R), F(R, R, R), F(R, R, R))"


def test_algorithm_three_rejects_a_cycle():
    """Переменная встречается в правой части своего же уравнения — неудача.

    Это и есть проверка вхождения в терминах системы мультиуравнений:
    выбрать уравнение, переменные которого не заняты, становится нельзя.
    """
    from tfl.trs import unify_mm

    assert unify_mm(parse_term("x", {"x"}), parse_term("F(x)", {"x"})) is None


def test_algorithm_three_agrees_with_plain_unification():
    """Два независимо написанных унификатора обязаны совпадать."""
    from tfl.trs import substitution_of, unify_mm

    names = {"x", "y", "z"}
    samples = ["x", "F(x)", "F(y)", "G(x, y)", "G(F(x), y)", "G(z, G(x, y))", "F(G(x, x))"]
    for first in samples:
        for second in samples:
            left, right = parse_term(first, names), parse_term(second, names)
            plain = unify(left, right)
            system = unify_mm(left, right)
            assert (plain is None) == (system is None), (first, second)
            if system is not None:
                binding = substitution_of(system)
                assert substitute(left, binding) == substitute(right, binding), (first, second)


# --------------------------------------------------------------------------
# Критические пары
# --------------------------------------------------------------------------


def test_overlap_in_a_variable_gives_no_critical_pair():
    """`f(f(x)) → x` накладывается сама на себя только в переменной."""
    system = parse_trs("variables = [x]\nf(f(x)) -> x")
    assert system.critical_pairs() == []
    assert system.locally_confluent().value is True


def test_two_rules_for_one_term_are_not_confluent():
    system = parse_trs("variables = []\na -> b\na -> c")
    verdict = system.locally_confluent()
    assert verdict.value is False
    assert "не сходится" in verdict.reason


def test_overlap_inside_the_left_side():
    """`f(g(x)) → x` и `g(a) → b` накладываются в позиции (0,)."""
    system = parse_trs("variables = [x]\nf(g(x)) -> x\ng(a) -> b")
    pairs = system.critical_pairs()
    assert len(pairs) == 1
    assert (str(pairs[0].left), str(pairs[0].right)) == ("a", "f(b)")
    assert pairs[0].position == (0,)
    assert system.locally_confluent().value is False


def test_joinable_reports_the_common_descendant():
    system = parse_trs("variables = []\na -> c\nb -> c")
    verdict = system.joinable(parse_term("a", set()), parse_term("b", set()))
    assert verdict.value is True
    assert str(verdict.witness) == "c"


def test_joinability_of_a_growing_system_is_not_refuted():
    """Обход обрезан по размеру — значит «не выяснено», а не «не сходятся»."""
    system = parse_trs("variables = [x]\nf(x) -> f(f(x))")
    verdict = system.joinable(
        parse_term("f(a)", set()), parse_term("g(a)", set()), max_size=6
    )
    assert verdict.value is None
    assert "обход обрезан" in verdict.reason


# --------------------------------------------------------------------------
# Рекурсивный путевой порядок
# --------------------------------------------------------------------------


def test_term_is_greater_than_its_own_subterm():
    """Свойство подтерма: `f(x) > x`. Отсюда же `Var(r) ⊆ Var(l)` для правил."""
    assert lpo_greater(parse_term("f(x)", "x"), parse_term("x", "x"), "f")
    assert not lpo_greater(parse_term("x", "x"), parse_term("f(x)", "x"), "f")


def test_a_variable_it_does_not_contain_is_incomparable():
    """`f(x)` и `y` несравнимы: правило `f(x) → y` не ориентируется никогда."""
    left, right = parse_term("f(x)", "x y"), parse_term("y", "x y")
    assert not lpo_greater(left, right, "f")
    assert not lpo_greater(right, left, "f")


def test_precedence_decides_when_heads_differ():
    """Случай 2: старший символ побеждает, если больше каждого аргумента."""
    left, right = parse_term("f(x)", "x"), parse_term("g(x)", "x")
    assert lpo_greater(left, right, "f g")
    assert lpo_greater(right, left, "g f")


def test_equal_heads_are_compared_lexicographically():
    """Случай 3: головы совпали — идём по аргументам слева направо."""
    left = parse_term("f(g(a), a)", "")
    right = parse_term("f(a, g(a))", "")
    assert lpo_greater(left, right, "g f a")
    assert not lpo_greater(right, left, "g f a")


def test_self_embedding_rule_is_beyond_any_path_order():
    """`f(x) → g(f(x))`: правая часть содержит левую, порядка нет ни одного.

    LPO — упрощающий порядок, а у упрощающего порядка терм всегда меньше
    того, в который он вложен. Никакой перебор старшинства тут не поможет,
    и `find_precedence` обязан честно вернуть `None`.
    """
    system = parse_trs("variables = [x]\nf(x) -> g(f(x))")
    assert system.find_precedence() is None
    assert system.lpo_terminates("f g").value is None


def test_lpo_proves_termination_of_peano_addition():
    system = parse_trs("variables = [x, y]\nplus(0, y) -> y\nplus(s(x), y) -> s(plus(x, y))")
    order = system.find_precedence()
    assert order is not None
    assert system.lpo_terminates(order).value is True


def test_orientation_flips_a_rule_and_reports_what_it_cannot():
    """Коммутативность не ориентируется ни одним порядком редукции.

    Обе части получаются друг из друга подстановкой, поэтому «меньше»
    и «больше» тут не определить в принципе. Это не изъян LPO, а причина,
    по которой существует пополнение по модулю AC.
    """
    flipped, stuck = parse_trs("variables = [x]\ng(x) -> f(x)").oriented("f g")
    assert [str(r) for r in flipped.rules] == ["f(x) → g(x)"]
    assert stuck == ()

    _, stuck = parse_trs("variables = [x, y]\nf(x, y) -> f(y, x)").oriented("f")
    assert len(stuck) == 1


# --------------------------------------------------------------------------
# Пополнение по Кнуту–Бендиксу
# --------------------------------------------------------------------------


def group_axioms():
    return parse_trs(
        """
        variables = [x, y, z]
        f(f(x, y), z) -> f(x, f(y, z))
        f(e, x) -> x
        f(i(x), x) -> e
        """
    )


def test_group_axioms_complete_to_the_canonical_ten_rules():
    """Хрестоматийная контрольная точка: свободная группа.

    Три аксиомы пополняются до канонической системы из десяти правил,
    в которой каждое слово имеет единственную нормальную форму. Сверять
    тут есть с чем: это самый разобранный в литературе пример пополнения.
    """
    done, verdict = group_axioms().complete()
    assert verdict.value is True, verdict.reason
    assert len(done) == 10

    printed = {str(rule) for rule in done.rules}
    assert "f(x′, e) → x′" in printed  # правая единица выведена из левой
    assert "i(e) → e" in printed
    assert "i(i(x′′′)) → x′′′" in printed
    assert "i(f(x′′′, y′′′)) → f(i(y′′′), i(x′′′))" in printed  # инверсия произведения


def test_the_completed_system_is_canonical():
    done, _ = group_axioms().complete()
    assert done.terminates().value is True
    assert done.locally_confluent().value is True


def test_the_choice_of_precedence_is_not_a_detail():
    """Первый же порядок, ориентирующий аксиомы, пополнение не доводит.

    При `f ≻ e ≻ i` дело упирается в уравнение `i(f(x,y)) = f(i(y),i(x))`:
    ориентировать его в этом порядке нельзя ни в какую сторону. Поэтому
    `complete` перебирает кандидатов, а не берёт первого попавшегося.
    """
    system = group_axioms()
    assert system.find_precedence() == "f e i"

    _, verdict = system.complete("f e i")
    assert verdict.value is None
    assert "не ориентируется" in verdict.reason

    _, good = system.complete("i f e")
    assert good.value is True


def test_associativity_alone_is_already_canonical():
    system = parse_trs("variables = [x, y, z]\nf(f(x, y), z) -> f(x, f(y, z))")
    done, verdict = system.complete()
    assert verdict.value is True
    assert len(done) == 1


def test_completion_of_a_convergent_pair():
    """`f(f(x)) → g(x)` пополняется одним правилом перестановки."""
    done, verdict = parse_trs("variables = [x]\nf(f(x)) -> g(x)").complete()
    assert verdict.value is True
    assert {str(r) for r in done.rules} == {"f(f(x)) → g(x)", "f(g(x′)) → g(f(x′))"}


def test_commutativity_stops_completion_honestly():
    _, verdict = parse_trs("variables = [x, y]\nf(x, y) -> f(y, x)").complete()
    assert verdict.value is None
    assert "не ориентируется" in verdict.reason


def test_hitting_the_limit_is_not_a_failure_verdict():
    """Расходимость — штатный исход полуразрешающей процедуры, а не сбой."""
    _, verdict = group_axioms().complete("i f e", max_rules=5)
    assert verdict.value is None
    assert "расходиться" in verdict.reason


def test_interreduction_drops_a_derivable_rule():
    """`f(f(x)) → x` выводимо из `f(x) → x` и в системе не нужно."""
    system = parse_trs("variables = [x]\nf(x) -> x\nf(f(x)) -> x")
    assert [str(r) for r in system.minimized().rules] == ["f(x) → x"]


# --------------------------------------------------------------------------
# Обратный мост: строки как унарные термы
# --------------------------------------------------------------------------


def test_a_string_system_becomes_a_unary_term_system():
    from tfl.srs import parse_srs
    from tfl.trs import from_srs

    string_system = parse_srs("ab -> bba\nb -> a")
    terms = from_srs(string_system)
    assert [str(r) for r in terms.rules] == ["a(b(X)) → b(b(a(X)))", "b(X) → a(X)"]
    assert terms.is_unary()
    assert terms.as_srs() == string_system


def test_rewriting_agrees_on_both_sides_of_the_bridge():
    """Шаг по строке и шаг по цепочке термов дают одно и то же.

    Подстановка в `X` — это суффикс слова, контекст над цепочкой — префикс,
    поэтому позиции применения правила совпадают ровно.
    """
    from tfl.srs import parse_srs
    from tfl.trs import from_srs

    string_system = parse_srs("ab -> ba")
    terms = from_srs(string_system)
    for word in ("abab", "aabb", "ba", "abba"):
        by_string = set(string_system.step(word))
        by_terms = {str(t).replace("(", "").replace(")", "").replace(", ", "")[:-1]
                    for t in terms.step(parse_term("".join(f"{c}(" for c in word) + "X"
                                                   + ")" * len(word), "X"))}
        assert by_string == by_terms, word


def test_the_path_order_orients_a_lengthening_string_rule():
    """`ab → bba` длиннее слева направо, и путевому порядку это не мешает."""
    from tfl.srs import parse_srs
    from tfl.trs import from_srs

    assert from_srs(parse_srs("ab -> bba")).find_precedence() == "a b"
    verdict = parse_srs("ab -> bba").terminates()
    assert verdict.value is True
