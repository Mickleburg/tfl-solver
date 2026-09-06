"""Тесты систем переписывания строк.

Сквозная тема файла — три исхода вместо двух. Почти каждое свойство SRS
неразрешимо, поэтому проверки обязаны отличать «опровергнуто» от «не удалось
выяснить в пределах бюджета». Тесты ниже фиксируют именно это различие:
несколько из них проверяют, что модуль **отказывается** делать вывод.
"""

from __future__ import annotations

import pathlib

import pytest

from tfl.srs import Rule, SRS, Verdict, parse_srs, shortlex_key

VARIANT_20 = pathlib.Path(__file__).parent.parent / "evals" / "lab1_2025" / "variant-20.srs"


@pytest.fixture(scope="module")
def v20() -> SRS:
    return parse_srs(VARIANT_20.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v20_no_eps(v20: SRS) -> SRS:
    """Вариант 20 без правила `babc → ε` — как во второй половине отчёта boomhaa."""
    return SRS(tuple(r for r in v20.rules if r.rhs), v20.alphabet)


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arrow", ["->", "→", "-->", "=>"])
def test_parse_arrows(arrow):
    assert parse_srs(f"ab {arrow} ba").rules == (Rule("ab", "ba"),)


@pytest.mark.parametrize("eps", ["ε", "eps", ".", "λ"])
def test_parse_epsilon_tokens(eps):
    assert parse_srs(f"ab -> {eps}").rules == (Rule("ab", ""),)


def test_parse_empty_rhs():
    assert parse_srs("ab ->").rules == (Rule("ab", ""),)


def test_parse_skips_comments_and_blanks():
    assert len(parse_srs("# коммент\n\nab -> ba\n")) == 1


def test_empty_lhs_rejected():
    with pytest.raises(ValueError):
        parse_srs("ε -> ab")
    with pytest.raises(ValueError):
        Rule("", "a")


def test_parse_requires_arrow():
    with pytest.raises(ValueError):
        parse_srs("ab ba")


def test_variant_20_parses(v20: SRS):
    assert len(v20) == 19
    assert v20.alphabet == frozenset("abc")
    assert Rule("babc", "") in v20.rules


# --------------------------------------------------------------------------
# Переписывание
# --------------------------------------------------------------------------


def test_overlapping_redexes_all_found():
    """Позиции вхождений должны учитывать наложения: в `aaa` редекс `aa` дважды."""
    rule = Rule("aa", "b")
    assert list(rule.positions("aaa")) == [0, 1]
    assert SRS((rule,)).step("aaa") == {"ba", "ab"}


def test_normal_form_detection():
    srs = parse_srs("aa -> a")
    assert srs.is_normal_form("ab")
    assert not srs.is_normal_form("aab")


def test_reachable_reports_exactness():
    """Флаг `exact` — единственное, что отличает вывод от догадки."""
    shrinking = parse_srs("aa -> a")
    assert shrinking.reachable("aaaa", max_len=4).exact

    growing = parse_srs("a -> aa")
    assert not growing.reachable("a", max_len=6).exact


def test_normal_forms_of_confluent_system():
    srs = parse_srs("aa -> a")
    nfs = srs.normal_forms("aaaa", max_len=4)
    assert nfs.words == {"a"}
    assert nfs.exact


# --------------------------------------------------------------------------
# Завершимость
# --------------------------------------------------------------------------


def test_shortlex_key_orders_by_length_then_letters():
    assert shortlex_key("ab", "abc") < shortlex_key("aaa", "abc")
    assert shortlex_key("ab", "abc") < shortlex_key("ba", "abc")


def test_decreasing_order_proves_termination():
    verdict = parse_srs("ba -> ab").terminates(max_len=6, context=1)
    assert verdict.value is True
    assert "армейск" in verdict.reason


def test_variant_20_cycle_matches_boomhaa(v20: SRS):
    """Незавершимость варианта 20 — тот же цикл, что в отчёте boomhaa.

    Они записали его как `caba → cba → baa → caba`; найденный отличается лишь
    точкой входа в цикл.
    """
    cycle = v20.find_cycle(max_len=8, context=1)
    assert cycle is not None
    assert cycle[0] == cycle[-1], "цикл должен замыкаться"
    assert set(cycle) == {"cba", "baa", "caba"}


def test_variant_20_not_terminating(v20: SRS):
    verdict = v20.terminates(max_len=8, context=1)
    assert verdict.value is False
    assert "цикл" in verdict.reason


def test_no_order_found_is_not_a_refutation():
    """Отсутствие убывающего порядка не означает незавершимости."""
    verdict = parse_srs("ab -> ba\nba -> ab").decreasing_under("ab")
    assert verdict.value is None
    assert "не доказывает незавершимость" in verdict.reason


# --------------------------------------------------------------------------
# Критические пары и конфлюэнтность
# --------------------------------------------------------------------------


def test_no_critical_pairs_when_no_overlap():
    assert parse_srs("ba -> ab").critical_pairs() == []


def test_critical_pair_from_overlap():
    """Классика: `aa → ε` и `aba → b` накладываются на слове `aaba`."""
    pairs = parse_srs("aa -> ε\naba -> b").critical_pairs()
    criticals = {(c, l, r) for c, l, r, _, _ in pairs}
    assert ("aaba", "ba", "ab") in criticals


def test_variant_20_not_locally_confluent(v20: SRS):
    """Найденный контрпример короче приведённого в отчёте boomhaa (`aabcaaa`)."""
    verdict = v20.locally_confluent(max_len=11)
    assert verdict.value is False
    critical, left, right, _, _ = verdict.witness
    assert critical == "aaac"
    assert {left, right} == {"aac", "aacc"}


def test_joinability_distinguishes_three_outcomes():
    srs = parse_srs("aa -> a")
    assert srs.joinable("aa", "a", max_len=4).value is True

    disjoint = parse_srs("ab -> c\nba -> d")
    assert disjoint.joinable("c", "d", max_len=6).value is False

    growing = parse_srs("a -> aa\nb -> bb")
    assert growing.joinable("a", "b", max_len=6).value is None


# --------------------------------------------------------------------------
# Кнут–Бендикс
# --------------------------------------------------------------------------


def test_completion_of_classic_example():
    """`{aa → ε, aba → b}` пополняется до `{aa → ε, ba → ab}`.

    Правило `aba → b` после пополнения становится лишним: `aba → aab → b`,
    и минимизация его убирает.
    """
    completed, verdict = parse_srs("aa -> ε\naba -> b").complete("ab", max_len=10)
    assert verdict.value is True
    assert set(completed.rules) == {Rule("aa", ""), Rule("ba", "ab")}
    assert completed.locally_confluent(max_len=10).value is True


def test_completion_reports_divergence(v20: SRS):
    """Пополнение варианта 20 расходится — и это должно быть сказано прямо.

    Процедура Кнута–Бендикса в принципе может не завершаться; молча вернуть
    обрезанную систему как «пополненную» значило бы соврать.
    """
    _, verdict = v20.complete("abc", max_rules=30, max_rounds=4, max_len=8)
    assert verdict.value is None
    assert "прервано" in verdict.reason


def test_orientation_by_shortlex():
    oriented = parse_srs("a -> bb").oriented("ab")
    assert oriented.rules == (Rule("bb", "a"),)


# --------------------------------------------------------------------------
# Эквивалентность систем
# --------------------------------------------------------------------------


def test_identical_systems_are_equivalent():
    srs = parse_srs("ab -> ba")
    assert srs.same_equivalence(srs, max_len=3, search_len=6).value is True


def test_boomhaa_minimized_system_is_wrong(v20_no_eps: SRS):
    """Найденная ошибка в чужой работе.

    `boomhaa-tfl-labs/lab1` для варианта 20 без ε-правила приводит
    пополненную систему, содержащую `aa → a`, но при минимизации это правило
    теряется и остаётся `{b → a, c → a}`. Такая система сохраняет длину слова,
    тогда как исходная — нет.

    Свидетель: `a ↔ aa` тремя шагами исходной системы

        a → aabcaa   (обратное к `aabcaa → a`)
          → aaabcaa  (обратное к `aaa → aa`)
          → aa       (правило `aabcaa → a` в позиции 1)

    а класс слова `a` в `{b → a, c → a}` вычисляется целиком и равен
    `{a, b, c}`.
    """
    boomhaa = parse_srs("b -> a\nc -> a")
    verdict = v20_no_eps.same_equivalence(boomhaa, max_len=3, search_len=7)
    assert verdict.value is False
    word, missing = verdict.witness
    assert (word, missing) == ("a", "aa")


def test_class_of_a_in_boomhaa_system_is_exact():
    """Опровержение опирается на полноту обхода — проверяем её отдельно."""
    search = parse_srs("b -> a\nc -> a").symmetric().reachable("a", max_len=7)
    assert search.exact
    assert search.words == {"a", "b", "c"}


def test_path_from_a_to_aa_exists(v20_no_eps: SRS):
    """Прямая проверка свидетеля: `aa` лежит в классе `a` исходной системы."""
    search = v20_no_eps.symmetric().reachable("a", max_len=7)
    assert "aa" in search.words


def test_adding_missing_rule_is_not_refuted(v20_no_eps: SRS):
    """С восстановленным `aa → a` опровержения уже нет.

    Вердикт при этом не «эквивалентны», а «не выяснено»: классы не удалось
    вычислить целиком. Разница существенная, поэтому проверяется явно.
    """
    fixed = parse_srs("b -> a\nc -> a\naa -> a")
    assert v20_no_eps.same_equivalence(fixed, max_len=3, search_len=7).value is None


def test_fuzz_equivalence_follows_task_scheme(v20_no_eps: SRS):
    boomhaa = parse_srs("b -> a\nc -> a")
    verdict = v20_no_eps.fuzz_equivalence(boomhaa, trials=200, seed=1)
    assert verdict.value is False


# --------------------------------------------------------------------------
# Классы эквивалентности и инварианты
# --------------------------------------------------------------------------


def test_equivalence_classes_of_length_preserving_system():
    """`{b → a}` склеивает слова, различающиеся заменой b на a."""
    classes = parse_srs("b -> a").equivalence_classes(max_len=3, slack=0)
    assert classes["bb"] == classes["aa"] == "aa"
    assert classes["a"] != classes["aa"]


def test_invariant_preserved():
    """Число букв сохраняется при `ab → ba`."""
    srs = parse_srs("ab -> ba")
    verdict = srs.check_invariant(lambda w: w.count("a"), ["abab", "aabb", "ba"])
    assert verdict.value is True


def test_invariant_violation_reported():
    srs = parse_srs("aa -> a")
    verdict = srs.check_invariant(len, ["aa", "aaa"])
    assert verdict.value is False


def test_monotone_invariant():
    """Длина не возрастает при `aa → a` — годится как монотонный инвариант."""
    srs = parse_srs("aa -> a")
    assert srs.check_invariant(len, ["aa", "aaa"], monotone=True).value is True


# --------------------------------------------------------------------------
# Защита от неверного использования
# --------------------------------------------------------------------------


def test_verdict_cannot_be_used_as_bool():
    """`if srs.terminates():` молча принял бы «не выяснено» за «да»."""
    with pytest.raises(TypeError, match="три исхода"):
        bool(Verdict(None, "не выяснено"))


# --------------------------------------------------------------------------
# Подбор порядка и линейные инварианты
# --------------------------------------------------------------------------


def test_no_shortlex_order_when_a_rule_lengthens():
    """Шортлекс сравнивает сначала длины, поэтому удлиняющее правило
    не сделать убывающим никаким приоритетом букв."""
    assert parse_srs("a -> bb").find_shortlex_order() is None


def test_shortlex_order_from_equal_length_rules():
    """Правила равной длины дают ограничения на приоритет; порядок находится
    топологической сортировкой, а не перебором перестановок.

    Перебор здесь недопустим: у варианта 12 одиннадцать букв, то есть
    11! ≈ 4·10⁷ перестановок — на этом прогон вставал намертво.
    """
    order = parse_srs("ba -> ab\nca -> ac").find_shortlex_order()
    assert order is not None
    assert order.index("b") > order.index("a")
    assert order.index("c") > order.index("a")
    assert parse_srs("ba -> ab\nca -> ac").decreasing_under(order).value is True


def test_contradictory_constraints_have_no_order():
    """`ba → ab` требует b > a, `ab → ba` требует a > b — вместе невозможно."""
    assert parse_srs("ba -> ab\nab -> ba").find_shortlex_order() is None


def test_identity_rule_has_no_order():
    assert parse_srs("ab -> ab").find_shortlex_order() is None


def test_linear_invariants_of_permutation_system():
    """`ab → ba` переставляет буквы, значит счётчики букв сохраняются точно."""
    invariants = parse_srs("ab -> ba").linear_invariants(5)
    assert len(invariants) == 2
    for vector in invariants:
        check = parse_srs("ab -> ba").check_invariant(
            lambda w, v=vector: sum(c * w.count(x) for x, c in v.items()) % 5,
            ["abab", "aabb", "ba", "bbaa"],
        )
        assert check.value is True


def test_linear_invariant_modulo_two():
    """`aa → ε` меняет число a на два, поэтому чётность |w|_a сохраняется."""
    invariants = parse_srs("aa -> ε").linear_invariants(2)
    assert {"a": 1} in invariants


def test_no_linear_invariants_for_variant_20(v20: SRS):
    """У варианта 20 линейных инвариантов нет — нужны нелинейные.

    `boomhaa` строит для них гомоморфизм в моноид диагональных матриц.
    """
    assert all(not v20.linear_invariants(m) for m in (2, 3, 5, 7))


def test_invariant_search_rejects_composite_modulus():
    with pytest.raises(ValueError, match="простым"):
        parse_srs("ab -> ba").linear_invariants(4)


# --------------------------------------------------------------------------
# Фикстуры вариантов
# --------------------------------------------------------------------------


def test_all_28_variants_present_and_parse():
    files = sorted(VARIANT_20.parent.glob("variant-*.srs"))
    assert len(files) == 28
    for path in files:
        srs = parse_srs(path.read_text(encoding="utf-8"))
        assert srs.rules, f"{path.name}: пустая система"
        assert srs.alphabet, f"{path.name}: пустой алфавит"


def test_extracted_variant_20_matches_boomhaa(v20: SRS):
    """Автоизвлечение из PDF совпало с ручной сверкой по отчёту boomhaa.

    Вёрстка PDF двухколоночная, номер варианта стоит по центру блока,
    поэтому автоматическому разбору нужна независимая проверка.
    """
    expected = [
        ("cb", "ba"), ("aaa", "aa"), ("aba", "ba"), ("ac", "cc"), ("baa", "ba"),
        ("bba", "ba"), ("bbb", "b"), ("bbc", "c"), ("bcc", "cc"), ("ba", "cab"),
        ("cac", "cc"), ("bab", "cac"), ("ccc", "c"), ("babb", "ba"), ("babc", ""),
        ("baca", "cabba"), ("caab", "bb"), ("caac", "bc"), ("aabcaa", "a"),
    ]
    assert [(r.lhs, r.rhs) for r in v20.rules] == expected
