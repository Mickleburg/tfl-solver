"""Атрибутные грамматики: разбор, вычисление, язык.

Условия взяты из `corpus/txt/FormalLanguageTheory_2025_rk2_tfl_2025.txt`
и сверены с обезличенными контрольными примерами.
"""

from __future__ import annotations

import pytest

from tfl.attr import AttrGrammar, Ref, parse_attr_grammar
from tfl.words import iter_words

# --------------------------------------------------------------------------
# Условия РК2 2025
# --------------------------------------------------------------------------

VARIANT_1 = """
S  -> S'     ; S'.a > S'.b
S' -> T S'   ; S'0.a := T.a + S'1.a, S'0.b := max(T.b, S'1.b)
S' -> T      ; S'.a := T.a, S'.b := T.b
T  -> T a B a ; T0.a := T1.a + 1, T0.b := T1.b + B.b
T  -> ε      ; T.a := 0, T.b := 0
B  -> b B    ; B0.b := B1.b + 1
B  -> ε      ; B.b := 0
"""

VARIANT_7 = """
S -> Q S Q ;
S -> b b   ;
Q -> Q Q   ; Q1.attr <= Q2.attr, Q0.attr := Q1.attr
Q -> a A a ; Q.attr := A.attr + 2
A -> B B   ; A.attr := B1.attr + B2.attr
A -> A A   ; A0.attr := A1.attr + A2.attr
B -> b     ; B.attr := 1
"""

VARIANT_24 = """
S -> S S   ; S2.attr < S1.attr, S0.attr := S1.attr − S2.attr
S -> b A   ; S.attr := A.attr
A -> b A b ; A0.attr := A1.attr + 2
A -> a A b ; A0.attr := A1.attr + 1
A -> ε     ; A.attr := 0
"""

VARIANT_28 = """
S -> S S S ; S3.attr == S1.attr ∨ S2.attr == S1.attr, S0.attr := min(S1.attr, S2.attr, S3.attr)
S -> A     ; S.attr := A.attr
A -> a A   ; A0.attr := A1.attr * 2
A -> b b   ; A.attr := 1
A -> b     ; A.attr := 0
"""

COUNTER = """
S -> a S b ; S0.n := S1.n + 1
S -> ε     ; S.n := 0
"""


# --------------------------------------------------------------------------
# Разбор записи
# --------------------------------------------------------------------------


def test_parses_indexed_occurrences():
    grammar = parse_attr_grammar(VARIANT_24)
    rule = grammar.rules_for("S")[0]
    assert rule.rhs == ("S", "S")
    (target, expr), = rule.assignments
    assert target == Ref("S", 0, "attr")
    assert str(expr) == "(S1.attr - S2.attr)"
    assert str(rule.conditions[0]) == "S2.attr < S1.attr"


def test_bare_name_means_left_part():
    """`S.attr := A.attr` в правиле `S → bA`: слева `S`, справа `A`."""
    grammar = parse_attr_grammar(VARIANT_24)
    rule = next(r for r in grammar.rules if r.rhs == ("b", "A"))
    (target, expr), = rule.assignments
    assert rule.slot_of(target) == -1  # левая часть
    assert rule.slot_of(expr) == 1  # A стоит второй в правой части


def test_ambiguous_bare_name_is_an_error():
    """Если символ встречается справа дважды, индекс обязателен."""
    with pytest.raises(ValueError, match="нужен явный индекс"):
        parse_attr_grammar("S -> b A A ; A.attr := 1")


def test_bare_name_prefers_the_left_part():
    """`A → bAA ; A.attr := 1` — это левая часть, а не «непонятно какая»."""
    grammar = parse_attr_grammar("A -> b A A ; A.attr := 1")
    (target, _), = grammar.rules[0].assignments
    assert grammar.rules[0].slot_of(target) == -1


def test_index_zero_must_match_left_part():
    with pytest.raises(ValueError, match="индекс 0"):
        parse_attr_grammar("S -> a A ; A0.attr := 1")


def test_primed_nonterminals_survive():
    grammar = parse_attr_grammar(VARIANT_1)
    assert "S'" in grammar.nonterminals
    assert grammar.terminals == frozenset("ab")


def test_subscripts_are_normalised():
    grammar = parse_attr_grammar("A -> b A b ; A₀.attr := A₁.attr + 2")
    (target, _), = grammar.rules[0].assignments
    assert target == Ref("A", 0, "attr")


def test_alternatives_with_semantics_are_rejected():
    """К какой альтернативе относится семантика — угадывать нельзя."""
    with pytest.raises(ValueError, match="альтернативы"):
        parse_attr_grammar("A -> a A | b ; A.attr := 1")


def test_min_keeps_its_commas():
    grammar = parse_attr_grammar(VARIANT_28)
    (_, expr), = grammar.rules[0].assignments
    assert str(expr) == "min(S1.attr, S2.attr, S3.attr)"


def test_disjunction_of_comparisons():
    grammar = parse_attr_grammar(VARIANT_28)
    condition, = grammar.rules[0].conditions
    assert str(condition) == "(S3.attr == S1.attr ∨ S2.attr == S1.attr)"


# --------------------------------------------------------------------------
# Вычисление и разбор
# --------------------------------------------------------------------------


def test_counter_grammar():
    grammar = parse_attr_grammar(COUNTER)
    assert grammar.accepts("aaabbb").value is True
    assert grammar.accepts("aabbb").value is False
    trees, truncated = grammar.derivations("aabb")
    assert not truncated and len(trees) == 1
    assert grammar.evaluate(trees[0]).of(trees[0], "n") == 2


def test_condition_actually_filters():
    """Без условия язык шире — это и есть смысл задачи `RK2-C`."""
    grammar = parse_attr_grammar(VARIANT_24)
    plain = grammar.cfg()
    assert len(plain) == len(grammar)
    # `bbb` выводится и с условием, а `bbbbab` — только без него
    assert grammar.accepts("bbb").value is True
    assert grammar.accepts("bab" + "b").value is True


def test_underlying_cfg_is_a_superset():
    from tfl.lang import from_cfg

    grammar = parse_attr_grammar(VARIANT_24)
    plain = from_cfg(grammar.cfg())
    for word in grammar.words(5):
        assert word in plain, word


def test_empty_language_when_conditions_never_hold():
    grammar = parse_attr_grammar("S -> a ; S.n := 1\nS -> b ; S.n := 2")
    assert grammar.accepts("a").value is True
    grammar = parse_attr_grammar("S -> a ; S.n := 1, S.n == 2")
    assert grammar.accepts("a").value is False


# --------------------------------------------------------------------------
# Варианты РК2 2025 — контрольные примеры
# --------------------------------------------------------------------------


def blocks_of(word: str) -> list[int] | None:
    """Разобрать слово как `(a b^k a)*`; вернуть список k или None."""
    counts: list[int] = []
    i = 0
    while i < len(word):
        if word[i] != "a":
            return None
        j = i + 1
        while j < len(word) and word[j] == "b":
            j += 1
        if j >= len(word) or word[j] != "a":
            return None
        counts.append(j - i - 1)
        i = j + 1
    return counts


def test_variant_1_matches_the_reference_language():
    """Язык `(a b^{nᵢ} a)^m` при `m > max nᵢ`.

    `S'.a` — число блоков, `S'.b` — максимум по разбиению на группы `T`,
    а минимум этого максимума достигается разбиением по одному блоку.
    """
    grammar = parse_attr_grammar(VARIANT_1)

    def claim(word: str) -> bool:
        counts = blocks_of(word)
        if not counts:
            return False
        return len(counts) > max(counts)

    assert grammar.disagreements(claim, max_len=7) == []


def test_variant_1_needs_the_fixpoint_engine():
    """`S' → T S'` при `T ⇒ ε` — цикл на том же отрезке."""
    grammar = parse_attr_grammar(VARIANT_1)
    assert grammar.has_span_cycle()
    assert grammar.synthesized_only()
    trees, truncated = grammar.derivations("aa")
    assert truncated  # деревьев бесконечно много
    assert grammar.accepts("aa").value is True  # но ответ всё равно есть


def test_variant_1_attributes_count_letters():
    grammar = parse_attr_grammar(VARIANT_1)
    assert grammar.check_attribute("T", "b", lambda w: w.count("b"), 6).value is True
    assert grammar.check_attribute("T", "a", lambda w: w.count("a") // 2, 6).value is True


def test_variant_24_attribute_counts_b():
    """`A → bAb | aAb | ε` даёт `u b^{|u|}`, и `A.attr` — это `|w|_b`."""
    grammar = parse_attr_grammar(VARIANT_24)
    verdict = grammar.check_attribute("A", "attr", lambda w: w.count("b"), 8)
    assert verdict.value is True


def test_variant_24_language_starts_with_b():
    grammar = parse_attr_grammar(VARIANT_24)
    words = grammar.words(6)
    assert words
    assert all(word.startswith("b") for word in words)
    assert "b" in words and "bab" in words and "bbb" in words
    assert "ab" not in words


def test_variant_7_language_is_thin():
    """`Q → aAa` требует хотя бы `bb` внутри, поэтому слов мало."""
    grammar = parse_attr_grammar(VARIANT_7)
    assert grammar.words(7) == ["bb"]
    assert grammar.accepts("abbabbabba").value is True  # Q S Q


def test_variant_7_inner_blocks_have_even_b():
    """`A → BB | AA`, `B → b`: из A выводится только `b` в чётном числе."""
    inner = AttrGrammar("A", parse_attr_grammar(VARIANT_7).rules)
    assert inner.words(6) == ["bb", "bbbb", "bbbbbb"]


def test_variant_7_condition_is_not_vacuous():
    """`Q1.attr ≤ Q2.attr` упорядочивает блоки в правиле `Q → QQ`.

    `Q` порождает `a b^{2m} a` с атрибутом `2m + 2`, поэтому условие
    запрещает ставить длинный блок перед коротким.
    """
    rules = parse_attr_grammar(VARIANT_7).rules
    strict = AttrGrammar("Q", rules)
    loose = AttrGrammar(
        "Q", parse_attr_grammar(VARIANT_7.replace("Q1.attr <= Q2.attr, ", "")).rules
    )
    assert strict.accepts("abbaabbbba").value is True
    assert strict.accepts("abbbbaabba").value is False
    assert loose.accepts("abbbbaabba").value is True


def test_variant_28_ignoring_conditions_changes_the_answer():
    """Если отбросить условие, получится другой язык."""
    grammar = parse_attr_grammar(VARIANT_28)
    naive = parse_attr_grammar(
        VARIANT_28.replace(
            "S3.attr == S1.attr ∨ S2.attr == S1.attr, ", ""
        )
    )
    differing = [
        word
        for word in iter_words("ab", 6)
        if (grammar.accepts(word).value) != (naive.accepts(word).value)
    ]
    assert differing == ["bbabab"]


def test_variant_28_attribute_doubles():
    """`A → aA` удваивает, `A → bb` даёт 1, `A → b` даёт 0."""
    grammar = parse_attr_grammar(VARIANT_28)
    assert grammar.check_attribute(
        "A", "attr", lambda w: 2 ** w.count("a") if w.endswith("bb") else 0, 6
    ).value is True


# --------------------------------------------------------------------------
# Наследуемые атрибуты
# --------------------------------------------------------------------------

INHERITED = """
S -> a S' a ; S'.d := 1
S -> b S' b ; S'.d := 1
S' -> a S' a ; S'1.d := S'0.d + 1
S' -> b S' b ; S'1.d := S'0.d + 1
S' -> T      ; T.d := S'.d
T -> a T     ; T1.d := T0.d - 1
T -> ε       ; T.d == 0
"""


def test_inherited_attribute_flows_down():
    """Условие стоит на листе, а значение приходит сверху.

    Наследуемый счётчик считает глубину обёрток и требует ровно столько же
    букв `a` в середине: язык — это `{w a^{|w|} wᴿ}`.
    """
    grammar = parse_attr_grammar(INHERITED)
    assert not grammar.synthesized_only()
    assert grammar.words(6) == ["aaa", "bab", "aaaaaa", "abaaba", "baaaab", "bbaabb"]

    def claim(word: str) -> bool:
        if len(word) % 3:
            return False
        k = len(word) // 3
        head, middle, tail = word[:k], word[k : 2 * k], word[2 * k :]
        return bool(k) and middle == "a" * k and tail == head[::-1]

    assert grammar.disagreements(claim, max_len=6) == []


def test_check_attribute_refuses_inherited():
    grammar = parse_attr_grammar(INHERITED)
    verdict = grammar.check_attribute("T", "d", lambda w: len(w), 4)
    assert verdict.value is None
    assert "наследуемый" in verdict.reason
