"""Короткий учебный комплект: соответствие 12 типам и проверки ключей.

Конечные контекстные проверки не заменяют общих доказательств из KEYS.md.
Таймер, реальный диалог и учебный эффект модели этими тестами не проверяются.
"""

from collections import defaultdict
from pathlib import Path
import re

import pytest

from tfl.automata import DFA, NFA, counterexample, dfa_of, intersection
from tfl.glushkov import glushkov
from tfl.lookaround import parse_extended, to_regex
from tfl.regex import parse
from tfl.srs import parse_srs
from tfl.words import iter_words


PACK = Path(__file__).resolve().parents[1] / "study/rk1-2026/final-sprint"


def read(name):
    return (PACK / name).read_text(encoding="utf-8")


def sections(name):
    text = read(name)
    starts = list(re.finditer(r"^## (T\d{2}-[A-Z])(?:\.|$)", text, re.M))
    return {m[1]: text[m.end():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
            for i, m in enumerate(starts)}


def tables(section):
    blocks = re.findall(r"(?:^\|.*\n)+", section, re.M)
    return [[[cell.strip() for cell in row.strip("|").split("|")]
             for row in block.strip().splitlines()[2:]] for block in blocks]


def dfa_table(task, columns=(1, 2)):
    rows = tables(sections("KEYS.md")[task])[0]
    state = lambda row: row[0].replace("→", "").replace("★", "")
    start = next(state(row) for row in rows if "→" in row[0])
    finals = frozenset(state(row) for row in rows if "★" in row[0])
    return DFA(frozenset("ab"), start, finals,
               {(state(row), c): row[col] for row in rows
                for c, col in zip("ab", columns)})


def extended_dfa(pattern):
    return dfa_of(to_regex(parse_extended(pattern, anchored=False), "ab"), "ab")


def test_all_twelve_types_have_keys_and_core_tasks():
    tasks, keys = sections("TASKS.md"), sections("KEYS.md")
    expected = {f"T{i:02}-{letter}" for i, count in enumerate(
        (2, 3, 3, 1, 1, 2, 1, 1, 3, 3, 4, 2), 1)
        for letter in "ABCD"[:count]}
    assert tasks.keys() == keys.keys() == expected
    for filename in ("TASKS.md", "KEYS.md"):
        assert len(re.findall(r"^## T\d{2}-[A-Z](?:\.|$)", read(filename), re.M)) == 26
    assert sum("Очередь: основная" in s for s in tasks.values()) == 20
    assert sum("Очередь: резерв" in s for s in tasks.values()) == 6
    for i in range(1, 13):
        assert any(f"Тип: {i}. Очередь: основная" in s for s in tasks.values())
    for section in tasks.values():
        assert "Требуется" in section and "Источник:" in section
    for key in keys.values():
        assert "Подсказка:" in key and "Решение:" in key
    progress = set(re.findall(r"^\| (T\d{2}-[A-Z]) \|", read("PROGRESS.md"), re.M))
    assert progress == expected


def test_prompt_keeps_written_timed_and_private_progress_contracts():
    prompt = read("START-PROMPT.md")
    for filename in ("TASKS.md", "KEYS.md", "METHODS.md", "PROGRESS.md"):
        assert filename in prompt
    for phrase in ("После выдачи условия остановись", "по одной задаче",
                   "не назначай точный официальный штраф", "не обещай межчатовую память",
                   "не оценивай автоматически", "не измерено"):
        assert phrase.lower() in prompt.lower()
    for path in PACK.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b[A-Z]:[\\/]", text)
        assert "38,5%" not in text and "D01 по E08" not in text
    assert "a→aa" in read("METHODS.md")
    assert "неполной системы" in read("METHODS.md")


@pytest.mark.parametrize("task,rules", [
    ("T01-A", "ababa -> bb\nbabb -> aa"),
    ("T01-B", "aab -> ba\naaa -> ab"),
])
def test_documented_critical_pairs_are_exact(task, rules):
    actual = {(w, frozenset((a, b))) for w, a, b, *_ in parse_srs(rules).critical_pairs()}
    documented = {(row[0], frozenset(row[1:])) for row in tables(sections("KEYS.md")[task])[0]}
    assert actual == documented


def test_counting_measures_with_all_short_contexts_and_explicit_cycle():
    occurrence = lambda word, p: sum(word.startswith(p, i) for i in range(len(word)))
    contexts = list(iter_words("ab", 4))
    for left in contexts:
        for right in contexts:
            measure = lambda w: (w.count("a"), occurrence(w, "aa"))
            for lhs, rhs in (("aabb", "abab"), ("aba", "bab")):
                assert measure(left + lhs + right) > measure(left + rhs + right)
            assert occurrence(left + "babba" + right, "bb") > occurrence(left + "abababa" + right, "bb")
    system = parse_srs("ab -> ba\nba -> ab")
    assert "ba" in system.step("ab") and "ab" in system.step("ba")


def test_finite_monoid_completion_and_all_documented_joined_peaks():
    section = sections("KEYS.md")["T03-B"]
    body = re.search(r"```text\n(.*?)```", section, re.S)[1]
    system = parse_srs(body.replace(", ", "\n").replace(".", ""))
    nf = lambda word: next(iter(system.normal_forms(word, max_len=len(word)).words))
    documented = {w: "" if normal == "ε" else normal for w, normal in tables(section)[0]}
    criticals = system.critical_pairs()
    assert set(documented) == {w for w, *_ in criticals}
    for word, a, b, *_ in criticals:
        assert nf(a) == nf(b) == documented[word]
    initial = parse_srs("aaa -> eps\nbb -> eps\nbaa -> ab")
    completed, verdict = initial.complete("ab")
    assert verdict.value is True and set(completed.rules) == set(system.rules)
    rows = tables(section)[1]
    representatives = {"" if row[0] == "ε" else row[0] for row in rows}
    assert representatives == {"", "a", "b", "aa", "ab", "ba"}
    for row in rows:
        source = "" if row[0] == "ε" else row[0]
        for char, target in zip("ab", row[1:]):
            assert nf(source + char) == ("" if target == "ε" else target)


def test_collapsed_and_infinite_monoid_normal_forms():
    finite = parse_srs("b -> a\nc -> a\naa -> a")
    for lhs, rhs in (("bb", "c"), ("cc", "a"), ("aca", "b"),
                     ("aaa", "a"), ("ba", "b"), ("bc", "ab")):
        assert finite.normal_forms(lhs, max_len=4).words == finite.normal_forms(rhs, max_len=4).words == {"a"}
    infinite = parse_srs("ba -> ab")
    assert not infinite.critical_pairs()
    for word in iter_words("ab", 8):
        expected = "a" * word.count("a") + "b" * word.count("b")
        assert infinite.normal_forms(word, max_len=len(word)).words == {expected}


@pytest.mark.parametrize("task", ["T06-A", "T07-A"])
def test_small_documented_dfas(task):
    machine = dfa_table(task)
    if task == "T06-A":
        source = DFA(frozenset("ab"), "1", frozenset(["3"]),
                     {("1", "a"): "2", ("1", "b"): "3", ("2", "a"): "1",
                      ("2", "b"): "3", ("3", "a"): "3", ("3", "b"): "2"})
        assert counterexample(machine, source) is None
    else:
        assert counterexample(machine, dfa_of("a|(a|b)*b", "ab")) is None


def test_positional_table_and_expression_by_nfa():
    rows = tables(sections("KEYS.md")["T05-A"])[0]
    delta = {}
    for row in rows:
        source = row[0].replace("→", "").replace("★", "")
        for char, target in zip("ab", row[1:]):
            delta[source, char] = (frozenset() if target == "∅" else
                                   frozenset(target.strip("{}").split(",")))
    positional = NFA(frozenset("ab"), "S", frozenset(["S", "5"]), delta)
    assert counterexample(positional.determinize(), dfa_of("(a*(a|ba)*b)*", "ab")) is None
    source = NFA(frozenset("ab"), "1", frozenset(["2"]), {
        ("1", "a"): frozenset(["2"]), ("1", "b"): frozenset(["3"]),
        ("3", "b"): frozenset(["2"]), ("2", "a"): frozenset(["1", "2"])})
    pattern = "(a|bb)(a|aa|abb)*"
    assert pattern in sections("KEYS.md")["T04-A"]
    assert counterexample(source.determinize(), dfa_of(pattern, "ab")) is None


def test_documented_intersection_table():
    first = DFA(frozenset("ab"), "0", frozenset(["1"]), {
        ("0", "a"): "1", ("0", "b"): "2", ("1", "a"): "0",
        ("1", "b"): "1", ("2", "a"): "2", ("2", "b"): "1"})
    second = DFA(frozenset("ab"), "A", frozenset(["A"]), {
        ("A", "a"): "B", ("A", "b"): "A", ("B", "a"): "A", ("B", "b"): "B"})
    assert counterexample(dfa_table("T08-A"), intersection(first, second)) is None


def test_complement_tables_use_the_literal_conditions():
    split = dfa_table("T09-B", (2, 3))
    all_pairs = dfa_table("T09-C")
    for word in iter_words("ab", 10):
        has_split = any(c == "a" and "bb" not in word[:i] and "aa" not in word[i + 1:]
                        for i, c in enumerate(word))
        assert split.accepts(word) == (not has_split)
        starts = [i for i in range(len(word)) if word.startswith("aa", i)]
        original = bool(word) and all(
            j >= i + 2 and sum(word.startswith("bb", k) for k in range(i + 2, j - 1)) == 1
            for ix, i in enumerate(starts) for j in starts[ix + 1:])
        assert all_pairs.accepts(word) == (not original)


def test_equivalence_tasks_and_minimal_counterexample():
    tasks = sections("TASKS.md")
    for task, expected in (("T12-A", None), ("T12-B", "a")):
        patterns = re.findall(r"`R[₁₂]=(.*?)`", tasks[task])
        assert len(patterns) == 2
        assert counterexample(*map(extended_dfa, patterns)) == expected


def test_regular_counter_language_formula_and_nonregular_witnesses():
    A, B = "aa*", "bbb*"
    pattern = f"b*|(ε|b){A}({B}{A})*|{B}{A}({B}{A})*b{A}({B}{A})*|{B}{A}({B}{A})*bb*"
    machine = dfa_of(pattern, "ab")
    occurrence = lambda word, p: sum(word.startswith(p, i) for i in range(len(word)))
    for word in iter_words("ab", 11):
        assert machine.accepts(word) == (occurrence(word, "ab") == occurrence(word, "bba"))
    for i in range(2, 12):
        for j in range(2, 12):
            n, m = 2 * i, 2 * j
            prime = n >= 2 and all(n % k for k in range(2, n))
            assert (n != m or prime) == (i != j)


def test_ambiguity_witnesses_and_unique_code_on_bounded_words():
    def paths(machine, word):
        counts = {machine.start: 1}
        for char in word:
            following = defaultdict(int)
            for q, count in counts.items():
                for target in machine.delta.get((q, char), ()):
                    following[target] += count
            counts = following
        return sum(counts.get(q, 0) for q in machine.finals)

    for pattern, word in (("(a|ab)*b(b|ba)*", "abb"),
                          ("((a|aa)b|abaab)*", "abaab"), ("(a|ab)*(b|ba)*", "ab")):
        assert paths(glushkov(parse(pattern)), word) >= 2
    unique = glushkov(parse("(aab|b*abb)*aa*"))
    assert all(paths(unique, w) <= 1 for w in iter_words("ab", 10))
