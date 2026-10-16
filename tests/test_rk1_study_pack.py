"""Учебный комплект: целостность файлов и независимые предметные регрессии.

Конечные проверки не заменяют параметрические доказательства из KEYS.md.
Реальный диалог модели и отображение спойлеров здесь не проверяются.
"""

from collections import defaultdict, deque
from itertools import product
from pathlib import Path
import re

import pytest

from tfl.automata import DFA, counterexample, dfa_of
from tfl.code import Code
from tfl.glushkov import glushkov
from tfl.regex import parse
from tfl.srs import parse_srs
from tfl.words import iter_words


PACK = Path(__file__).resolve().parents[1] / "study/rk1-2026"


def _read(name):
    return (PACK / name).read_text(encoding="utf-8")


def _sections(name):
    text = _read(name)
    starts = list(re.finditer(r"^## ([A-H]\d{2})(?:\.|$)", text, re.M))
    return {
        match[1]: text[match.end(): starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        for i, match in enumerate(starts)
    }


def _table(section):
    rows = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip().replace("→", "").replace("★", "")
             for cell in row.strip("|").split("|")]
            for row in rows[2:]]


def _dfa_from_key(task, finals, columns=(1, 2)):
    rows = _table(_sections("KEYS.md")[task])
    transitions = {(row[0], letter): row[col]
                   for row in rows for letter, col in zip("ab", columns)}
    return DFA(frozenset("ab"), "S", frozenset(finals), transitions)


def _parses(nfa, word):
    counts = {nfa.start: 1}
    for char in word:
        following = defaultdict(int)
        for source, count in counts.items():
            for target in nfa.delta.get((source, char), ()):
                following[target] += count
        counts = following
    return sum(counts.get(state, 0) for state in nfa.finals)


def test_pack_has_matching_unique_conditions_and_keys():
    expected = {f"{block}{i:02}" for block, count in
                zip("ABCDEFGH", (4, 8, 6, 18, 8, 8, 1, 2))
                for i in range(1, count + 1)}
    for name in ("TASKS.md", "KEYS.md"):
        occurrences = re.findall(r"^## ([A-H]\d{2})(?:\.|$)", _read(name), re.M)
        assert len(occurrences) == len(set(occurrences))
        assert set(occurrences) == expected
    for section in _sections("TASKS.md").values():
        assert "требуется" in section.lower()
        assert "Семинар:" in section and "Маршрут:" in section
    assert "Решение:" not in _read("TASKS.md")


def test_upload_instructions_and_limits_are_explicit():
    for name in ("TASKS.md", "KEYS.md", "METHODS.md", "PROGRESS.md"):
        assert (PACK / name).is_file()
        assert name in _read("START-PROMPT.md")
    prompt = " ".join(_read("START-PROMPT.md").lower().split())
    assert "по одной задаче" in prompt and "после выдачи условия остановись" in prompt
    assert "кс-накачку" in prompt
    assert "нет автоматически подключённого плагина" in prompt
    assert "не обещай межчатовую память" in prompt
    for name in ("TASKS.md", "KEYS.md", "METHODS.md", "START-PROMPT.md"):
        text = _read(name)
        assert not re.search(r"(?:^|\s)@[A-Za-z0-9_]+", text)
        assert not re.search(r"\b[A-Z]:[\\/]", text)
        assert "Reacted by" not in text and "opened last" not in text


def test_incomplete_sources_are_not_silently_promoted_to_core_keys():
    readme = _read("README.md")
    assert "две задачи не выдаются" in readme
    assert "10 октября" in readme and "10-09.md" in readme
    keys = _sections("KEYS.md")
    for task in ("H01", "H02"):
        assert "Частичный ключ" in keys[task]
    assert "зависит от определения" in keys["F03"]
    assert "КС-накачку" in keys["G01"]


def test_critical_pairs_match_the_documented_table():
    system = parse_srs("ababa -> bb\nbabb -> aa")
    actual = {(w, frozenset((a, b))) for w, a, b, *_ in system.critical_pairs()}
    documented = {(row[0], frozenset(row[1:]))
                  for row in _table(_sections("KEYS.md")["B01"])}
    assert actual == documented


def test_copresentation_action_table_is_a_separating_model():
    rows = _table(_sections("KEYS.md")["C01"])
    delta = {(row[0], c): row[i] for row in rows for c, i in (("a", 1), ("b", 2))}

    def run(q, word):
        for c in word:
            q = delta[q, c]
        return q

    for row in rows:
        for lhs, rhs in (("ab", "baa"), ("aaa", "bb"), ("aba", "b")):
            assert run(row[0], lhs) == run(row[0], rhs)
    representatives = ("a", "b", "aa", "ab", "ba", "bb", "abb", "bab")
    assert len({run("o", w) for w in representatives}) == 8
    assert not any(delta[q, "a"] == "a" and delta[q, "b"] == "b" for q in representatives)


def _occ(word, pattern):
    return sum(word.startswith(pattern, i) for i in range(len(word)))


def test_fixed_rank_termination_measure_on_bounded_contexts():
    system = parse_srs("abba -> bababaa\naaa -> bbb")

    def measure(word):
        return tuple(_occ(word, "ab" * k + "abba") for k in range(7, -1, -1)) + (_occ(word, "aaa"),)

    for word in iter_words("ab", 11):
        for result in system.step(word):
            assert measure(result) < measure(word)


def _guarded_rules(task):
    body = re.search(r"```text\n(.*?)```", _sections("KEYS.md")[task], re.S)[1]
    rules = re.findall(r"([A-Za-z^$]+)\s*→\s*([A-Za-z^$]+)", body)
    return parse_srs("\n".join(f"{a} -> {b}" for a, b in rules))


def _closure(start, step):
    found = {start}
    pending = deque([start])
    while pending:
        for word in step(pending.popleft()):
            if word not in found:
                found.add(word)
                pending.append(word)
        assert len(found) < 100_000, "bounded regression unexpectedly exceeded its guard"
    return found


def _pattern_step(word):
    results = set()
    for j, c in enumerate(word):
        if c == "b":
            for i in range(j + 1):
                results.add(word[:i] + "aa" + word[i:j] + word[j + 1:])
                if word[i] == "a":
                    results.add(word[:i] + "b" + word[i + 1:j] + "a" + word[j + 1:])
    return results


@pytest.mark.parametrize("task", ["B06", "B07"])
def test_guarded_simulations_preserve_exact_clean_reachability(task):
    system = _guarded_rules(task)

    def deletion_step(word):
        return {word[:i] + word[j + 1:] for j, c in enumerate(word) if c == "b" for i in range(j + 1)}

    for word in iter_words("ab", 4):
        expected = _closure(word, _pattern_step if task == "B06" else deletion_step)
        reached = _closure("^" + word + "$", system.step)
        clean = {w[1:-1] for w in reached if re.fullmatch(r"\^[ab]*\$", w)}
        assert clean == expected
    if task == "B06":
        reached = _closure("^a$", system.step)
        assert {"LBM$", "LAAAN$"} <= reached
        assert all(not system.step(w) for w in ("LBM$", "LAAAN$"))


def test_morphism_is_uniquely_decodable_and_delayed():
    assert Code(("0", "01", "11")).uniquely_decodable().value is True
    for k in range(1, 30):
        a, b = "0" + "1" * (2 * k), "01" + "11" * (k - 1)
        assert len(b) == 2 * k and a.startswith(b)


def test_e06_complement_against_the_literal_condition():
    machine = _dfa_from_key("E06", {"S", "D"})
    for word in iter_words("ab", 12):
        assert machine.accepts(word) == (not word or "aaa" in word)


def test_e07_complement_against_all_pairs_of_occurrences():
    machine = _dfa_from_key("E07", {"S", "D"})
    for word in iter_words("ab", 12):
        occurrences = [i for i in range(len(word) - 1) if word[i:i + 2] == "aa"]
        original = bool(word) and all(
            _occ(word[i + 2:j], "bb") == 1
            for i, j in product(occurrences, repeat=2) if i < j
        )
        assert machine.accepts(word) != original


def test_e08_complement_checks_existence_of_every_possible_separator():
    machine = _dfa_from_key("E08", {"S", "V", "Z"}, (2, 3))
    for word in iter_words("ab", 12):
        original = any(c == "a" and "bb" not in word[:i] and "aa" not in word[i + 1:]
                       for i, c in enumerate(word))
        assert machine.accepts(word) != original


def test_f08_expression_is_exactly_equivalent_and_has_single_bounded_parses():
    formula = re.search(r"```text\n(.*?)\n```", _sections("KEYS.md")["F08"], re.S)[1]
    expression = formula.replace("A", "((a|ab)*)").replace("B", "((bb|ba)*)")
    assert counterexample(dfa_of(expression), dfa_of("(a|ab)*(bb|ba)*")) is None
    machine = glushkov(parse(expression))
    for word in iter_words("ab", 12):
        assert _parses(machine, word) <= 1


@pytest.mark.parametrize("expression,witness", [
    ("(a|ab)*b(b|ba)*", "abb"),
    ("(a|ab)*(bb|ba)*", "aba"),
    ("((a|aa)b|abaab)*", "abaab"),
    ("(ab*aa|b*abb*)*", "abbab"),
])
def test_ambiguity_witnesses(expression, witness):
    assert _parses(glushkov(parse(expression)), witness) >= 2


@pytest.mark.parametrize("expression", ["(aab|b*abb)*aa*", "(aabb*a|bb*abb)*"])
def test_prefix_code_based_unique_expressions_on_bounded_words(expression):
    machine = glushkov(parse(expression))
    assert all(_parses(machine, word) <= 1 for word in iter_words("ab", 12))


def test_floating_square_finite_complements():
    excluded = {"", "a", "b", "ab", "ba", "aba", "bab"}
    excluded_different = {"", "a", "b", "aa", "ab", "ba", "bb", "aba", "bab",
                          "abab", "baba", "abba", "baab"}
    for word in iter_words("ab", 12):
        positions = [(i, width) for width in range(1, len(word) // 2 + 1)
                     for i in range(len(word) - 2 * width + 1)
                     if word[i:i + width] == word[i + width:i + 2 * width]]
        assert bool(positions) == (word not in excluded)
        assert any(word[:i] != word[i + 2 * width:] for i, width in positions) == (word not in excluded_different)
