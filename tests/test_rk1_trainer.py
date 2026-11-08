"""Audit the actual teaching cards; finite samples are regressions, not proofs."""

from collections import deque
from math import gcd
import re
import zipfile

import pytest

from tfl.automata import DFA, NFA, counterexample, dfa_of
from tfl.lookaround import parse_extended, to_regex
from tfl.srs import parse_srs
from tfl.words import iter_words
from tools.build_rk1_trainer import SOURCE, build, documents, sections


def read(name):
    return (SOURCE / name).read_text(encoding="utf-8")


TASKS = sections(read("EXTRA-TASKS.md"))
KEYS = sections(read("EXTRA-KEYS.md"))


def count(word, part):
    return sum(word.startswith(part, i) for i in range(len(word)))


def table(body):
    block = re.search(r"(?:^\|.*\n)+", body, re.M)[0]
    return [[c.strip() for c in line.strip("|").split("|")]
            for line in block.strip().splitlines()[2:]]


def documented_dfa(body, start=None, finals=None):
    rows = table(body)
    state = lambda s: s.replace("→", "").replace("★", "").split("=")[0]
    if start is None:
        start = next(state(row[0]) for row in rows if "→" in row[0])
    if finals is None:
        finals = {state(row[0]) for row in rows if "★" in row[0]}
    return DFA(frozenset("ab"), start, frozenset(finals),
               {(state(row[0]), char): row[column]
                for row in rows for column, char in enumerate("ab", 1)})


def answer(card_id):
    return re.findall(r"`([^`]+)`", KEYS[card_id])[-1].replace(" ", "")


def test_bundle_is_complete_self_contained_and_protects_progress(tmp_path):
    payload = documents()
    tasks, keys = sections(payload["TASKS.md"]), sections(payload["KEYS.md"])
    assert len(tasks) == 62 and tasks.keys() == keys.keys()
    assert set(TASKS) == {f"P{i:02}" for i in range(1, 16)} | {"R01", "R02"} | {
        f"G{i:02}" for i in range(1, 20)}
    assert sum("Тип: 10." in body for body in tasks.values()) == 25
    for kind in range(1, 13):
        assert sum(f"Тип: {kind}." in body for body in tasks.values()) >= 2
    progress = set(re.findall(r"^\| ((?:T\d{2}-[A-Z]|[PRG]\d{2})) \|",
                              payload["PROGRESS.md"], re.M))
    assert progress == tasks.keys()
    for name in ("TASKS.md", "KEYS.md", "METHODS.md", "GENERATION.md", "PROGRESS.md"):
        assert name in payload["START-PROMPT.md"]
    for body in payload.values():
        assert not re.search(r"\b[A-Z]:[\\/]", body)
    target = build(tmp_path / "bundle", archive=True)
    with zipfile.ZipFile(target.with_suffix(".zip")) as archive:
        assert set(archive.namelist()) == payload.keys()
        for name, body in payload.items():
            assert archive.read(name).decode("utf-8") == body + "\n"
    (target / "PROGRESS.md").write_text("личный прогресс", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build(target, archive=True)
    assert (target / "PROGRESS.md").read_text(encoding="utf-8") == "личный прогресс"
    collision = tmp_path / "archive-only"
    collision.with_suffix(".zip").touch()
    with pytest.raises(FileExistsError):
        build(collision, archive=True)
    assert not collision.exists()


def test_interaction_and_generation_contracts():
    prompt, generation = read("START-PROMPT.md"), read("GENERATION.md")
    for phrase in ("После выдачи условия остановись", "не измерена", "не обнуляй",
                   "не назначай точный официальный штраф", "не обещай межчатовую память",
                   "разобрана", "самостоятельно", "всех возможных свидетелей"):
        assert phrase.lower() in " ".join(prompt.lower().split())
    for phrase in ("общего доказательства нет", "не являются официальными",
                   "NEW-KEYS.md", "конечное совпадение", "N-ID"):
        assert phrase.lower() in generation.lower()
    assert "Σ*" not in TASKS["R01"].split("Требуется")[0].replace("∈Σ*", "")
    assert "x,y∈N₀" in TASKS["R02"]


@pytest.mark.parametrize("card_id", ["P01", "P03"])
def test_gcd_witnesses_check_every_hidden_split(card_id):
    assert "4^i" in KEYS[card_id] and "3t_i" in KEYS[card_id]
    def member(p, q):
        if card_id == "P01":
            return any(gcd(p - k, k) < gcd(q - k, k) for k in range(min(p, q) + 1))
        return any(gcd(p, m) < gcd(m, q - m) for m in range(q + 1))
    for i in range(4):
        t = 4**i
        assert member(3 * t, 4 * t)
        for j in range(i + 1, 5):
            assert not member(3 * 4**j, 4 * t)


def palindrome_prefix(word):
    return any(word[:2 * k] == word[:k] + word[:k][::-1]
               and ("aa" in word[2*k:] or "bb" in word[2*k:])
               for k in range(1, len(word)//2 + 1))


def context_palindrome(word):
    return any(word[i:i + 2*k] == word[i:i+k] + word[i:i+k][::-1]
               and i < len(word) - i - 2*k
               for i in range(len(word)) for k in range(2, (len(word)-i)//2 + 1))


def test_palindrome_families_and_context_shrinking():
    assert "(ab)^i" in KEYS["P02"] and "abba(ab)^(i+1)" in KEYS["P04"]
    for i in range(6):
        for j in range(6):
            assert palindrome_prefix("ab"*i + "aa" + "ba"*j + "bb") == (i <= j)
            assert context_palindrome("ab"*i + "abba" + "ab"*j) == (i < j)
    machine = dfa_of(answer("G10"), "ab")
    for w in iter_words("ab", 9):
        direct = any(w[i:i+2*k] == w[i:i+k] + w[i:i+k][::-1]
                     for i in range(len(w)) for k in range(2, (len(w)-i)//2+1))
        assert machine.accepts(w) == direct


def test_unary_intervals_and_difference_square_constructions():
    assert "[n²+n,n²+2n−1]" in KEYS["P05"]
    for m in range(1, 80, 2):
        assert all(not (n*n+n <= m*m <= n*n+2*n-1) for n in range(1, m+1))
    pattern = answer("R02")
    assert answer("G07") == pattern
    machine = dfa_of(pattern, "a")
    for length in range(600):
        assert machine.accepts("a"*length) == (length % 4 != 2)
        if length % 2:
            r = (length-1)//2
            assert (r+1)**2 - r*r == length
        elif length and length % 4 == 0:
            r = length//4
            assert (r+1)**2 - (r-1)**2 == length


def p06(word):
    return any((n > 0 and word[:i].count("a") > n and
                (word[:i] + word[i+1+n:]).count("b") > 0)
               or len(word[i+1+n:]) > i
               for i, c in enumerate(word) if c == "b"
               for n in range(len(word)-i) if set(word[i+1:i+1+n]) <= {"a"})


def test_single_separator_family_checks_all_exposed_blocks():
    for r in range(8):
        for s in range(8):
            assert p06("a"*r + "b" + "a"*s) == (s > r)


def test_empty_parts_and_moving_boundaries():
    for w in iter_words("ab", 8):
        p07 = any(w[:i].count("a") == w[j:].count("a") and (
            "b" in w[i:j] or re.fullmatch("b+a+", w[j:]))
            for i in range(len(w)+1) for j in range(i, len(w)+1))
        assert bool(p07) == ("b" in w)
        assert any(count(w[:i], "aa") == w[i:].count("b") for i in range(len(w)+1))
        assert any(count(w[:i], "bb") == w[i:].count("a") for i in range(len(w)+1))
    for n in range(1, 7):
        for m in range(1, 7):
            w = "baab"*n + "a" + "baab"*m
            assert any(count(w[:i], "ba") == count(w[i:], "ab")
                       for i in range(len(w)+1)) == (n != m)
    assert "|u|_ba=|v|_ab" in TASKS["G05"]


def test_complement_of_edge_condition():
    machine = dfa_of(answer("P09").replace("S", "(a|b)*"), "ab")
    for w in iter_words("ab", 11):
        assert machine.accepts(w) == (count(w, "ab") != count(w, "ba") or "aaa" in w)


def test_complement_with_one_aa_exact_equivalence():
    key = KEYS["P10"]
    assert "N=(b|ab)*(ε|a)" in key and "T=S(aaa|aaSaa)S" in key
    pattern = answer("P10").replace("N", "((b|ab)*(ε|a))").replace(
        "T", "(S(aaa|aaSaa)S)").replace("S", "(a|b)*")
    original = dfa_of("(b|ab)*aa(b|ba)*aba*ba*", "ab")
    complement = dfa_of(pattern, "ab")
    assert counterexample(complement, original.complement()) is None
    for w in iter_words("ab", 9):
        literal = any(w[i:i+2] == "ab" and count(w[:i], "aa") == 1
                      and w[i+2:].count("b") == 1 for i in range(len(w)))
        assert original.accepts(w) == literal


def modular_pair_dfa():
    # Independent streaming predicate: remember all residues at earlier b's.
    start = (0, frozenset(), False)
    queue, seen, delta = deque([start]), {start}, {}
    while queue:
        r, labels, accepted = source = queue.popleft()
        for c in "ab":
            target = (((r+1) % 3, labels, accepted) if c == "a" else
                      (r, labels | {r}, accepted or any(r == 2*p % 3 for p in labels)))
            delta[source, c] = target
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return DFA(frozenset("ab"), start, frozenset(q for q in seen if q[2]), delta)


def test_modular_complement_arbitrary_pair_exactly():
    pattern = re.search(r"```text\n(.*?)\n```", KEYS["P11"], re.S)[1]
    macros = {"B₁": "(A₁b(A₀b)*)", "B₂": "(A₂b(A₀b)*)",
              "A₀": "((aaa)*)", "A₁": "(a(aaa)*)", "A₂": "(aa(aaa)*)"}
    for symbol, expansion in macros.items():
        pattern = pattern.replace(symbol, expansion)
    original = modular_pair_dfa()
    assert counterexample(dfa_of(pattern, "ab"), original.complement()) is None
    for w in iter_words("ab", 9):
        indices = [i for i, c in enumerate(w) if c == "b"]
        literal = any(w[:i].count("a") % 3 == w[i+1:j].count("a") % 3
                      for i in indices for j in indices if i < j)
        assert original.accepts(w) == literal


def test_expression_complement_and_inequivalence():
    source = re.search(r"`R=(.*?)`", TASKS["P12"])[1]
    assert counterexample(dfa_of(answer("P12"), "ab"), dfa_of(source, "ab").complement()) is None
    patterns = re.findall(r"`R[₁₂]=(.*?)`", TASKS["P14"])
    machines = [dfa_of(to_regex(parse_extended(p, anchored=False), "ab"), "ab") for p in patterns]
    assert not machines[0].accepts("abb") and machines[1].accepts("abb")


def test_unambiguous_code_reachable_residuals_are_exhaustive():
    blocks = re.search(r"`R=\((.*?)\)\*`", TASKS["P13"])[1].split("|")
    residuals = {y[len(x):] for x in blocks for y in blocks if x != y and y.startswith(x)}
    transitions, seen = set(), set(residuals)
    queue = deque(residuals)
    while queue:
        residual = queue.popleft()
        assert residual
        for block in blocks:
            target = (residual[len(block):] if residual.startswith(block) else
                      block[len(residual):] if block.startswith(residual) else None)
            if target is not None:
                transitions.add((residual, block, target))
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
    assert transitions == {tuple(row) for row in table(KEYS["P13"])}
    assert seen == {"aba", "a", "baa", "aaba", "ba"}


def test_expression_by_documented_pdf_automaton():
    source = documented_dfa(TASKS["P15"], "q2", {"q4"})
    assert counterexample(source, dfa_of(answer("P15"), "ab")) is None


def retake_member(w, nonempty=False):
    for size in range(1 if nonempty else 0, len(w)//2+1):
        left = w[:size]
        right = w[-size:] if size else ""
        middle = w[size:len(w)-size] if size else w
        if nonempty and not middle:
            continue
        if left == right[::-1] and middle.count("a") == right.count("b"):
            return True
    return False


def test_retake_one_all_splits_and_both_nonemptiness_conventions():
    assert "a+b+a+" in KEYS["R01"]
    for i in range(1, 6):
        for j in range(1, 5):
            for k in range(1, 6):
                for nonempty in (False, True):
                    assert retake_member("a"*i + "b"*j + "a"*k, nonempty) == (i == k)
    assert not retake_member("a")  # Empty sides do not make the language universal.


def test_generated_regular_constructions():
    first, series = dfa_of(answer("G01"), "ab"), dfa_of(answer("G03"), "ab")
    for w in iter_words("ab", 10):
        assert first.accepts(w) == (count(w, "ab") == count(w, "ba") + 1)
        assert series.accepts(w) == (count(w, "bb") == count(w, "bbb"))
    for i in range(8):
        for j in range(8):
            assert (2*j == 2*i) == (i == j)
            assert (i <= 3*j or j <= 3*i)
            assert (3**i >= 3*3**j or 3**j >= 3*3**i) == (i != j)


def test_new_critical_pairs_and_context_measure():
    system = parse_srs("aa -> b\naaa -> a")
    actual = {(w, frozenset((a, b))) for w, a, b, *_ in system.critical_pairs() if a != b}
    documented = {(w, frozenset((a, b))) for w, a, b in table(KEYS["G13"])}
    assert actual == documented
    assert "(|w|_b,|w|_bb)" in KEYS["G14"]
    for left in iter_words("ab", 4):
        for right in iter_words("ab", 4):
            measure = lambda w: (w.count("b"), count(w, "bb"))
            for lhs, rhs in (("bbaa", "baba"), ("bab", "aba")):
                assert measure(left+lhs+right) > measure(left+rhs+right)


def test_new_monoid_normal_forms_and_documented_tree():
    system = parse_srs("aa -> a\nbb -> b\nba -> ab")
    normal = lambda w: system.normal_forms(w, max_len=len(w)).words
    for w, a, b, *_ in system.critical_pairs():
        assert normal(a) == normal(b)
    for state, a, b in table(KEYS["G15"]):
        state = "" if state == "ε" else state
        assert normal(state+"a") == {a} and normal(state+"b") == {b}
    for w in iter_words("ab", 7):
        assert normal(w) == {"".join(c for c in "ab" if c in w)}


def test_new_automata_tables_against_sources():
    rows, delta, finals = table(KEYS["G16"]), {}, set()
    for label, a, b in rows:
        q = label.replace("→", "").replace("★", "")
        if "★" in label:
            finals.add(q)
        for c, cell in zip("ab", (a, b)):
            delta[q, c] = frozenset() if cell == "∅" else frozenset(cell.strip("{}").split(","))
    positional = NFA(frozenset("ab"), "S", frozenset(finals), delta)
    assert counterexample(positional.determinize(), dfa_of("(ab|b)*(ε|a)", "ab")) is None
    minimized = DFA(frozenset("ab"), "X", frozenset({"Y"}), {
        ("X", "a"): "X", ("X", "b"): "Y", ("Y", "a"): "Y", ("Y", "b"): "X"})
    assert "X-a→X, X-b→Y, Y-a→Y, Y-b→X" in KEYS["G17"]
    assert counterexample(documented_dfa(TASKS["G17"], "s", {"f"}), minimized) is None
    deterministic = documented_dfa(KEYS["G18"])
    source = NFA(frozenset("ab"), "0", frozenset({"2"}), {
        ("1", "a"): frozenset({"1", "2"}),
        ("1", "b"): frozenset({"1"})}, {"0": frozenset({"1"})})
    assert counterexample(deterministic, source.determinize()) is None
    product = documented_dfa(KEYS["G19"])
    for w in iter_words("ab", 9):
        assert product.accepts(w) == (w.count("a") % 2 == 0 and w.endswith("b"))


def test_generated_family_key_certificates_on_parameters():
    # Boundary parameter values of the published generic proofs.
    for k in range(1, 5):
        for w in iter_words("ab", 6):
            assert count(w, "b"*k) - count(w, "b"*(k+1)) == sum(
                len(run) >= k for run in w.split("a"))
            assert any(count(w[:i], "a"*k) == w[i:].count("b") for i in range(len(w)+1))
        for r in range(3):
            for i in range(5):
                for j in range(5):
                    assert (k*j+r == k*i+r) == (i == j)
    for c in range(4):
        for i in range(5):
            assert i+c < i+c+1
            for h in range(i+1, 6):
                assert not h+c < i+c+1


def test_modular_family_and_sparse_lengths():
    assert "r→r+1 mod 5" in KEYS["G12"] and "r→r−2" in KEYS["G12"]
    for modulus in range(2, 8):
        for coefficient in range(1, modulus):
            machine = DFA(frozenset("ab"), 0, frozenset({0}), {
                (r, c): (r + (1 if c == "a" else -coefficient)) % modulus
                for r in range(modulus) for c in "ab"})
            for w in iter_words("ab", 6):
                assert machine.accepts(w) == ((w.count("a") - coefficient*w.count("b")) % modulus == 0)
    assert "n(n+1)" in TASKS["G06"] and "2n+2>d" in KEYS["G06"]
    for constant in range(5):
        for period in range(1, 20):
            n = period + 10
            assert n*(n+constant) < n*(n+constant)+period < (n+1)*(n+constant+1)
