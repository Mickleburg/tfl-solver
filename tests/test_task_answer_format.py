from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from tfl.automata import DFA, NFA, counterexample, dfa_of
from tfl.glushkov import glushkov
from tfl.regex import parse
from tfl.words import iter_words
from tools.check_task_answer import check_document, check_task, main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "evals/task_answer_format/examples.md"
SAMPLE = """## Задача. Однозначность

### Условие

Дано выражение `((a|aa)b|abaab)*`.
Требуется установить, является ли выражение однозначным, и обосновать ответ.

<details>
<summary>Подсказка к решению</summary>

Сравни длинный блок с последовательностью коротких.

</details>

<details>
<summary>Решение</summary>

У слова `abaab` два разбора: один блок либо `ab` и `aab`.
Ответ: выражение неоднозначно.

</details>
"""


def test_examples_have_open_conditions_and_two_closed_sibling_spoilers():
    assert check_document(EXAMPLES.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (("### Условие", "### Данные"), "missing-condition"),
        (("Требуется установить", "Следует установить"), "missing-requirement"),
        (("<details>", "<details open>", 1), "open-details"),
        (("Подсказка к решению", "Ответ"), "wrong-spoiler-label-or-order"),
        (("<summary>Решение</summary>", ""), "wrong-spoiler-label-or-order"),
        (("</details>", "", 1), "nested-details"),
        (("</details>", "", 1), "unclosed-details"),
        (("Сравни длинный блок с последовательностью коротких.", ""), "empty-spoiler"),
    ],
)
def test_structural_regressions(replacement, expected):
    assert expected in check_task(SAMPLE.replace(*replacement))


def test_answer_cannot_be_appended_outside_solution():
    assert "text-outside-spoilers" in check_task(SAMPLE + "\nОтвет: неоднозначно.\n")


def test_answer_cannot_be_inserted_between_hint_and_solution():
    changed = SAMPLE.replace("</details>", "</details>\nОтвет: неоднозначно.\n", 1)
    assert "text-outside-spoilers" in check_task(changed)


def test_duplicate_summary_is_rejected():
    changed = SAMPLE.replace("<summary>Решение</summary>",
                             "<summary>Решение</summary><summary>Решение</summary>")
    assert "wrong-spoiler-label-or-order" in check_task(changed)


def test_reversed_spoilers_are_rejected():
    changed = SAMPLE.replace("Подсказка к решению", "TEMP")
    changed = changed.replace("<summary>Решение</summary>",
                              "<summary>Подсказка к решению</summary>")
    changed = changed.replace("TEMP", "Решение")
    assert "wrong-spoiler-label-or-order" in check_task(changed)


def test_fenced_html_and_task_headings_do_not_create_real_spoilers_or_tasks():
    code = '\n```html\n## Задача. Пример кода\n<details open></details>\n```\n'
    changed = SAMPLE.replace("Сравни длинный блок", code + "Сравни длинный блок")
    assert check_document(changed) == []


def test_every_task_is_checked_independently():
    bad = SAMPLE.replace("## Задача.", "## Задача 2.").replace("<details>", "<details open>", 1)
    issues = check_document(SAMPLE + "\n" + bad)
    assert len(issues) == 1
    assert "Задача 2" in issues[0] and "open-details" in issues[0]


def test_math_inequalities_and_inline_html_examples_remain_literal_text():
    changed = SAMPLE.replace("У слова `abaab`", "При $0\\le k<n$ и `x<y` <em>читаем</em> "
                             "`<details open>` как пример кода. У слова `abaab`")
    assert check_task(changed) == []


def test_structural_pass_is_not_a_semantic_or_language_verdict():
    changed = SAMPLE.replace("Сравни длинный блок с последовательностью коротких.",
                             "Ответ: неоднозначно; готовый разбор abaab = ab · aab.")
    # Deliberately bad hint still has valid structure: an editorial audit is needed.
    assert check_task(changed) == []


def test_cli_reports_structure_only_and_nonzero_for_invalid_answer(tmp_path, capsys):
    assert main([str(EXAMPLES)]) == 0
    assert "structure only" in capsys.readouterr().out
    invalid = tmp_path / "answer.md"
    invalid.write_text(SAMPLE + "\nОтвет раскрыт.\n", encoding="utf-8")
    assert main([str(invalid)]) == 1
    assert "text-outside-spoilers" in capsys.readouterr().out


def _paths(nfa, word):
    counts = {nfa.start: 1}
    for char in word:
        following = defaultdict(int)
        for source, count in counts.items():
            for target in nfa.delta.get((source, char), ()):
                following[target] += count
        counts = following
    return sum(counts.get(state, 0) for state in nfa.finals)


def _block_parses(word, blocks):
    counts = [0] * (len(word) + 1)
    counts[0] = 1
    for index in range(len(word)):
        for block in blocks:
            if word.startswith(block, index):
                counts[index + len(block)] += counts[index]
    return counts[-1]


def test_determinization_example_preserves_the_original_language_exactly():
    alphabet = frozenset("ab")
    original = NFA(alphabet, 1, frozenset({3}), {
        (1, "a"): frozenset({3}), (2, "a"): frozenset({2}),
        (2, "b"): frozenset({2, 3}),
    }, {1: frozenset({2})})
    answer = DFA(alphabet, "A", frozenset({"B"}), {
        ("A", "a"): "B", ("A", "b"): "B",
        ("B", "a"): "C", ("B", "b"): "B",
        ("C", "a"): "C", ("C", "b"): "B",
    })
    assert counterexample(original.determinize(), answer) is None
    assert counterexample(answer, dfa_of("a|(a|b)*b")) is None


def test_minimality_example_preserves_language_and_has_two_distinct_states():
    original = DFA(frozenset("ab"), 1, frozenset({3}), {
        (1, "a"): 2, (1, "b"): 3, (2, "a"): 1,
        (2, "b"): 3, (3, "a"): 3, (3, "b"): 2,
    })
    answer = DFA(frozenset("ab"), "X", frozenset({"Y"}), {
        ("X", "a"): "X", ("X", "b"): "Y",
        ("Y", "a"): "Y", ("Y", "b"): "X",
    })
    assert counterexample(original, answer) is None
    assert len(answer.minimize()) == 2


def test_corrected_regex_examples_match_independent_parse_counts():
    c = glushkov(parse("(a|ab)*(b|ba)*"))
    g = glushkov(parse("((a|aa)b|abaab)*"))
    for word in iter_words("ab", 10):
        split_count = sum(
            _block_parses(word[:index], ("a", "ab"))
            * _block_parses(word[index:], ("b", "ba"))
            for index in range(len(word) + 1)
        )
        assert _paths(c, word) == split_count
        assert _paths(g, word) == _block_parses(word, ("ab", "aab", "abaab"))
    assert _paths(c, "ab") == 2
    assert _paths(g, "abaab") == 2


def test_corrected_regex_witness_families():
    c = glushkov(parse("(a|ab)*(b|ba)*"))
    g = glushkov(parse("((a|aa)b|abaab)*"))
    for n in range(1, 101):
        assert _paths(c, "ab" * n) == n + 1
        assert _paths(g, "abaab" * n) == 2 ** n
