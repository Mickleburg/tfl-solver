"""Педагогический holdout: режим, аналог, признаки и малые приёмы."""

from __future__ import annotations

import hashlib
import json

from tfl.cli import main
from tfl.intake import load_seminar_index
from tfl.pedagogy_eval import (
    CASES_PATH,
    MANIFEST_PATH,
    build_prompt,
    load_cases,
    score_response,
)


def good_response(case):
    assert case.id == "training-unbounded-delay"
    return {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-unbounded-delay",
        "observations": [
            {
                "text": "Это морфизм, полностью заданный образами букв.",
                "evidence": "В условии явно записаны h(x), h(y), h(z).",
            },
            {
                "text": "Задержка определяется длиной общего префикса образов.",
                "evidence": "Нужно различить первую букву прообраза x или y.",
            },
        ],
        "micro_methods": [
            "Сначала отличить префиксность от однозначной декодируемости.",
            "Заменить конечные примеры параметрическим семейством для произвольного k.",
        ],
        "prerequisites_used": ["морфизм слов", "однозначно декодируемый код"],
        "chosen_method": (
            "Проверить код алгоритмом Сардинаса–Паттерсона, затем взять xz^k и "
            "yz^(k-1), образы которых имеют растущий общий префикс."
        ),
        "oracle_calls": [
            {
                "module": "tfl.code",
                "operation": "uniquely_decodable",
                "input": "0, 01, 11",
                "result": "код однозначно декодируем",
            }
        ],
        "final_answer": (
            "Морфизм инъективен, поскольку код {0,01,11} однозначно "
            "декодируем по алгоритму Сардинаса–Паттерсона. Задержка "
            "неограничена: для произвольного k образы xz^k и yz^(k-1) "
            "имеют общий префикс, длина которого растёт вместе с k."
        ),
        "limitations": ["Конечная кривая задержки используется только как иллюстрация."],
    }


def test_pedagogy_holdout_is_frozen_and_has_both_modes():
    cases = load_cases()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CASES_PATH.read_text(encoding="utf-8").encode()).hexdigest()
    assert manifest["version"] == 1
    assert manifest["rubric_revision"] == 4
    assert manifest["case_count"] == len(cases) == 8
    assert manifest["cases_sha256"] == digest
    assert {case.mode for case in cases} == {"training", "full"}


def test_all_named_seminar_analogs_exist():
    known = {record["id"] for record in load_seminar_index().records}
    assert all(
        not case.seminar_analog or case.seminar_analog in known
        for case in load_cases()
    )


def test_prompt_exposes_mode_and_task_but_hides_rubric():
    for case in load_cases():
        prompt = build_prompt(case)
        assert case.statement in prompt
        assert f"Режим: {case.mode}" in prompt
        assert f"--mode {case.mode}" in prompt
        assert f'--hint "{case.hint}"' in prompt
        assert "Не трать команды на чтение `tfl/cli.py`" in prompt
        if case.seminar_analog:
            assert case.seminar_analog not in prompt
        assert "Не читай evals/pedagogy_holdout" in prompt


def test_complete_pedagogical_response_scores_thirteen():
    case = load_cases()[0]
    score = score_response(case, good_response(case), ["py -3 -c import tfl.code"])
    assert score.total == 13
    assert score.passed
    assert not score.issues


def test_wrong_analog_is_a_hard_failure():
    case = load_cases()[0]
    response = good_response(case)
    response["seminar_analog"] = "sem-2026-09-05-rotation"
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["аналог"] == 0
    assert not score.passed


def test_unobserved_oracle_is_a_hard_failure():
    case = load_cases()[0]
    score = score_response(case, good_response(case))
    assert score.components["запуск оракула"] == 0
    assert not score.passed


def test_forbidden_heavy_shortcut_is_rejected():
    case = load_cases()[0]
    response = good_response(case)
    response["chosen_method"] += " Затем применить матричную интерпретацию."
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["границы"] == 0
    assert not score.passed
    assert any("запрещённое сокращение" in issue for issue in score.issues)


def test_missing_verification_boundary_loses_component():
    case = load_cases()[0]
    response = good_response(case)
    response["limitations"] = []
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["границы"] == 0
    assert not score.passed
    assert any("граница машинной проверки" in issue for issue in score.issues)


def test_missing_micro_methods_is_a_hard_failure():
    case = load_cases()[0]
    response = good_response(case)
    response["micro_methods"] = []
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["малые приёмы"] == 0
    assert not score.passed


def test_cli_lists_and_prints_pedagogy_prompt(capsys):
    assert main(["pedagogy", "list"]) == 0
    assert "training-unbounded-delay" in capsys.readouterr().out
    assert main(
        ["pedagogy", "prompt", "--case", "training-logic-collapse"]
    ) == 0
    output = capsys.readouterr().out
    assert "режиме training" in output
    assert "n,m,k" in output
