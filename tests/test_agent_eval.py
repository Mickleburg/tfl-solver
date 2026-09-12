"""Сквозной holdout: заморозка, контракт ответа и детерминированный scorer."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import tfl.agent_eval as agent_eval
from tfl.agent_eval import (
    CASES_PATH,
    HOLDOUT_ROOT,
    build_prompt,
    load_cases,
    parse_claude_stream,
    parse_codex_stream,
    run_claude_case,
    score_response,
    score_runs,
)
from tfl.cli import main
from tfl.intake import load_index


def good_response(case):
    return {
        "case_id": case.id,
        "task_class": case.task_class,
        "recipe": case.recipe,
        "formalization": (
            "Алфавит {a,b,c}; условие: есть хотя бы две буквы c или две "
            "одинаковыми соседними буквами. Требуется минимальный DFA."
        ),
        "oracle_calls": [
            {
                "module": "tfl.automata",
                "operation": "minimize and check equivalence",
                "input": "DFA over abc",
                "result": "эквивалентность подтверждена, классы различимы",
            }
        ],
        "claims": [
            {
                "status": "ДОКАЗАНО",
                "text": "Построенный DFA минимален.",
                "evidence": "Для каждой пары состояний предъявлено различающее продолжение.",
            }
        ],
        "final_answer": (
            "Построен минимальный DFA: его состояния запоминают число встреченных c "
            "до двух и последнюю букву до попадания в принимающую ловушку. "
            "Минимальность следует из попарно различающих продолжений состояний."
        ),
        "limitations": [],
    }


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_holdout_is_frozen_and_spans_every_historical_year(cases):
    manifest = json.loads((HOLDOUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        CASES_PATH.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert manifest["version"] == 1
    assert manifest["case_count"] == len(cases) == 7
    assert manifest["years"] == sorted({case.year for case in cases}) == list(range(2021, 2026))
    assert manifest["cases_sha256"] == digest


def test_holdout_sources_exist_in_the_task_index(cases):
    records = {record["id"]: record for record in load_index().records}
    for case in cases:
        source = records[case.source_task_id]
        assert source["year"] == case.year
        assert source["form"] == case.hint
        assert source["code"] == case.task_class


def test_response_schema_restricts_route_to_canonical_values(cases):
    schema = json.loads(
        (HOLDOUT_ROOT / "response.schema.json").read_text(encoding="utf-8")
    )
    classes = schema["properties"]["task_class"]["enum"]
    recipes = schema["properties"]["recipe"]["enum"]
    assert {case.task_class for case in cases} <= set(classes)
    assert {case.recipe for case in cases} <= set(recipes)
    assert len(classes) == len(recipes) == 17


def test_prompt_contains_task_but_not_expected_route(cases):
    for case in cases:
        prompt = build_prompt(case)
        assert case.statement in prompt
        assert case.task_class not in prompt
        assert "$tfl-solver" in prompt
        assert "Не читай evals/agent_holdout" in prompt
        assert "не более 10 команд" in prompt


def test_complete_observed_trajectory_scores_ten(cases):
    case = cases[0]
    score = score_response(case, good_response(case), ["py -3 -c import tfl.automata"])
    assert score.total == 10
    assert score.passed
    assert not score.issues


def test_claimed_but_unobserved_oracle_does_not_pass(cases):
    case = cases[0]
    score = score_response(case, good_response(case))
    assert score.total == 9
    assert not score.passed
    assert any("фактического запуска" in issue for issue in score.issues)


def test_wrong_route_is_a_hard_failure(cases):
    case = cases[0]
    response = good_response(case)
    response["task_class"] = "EXAM-1"
    score = score_response(case, response, ["python -m tfl doctor"])
    assert score.components["маршрут"] == 0
    assert not score.passed


def test_bounded_experiment_cannot_be_called_a_proof(cases):
    case = cases[0]
    response = good_response(case)
    response["claims"][0]["evidence"] = "Проверено до длины 8, расхождений нет."
    score = score_response(case, response, ["py -3 oracle.py"])
    assert score.components["честность"] == 0
    assert any("ограниченный перебор" in issue for issue in score.issues)


def test_command_budget_is_a_hard_gate(cases):
    case = cases[0]
    commands = [f"py -3 check-{number}.py" for number in range(11)]
    score = score_response(
        case, good_response(case), commands, command_budget=10
    )
    assert score.total == 10
    assert not score.passed
    assert any("превышен бюджет команд" in issue for issue in score.issues)


def test_codex_jsonl_stream_keeps_commands_and_final_response(cases):
    response = good_response(cases[0])
    events = [
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "py -3 check.py"},
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": json.dumps(response, ensure_ascii=False)},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 20}},
    ]
    body = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    parsed, commands, usage, error = parse_codex_stream(body)
    assert parsed == response
    assert commands == ("py -3 check.py",)
    assert usage["output_tokens"] == 20
    assert not error


def test_claude_jsonl_stream_keeps_tool_calls_and_structured_output(cases):
    response = good_response(cases[0])
    events = [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-test"},
        {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "read-1",
                        "name": "Read",
                        "input": {"file_path": "docs/PROJECT-STATE.md"},
                    },
                    {
                        "type": "tool_use",
                        "id": "bash-1",
                        "name": "Bash",
                        "input": {"command": "py -3 -m tfl doctor"},
                    },
                    {
                        "type": "tool_use",
                        "id": "bash-not-completed",
                        "name": "Bash",
                        "input": {"command": "py -3 denied.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "read-1", "content": "ok"},
                    {"type": "tool_result", "tool_use_id": "bash-1", "content": "ok"},
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": response,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    ]
    body = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    parsed, commands, usage, error, model = parse_claude_stream(body)
    assert parsed == response
    assert commands == (
        'Read {"file_path": "docs/PROJECT-STATE.md"}',
        "py -3 -m tfl doctor",
    )
    assert usage["output_tokens"] == 20
    assert model == "claude-sonnet-test"
    assert not error


def test_claude_jsonl_stream_surfaces_api_error():
    body = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "oauth_org_not_allowed",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )
    response, commands, usage, error, model = parse_claude_stream(body)
    assert response is None
    assert not commands
    assert usage["input_tokens"] == 0
    assert "oauth_org_not_allowed" in error
    assert "нет структурированного" in error
    assert model is None


def test_claude_runner_restricts_tools_and_adapts_schema(monkeypatch, cases):
    response = good_response(cases[0])
    stream = "\n".join(
        json.dumps(event, ensure_ascii=False)
        for event in (
            {"type": "system", "subtype": "init", "model": "claude-test"},
            {
                "type": "result",
                "is_error": False,
                "structured_output": response,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=stream, stderr="")

    monkeypatch.setattr(agent_eval, "_executable_command", lambda executable: [executable])
    monkeypatch.setattr(agent_eval, "_repository_commit", lambda: "commit-test")
    monkeypatch.setattr(agent_eval.subprocess, "run", fake_run)
    record = run_claude_case(cases[0], version="claude-test-version")

    command = captured["command"]
    tools = command[command.index("--tools") + 1]
    allowed = command[command.index("--allowedTools") + 1]
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert tools == "Skill,Read,Grep,Glob,Bash"
    assert allowed == "Skill(tfl),Read,Grep,Glob,Bash(py -3 *),Bash(python3 *)"
    assert "$schema" not in schema
    assert schema["additionalProperties"] is False
    assert captured["kwargs"]["cwd"] == agent_eval.ROOT
    assert cases[0].statement in captured["kwargs"]["input"]
    assert record["response"] == response
    assert record["model"] == "claude-test"
    assert record["repository_commit"] == "commit-test"


def test_score_runs_rejects_duplicate_cases(cases):
    record = {
        "case_id": cases[0].id,
        "runner": "test",
        "response": good_response(cases[0]),
        "observed_commands": ["py -3 check.py"],
        "error": "",
    }
    with pytest.raises(ValueError, match="повтор"):
        score_runs(cases, [record, record])


def test_cli_lists_and_prints_holdout_prompt(capsys):
    assert main(["holdout", "list"]) == 0
    assert "pharma-2021-A1" in capsys.readouterr().out
    assert main(["holdout", "prompt", "--case", "pharma-2022-A11"]) == 0
    assert "суффиксной фильтрации" in capsys.readouterr().out


def test_cli_scores_a_saved_run(tmp_path, cases, capsys):
    case = cases[0]
    record = {
        "case_id": case.id,
        "runner": "test",
        "response": good_response(case),
        "observed_commands": ["py -3 check.py"],
        "runner_version": "test-runner 1.0",
        "model": "test-model",
        "repository_commit": "0123456789abcdef",
        "elapsed_s": 1.5,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "error": "",
    }
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    report = tmp_path / "report.md"
    assert main(["holdout", "score", "--input", str(run), "--report", str(report)]) == 0
    assert "10/10" in report.read_text(encoding="utf-8")
    assert "input_tokens=100" in report.read_text(encoding="utf-8")
    assert "PASS" in capsys.readouterr().out


def test_cli_requires_explicit_run_scope(tmp_path, capsys):
    output = tmp_path / "run.jsonl"
    assert main(["holdout", "run", "--output", str(output)]) == 2
    assert "--case" in capsys.readouterr().err
    assert not output.exists()
