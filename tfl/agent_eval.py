"""Сквозной holdout: запуск LLM-агента и проверка его траектории.

Обычный ``evals/suite.py`` вызывает оракулы напрямую. Этот модуль подаёт
агенту только исходное условие, получает структурированный итог и оценивает
весь путь: маршрут, формализацию, вызовы оракулов и честность доказательных
статусов.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOLDOUT_ROOT = ROOT / "evals" / "agent_holdout"
CASES_PATH = HOLDOUT_ROOT / "cases.jsonl"
SCHEMA_PATH = HOLDOUT_ROOT / "response.schema.json"
MANIFEST_PATH = HOLDOUT_ROOT / "manifest.json"
COMMAND_BUDGET = 10

STATUSES = frozenset(
    {"ДОКАЗАНО", "ПРОВЕРЕНО", "ЧАСТИЧНО", "ВРУЧНУЮ", "НЕ ПРОВЕРЕНО"}
)
PROOF_STATUSES = frozenset({"ДОКАЗАНО", "ПРОВЕРЕНО"})


@dataclass(frozen=True)
class HoldoutCase:
    id: str
    source_task_id: str
    year: int
    hint: str
    task_class: str
    statement: str
    oracle_modules: tuple[str, ...]
    formalization_patterns: tuple[str, ...]
    answer_patterns: tuple[str, ...]

    @property
    def recipe(self) -> str:
        return f"docs/recipes/{self.task_class}.md"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HoldoutCase":
        required = {
            "id",
            "source_task_id",
            "year",
            "hint",
            "task_class",
            "statement",
            "oracle_modules",
            "formalization_patterns",
            "answer_patterns",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"holdout-case без полей: {', '.join(missing)}")
        return cls(
            id=str(value["id"]),
            source_task_id=str(value["source_task_id"]),
            year=int(value["year"]),
            hint=str(value["hint"]),
            task_class=str(value["task_class"]),
            statement=str(value["statement"]),
            oracle_modules=tuple(map(str, value["oracle_modules"])),
            formalization_patterns=tuple(map(str, value["formalization_patterns"])),
            answer_patterns=tuple(map(str, value["answer_patterns"])),
        )


@dataclass(frozen=True)
class AgentScore:
    case_id: str
    components: dict[str, int]
    issues: tuple[str, ...]
    runner: str = "unknown"

    @property
    def total(self) -> int:
        return sum(self.components.values())

    @property
    def passed(self) -> bool:
        return (
            self.total >= 8
            and self.components.get("маршрут", 0) == 2
            and self.components.get("запуск оракула", 0) == 1
            and self.components.get("ответ", 0) >= 1
            and not any(issue.startswith("превышен бюджет команд") for issue in self.issues)
        )


def _json_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: не JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: ожидался JSON-объект")
        rows.append(value)
    return rows


def _content_hash(path: pathlib.Path) -> str:
    # read_text normalizes CRLF to LF: один manifest работает на Windows/Unix.
    canonical = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_cases(path: pathlib.Path = CASES_PATH) -> tuple[HoldoutCase, ...]:
    cases = tuple(HoldoutCase.from_dict(row) for row in _json_lines(path))
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("id случаев holdout должны быть уникальны")
    if path.resolve() == CASES_PATH.resolve():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("case_count") != len(cases):
            raise ValueError("число случаев не совпало с manifest.json")
        if manifest.get("cases_sha256") != _content_hash(path):
            raise ValueError("cases.jsonl изменён без новой версии manifest.json")
    return cases


def choose_cases(
    cases: Sequence[HoldoutCase], selected: Sequence[str] = ()
) -> tuple[HoldoutCase, ...]:
    if not selected:
        return tuple(cases)
    wanted = set(selected)
    found = tuple(case for case in cases if case.id in wanted)
    missing = wanted - {case.id for case in found}
    if missing:
        raise ValueError(f"неизвестные случаи: {', '.join(sorted(missing))}")
    return found


def build_prompt(case: HoldoutCase) -> str:
    """Промпт не раскрывает эталонный класс и смысловые якоря scorer-а."""
    return f"""Используй $tfl-solver (в Claude Code тот же skill доступен как /tfl)
и реши задачу по полному циклу S0–S7.

Это изолированный eval. Не читай evals/agent_holdout/cases.jsonl, manifest.json
и прежние runs/reports: там находится закрытая рубрика. Можно читать repo-skill,
AGENTS.md, docs/recipes, corpus и исходный код tfl; Python-оракулы нужно
действительно запускать. Не изменяй файлы репозитория.

Верни только объект по переданной JSON Schema. task_class выбери как один
канонический код без пояснений, recipe — как один точный путь из enum схемы.
В oracle_calls укажи реально выполненные операции и результаты.
ДОКАЗАНО/ПРОВЕРЕНО не ставь по одному ограниченному перебору; границы вынеси
в limitations.

Бюджет траектории: не более {COMMAND_BUDGET} команд. Не читай целиком файлы длиннее 250
строк — сначала найди релевантный фрагмент через rg. Не запускай полный pytest
или весь eval-набор. Достаточно одного предметного оракула и одной независимой
проверки его результата; после доказательства сразу формируй ответ.

case_id: {case.id}
Контекст формы контроля: {case.hint}

Условие:
{case.statement}
"""


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
    except re.error as error:
        raise ValueError(f"ошибка регулярного выражения holdout {pattern!r}: {error}") from error


def _module_name(value: Any) -> str:
    name = str(value).strip().lower().replace("\\", "/")
    name = name.removeprefix("tfl.").removeprefix("tfl/")
    return re.split(r"[./:]", name, maxsplit=1)[0]


def validate_response(response: Any) -> tuple[str, ...]:
    if not isinstance(response, dict):
        return ("ответ не является JSON-объектом",)
    issues: list[str] = []
    for field in ("case_id", "task_class", "recipe", "formalization", "final_answer"):
        if not isinstance(response.get(field), str) or not response[field].strip():
            issues.append(f"пустое или отсутствующее поле {field}")
    for field in ("oracle_calls", "claims", "limitations"):
        if not isinstance(response.get(field), list):
            issues.append(f"поле {field} должно быть массивом")
    calls = response.get("oracle_calls", [])
    if isinstance(calls, list):
        for number, call in enumerate(calls, 1):
            if not isinstance(call, dict):
                issues.append(f"oracle_calls[{number}] не объект")
                continue
            for field in ("module", "operation", "input", "result"):
                if not isinstance(call.get(field), str) or not call[field].strip():
                    issues.append(f"oracle_calls[{number}].{field} пусто")
    claims = response.get("claims", [])
    if isinstance(claims, list):
        for number, claim in enumerate(claims, 1):
            if not isinstance(claim, dict):
                issues.append(f"claims[{number}] не объект")
                continue
            if claim.get("status") not in STATUSES:
                issues.append(f"claims[{number}].status недопустим")
            for field in ("text", "evidence"):
                if not isinstance(claim.get(field), str) or not claim[field].strip():
                    issues.append(f"claims[{number}].{field} пусто")
    return tuple(issues)


def _observed_oracle(commands: Iterable[str]) -> bool:
    pattern = re.compile(
        r"(?<![\w.-])"
        r"(?:py(?:\.exe)?|python(?:3(?:\.\d+)?)?(?:\.exe)?|pytest(?:\.exe)?|tfl-agent)"
        r"(?=\s|[\"']|$)",
        re.I,
    )
    return any(pattern.search(str(command)) for command in commands)


def _overclaims_bounded_evidence(claim: dict[str, Any]) -> bool:
    if claim.get("status") not in PROOF_STATUSES:
        return False
    text = f"{claim.get('text', '')} {claim.get('evidence', '')}".lower()
    bounded = re.search(r"до (?:длины|глубины)|на \d+ (?:слов|пример)|конечн\w* срез", text)
    complete = re.search(r"полный обход|исчерпыва|эквивалентн|инвариант|теорем|контрпример", text)
    return bool(bounded and not complete)


def score_response(
    case: HoldoutCase,
    response: Any,
    observed_commands: Sequence[str] = (),
    *,
    runner: str = "unknown",
    runner_error: str = "",
    command_budget: int | None = None,
) -> AgentScore:
    components = {
        "маршрут": 0,
        "рецепт": 0,
        "формализация": 0,
        "выбор оракула": 0,
        "запуск оракула": 0,
        "свидетельства": 0,
        "честность": 0,
        "ответ": 0,
    }
    issues = list(validate_response(response))
    if runner_error:
        issues.append(f"ошибка runner: {runner_error}")
    if not isinstance(response, dict):
        return AgentScore(case.id, components, tuple(issues), runner)

    if response.get("case_id") != case.id:
        issues.append(f"case_id: ожидался {case.id!r}")
    if response.get("task_class") == case.task_class:
        components["маршрут"] = 2
    else:
        issues.append(f"класс: ожидался {case.task_class}")

    recipe = str(response.get("recipe", "")).replace("\\", "/").lower()
    if recipe.endswith(case.recipe.lower()):
        components["рецепт"] = 1
    else:
        issues.append(f"рецепт: ожидался {case.recipe}")

    formalization = str(response.get("formalization", ""))
    missed_form = [
        pattern for pattern in case.formalization_patterns
        if not _matches(pattern, formalization)
    ]
    if len(formalization.strip()) >= 40 and not missed_form:
        components["формализация"] = 1
    else:
        issues.append("формализация не покрыла: " + ", ".join(missed_form or ["детали условия"]))

    calls = response.get("oracle_calls", [])
    if isinstance(calls, list):
        used = {
            _module_name(call.get("module", ""))
            for call in calls
            if isinstance(call, dict)
        }
        allowed = {_module_name(module) for module in case.oracle_modules}
        if used & allowed:
            components["выбор оракула"] = 1
        else:
            issues.append("не выбран ожидаемый тип оракула: " + ", ".join(case.oracle_modules))
    if _observed_oracle(observed_commands):
        components["запуск оракула"] = 1
    else:
        issues.append("в трассе нет фактического запуска Python-оракула")
    if command_budget is not None and len(observed_commands) > command_budget:
        issues.append(
            f"превышен бюджет команд: {len(observed_commands)} > {command_budget}"
        )

    claims = response.get("claims", [])
    valid_claims = [claim for claim in claims if isinstance(claim, dict)] if isinstance(claims, list) else []
    if valid_claims and all(len(str(claim.get("evidence", "")).strip()) >= 12 for claim in valid_claims):
        components["свидетельства"] = 1
    else:
        issues.append("утверждения не снабжены содержательными свидетельствами")

    overclaims = [claim for claim in valid_claims if _overclaims_bounded_evidence(claim)]
    partial = any(claim.get("status") in {"ЧАСТИЧНО", "ВРУЧНУЮ", "НЕ ПРОВЕРЕНО"} for claim in valid_claims)
    limitations = response.get("limitations", [])
    has_limitations = isinstance(limitations, list) and any(str(item).strip() for item in limitations)
    if not overclaims and (not partial or has_limitations):
        components["честность"] = 1
    else:
        if overclaims:
            issues.append("ограниченный перебор выдан за доказательство")
        if partial and not has_limitations:
            issues.append("частичный статус не объяснён в limitations")

    answer = str(response.get("final_answer", ""))
    matched = sum(_matches(pattern, answer) for pattern in case.answer_patterns)
    if len(answer.strip()) >= 100 and matched == len(case.answer_patterns):
        components["ответ"] = 2
    elif len(answer.strip()) >= 60 and matched >= max(1, (len(case.answer_patterns) + 1) // 2):
        components["ответ"] = 1
        issues.append(f"ответ покрыл только {matched}/{len(case.answer_patterns)} смысловых опор")
    else:
        issues.append(f"ответ покрыл {matched}/{len(case.answer_patterns)} смысловых опор")

    return AgentScore(case.id, components, tuple(issues), runner)


def parse_codex_stream(text: str) -> tuple[Any, tuple[str, ...], dict[str, Any], str]:
    """Извлечь финальный JSON, команды и usage из ``codex exec --json``."""
    response: Any = None
    commands: list[str] = []
    usage: dict[str, Any] = {}
    errors: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"строка {number} потока runner не JSON")
            continue
        event_type = event.get("type")
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if event_type == "item.completed" and item.get("type") == "command_execution":
            command = item.get("command")
            if command:
                commands.append(str(command))
        if event_type == "item.completed" and item.get("type") == "agent_message":
            body = item.get("text")
            try:
                response = json.loads(body) if isinstance(body, str) else body
            except json.JSONDecodeError as error:
                errors.append(f"финальное сообщение не JSON: {error}")
        if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if event_type in {"error", "turn.failed"}:
            errors.append(str(event.get("message") or event.get("error") or event_type))
    if response is None:
        errors.append("в потоке нет структурированного финального ответа")
    return response, tuple(commands), usage, "; ".join(errors)


def parse_claude_stream(
    text: str,
) -> tuple[Any, tuple[str, ...], dict[str, Any], str, str | None]:
    """Извлечь structured output и tool calls из Claude ``stream-json``."""
    response: Any = None
    commands: list[str] = []
    usage: dict[str, Any] = {}
    errors: list[str] = []
    model: str | None = None
    pending_tools: dict[str, str] = {}
    completed_tools: set[str] = set()

    def remember_error(value: Any) -> None:
        message = str(value).strip() if value is not None else ""
        if message and message not in errors:
            errors.append(message)

    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"строка {number} потока runner не JSON")
            continue
        event_type = event.get("type")
        if event_type == "system" and event.get("subtype") == "init":
            if event.get("model"):
                model = str(event["model"])
        if event_type == "assistant" and isinstance(event.get("message"), dict):
            message = event["message"]
            if message.get("model") and message.get("model") != "<synthetic>":
                model = str(message["model"])
            remember_error(message.get("error"))
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tool_id = str(block.get("id", ""))
                    if not tool_id or tool_id in completed_tools:
                        continue
                    name = str(block.get("name", "tool"))
                    inputs = block.get("input") if isinstance(block.get("input"), dict) else {}
                    command = inputs.get("command")
                    if isinstance(command, str) and command.strip():
                        pending_tools[tool_id] = command
                    else:
                        pending_tools[tool_id] = (
                            f"{name} {json.dumps(inputs, ensure_ascii=False, sort_keys=True)}"
                        )
        if event_type == "user" and isinstance(event.get("message"), dict):
            content = event["message"].get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_id = str(block.get("tool_use_id", ""))
                    if tool_id in pending_tools and tool_id not in completed_tools:
                        commands.append(pending_tools[tool_id])
                        completed_tools.add(tool_id)
        if event_type == "result":
            if isinstance(event.get("structured_output"), dict):
                response = event["structured_output"]
            elif isinstance(event.get("result"), str):
                try:
                    candidate = json.loads(event["result"])
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    response = candidate
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if event.get("is_error"):
                remember_error(event.get("result") or event.get("error") or "ошибка Claude")
    if response is None:
        errors.append("в потоке нет структурированного финального ответа")
    return response, tuple(commands), usage, "; ".join(errors), model


def _executable_command(executable: str) -> list[str]:
    """Разрешить native/npm launcher без ``shell=True``."""
    resolved = shutil.which(executable) or executable
    if os.name == "nt" and pathlib.Path(resolved).suffix.lower() in {".bat", ".cmd"}:
        npm_claude = (
            pathlib.Path(resolved).parent
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        if pathlib.Path(resolved).stem.lower() == "claude" and npm_claude.is_file():
            return [str(npm_claude)]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", resolved]
    return [resolved]


def _runner_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [*_executable_command(executable), "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unknown ({error})"
    return (result.stdout or result.stderr).strip() or "unknown"


def _repository_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_codex_case(
    case: HoldoutCase,
    *,
    executable: str = "codex",
    model: str | None = None,
    timeout: int = 900,
    version: str | None = None,
) -> dict[str, Any]:
    command = [
        *_executable_command(executable),
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(SCHEMA_PATH),
        "--json",
        "-C",
        str(ROOT),
    ]
    if model:
        command += ["--model", model]
    command.append("-")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    response: Any = None
    observed_commands: tuple[str, ...] = ()
    usage: dict[str, Any] = {}
    error = ""
    returncode = 1
    try:
        completed = subprocess.run(
            command,
            input=build_prompt(case),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            env=environment,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        response, observed_commands, usage, error = parse_codex_stream(completed.stdout)
        if completed.returncode and not error:
            error = (completed.stderr or f"codex exited {completed.returncode}")[-1200:]
    except subprocess.TimeoutExpired:
        error = f"таймаут {timeout} с"
    except OSError as exc:
        error = f"не удалось запустить {executable}: {exc}"
    return {
        "case_id": case.id,
        "runner": "codex",
        "runner_version": version or _runner_version(executable),
        "model": model or "default",
        "repository_commit": _repository_commit(),
        "started_at": started_at,
        "elapsed_s": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "response": response,
        "observed_commands": list(observed_commands),
        "usage": usage,
        "command_budget": COMMAND_BUDGET,
        "error": error,
    }


def run_codex(
    cases: Sequence[HoldoutCase],
    output: pathlib.Path,
    *,
    executable: str = "codex",
    model: str | None = None,
    timeout: int = 900,
) -> tuple[dict[str, Any], ...]:
    output.parent.mkdir(parents=True, exist_ok=True)
    version = _runner_version(executable)
    records = []
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            record = run_codex_case(
                case, executable=executable, model=model, timeout=timeout, version=version
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
    return tuple(records)


def run_claude_case(
    case: HoldoutCase,
    *,
    executable: str = "claude",
    model: str | None = None,
    timeout: int = 900,
    version: str | None = None,
) -> dict[str, Any]:
    """Запустить один случай через Claude Code с узким allowlist инструментов."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema.pop("$schema", None)
    command = [
        *_executable_command(executable),
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Skill,Read,Grep,Glob,Bash",
        "--allowedTools",
        "Skill(tfl),Read,Grep,Glob,Bash(py -3 *),Bash(python3 *)",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
    ]
    if model:
        command += ["--model", model]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    response: Any = None
    observed_commands: tuple[str, ...] = ()
    usage: dict[str, Any] = {}
    error = ""
    detected_model: str | None = None
    returncode = 1
    try:
        completed = subprocess.run(
            command,
            input=build_prompt(case),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            env=environment,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        response, observed_commands, usage, error, detected_model = parse_claude_stream(
            completed.stdout
        )
        if completed.returncode:
            process_error = completed.stderr[-1200:]
            if process_error and process_error not in error:
                error = "; ".join(part for part in (error, process_error) if part)
            elif not error:
                error = f"claude exited {completed.returncode}"
    except subprocess.TimeoutExpired:
        error = f"таймаут {timeout} с"
    except OSError as exc:
        error = f"не удалось запустить {executable}: {exc}"
    return {
        "case_id": case.id,
        "runner": "claude",
        "runner_version": version or _runner_version(executable),
        "model": model or detected_model or "default",
        "repository_commit": _repository_commit(),
        "started_at": started_at,
        "elapsed_s": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "response": response,
        "observed_commands": list(observed_commands),
        "usage": usage,
        "command_budget": COMMAND_BUDGET,
        "error": error,
    }


def run_claude(
    cases: Sequence[HoldoutCase],
    output: pathlib.Path,
    *,
    executable: str = "claude",
    model: str | None = None,
    timeout: int = 900,
) -> tuple[dict[str, Any], ...]:
    output.parent.mkdir(parents=True, exist_ok=True)
    version = _runner_version(executable)
    records = []
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            record = run_claude_case(
                case, executable=executable, model=model, timeout=timeout, version=version
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
    return tuple(records)


def load_runs(path: pathlib.Path) -> tuple[dict[str, Any], ...]:
    return tuple(_json_lines(path))


def score_runs(
    cases: Sequence[HoldoutCase], records: Sequence[dict[str, Any]]
) -> tuple[AgentScore, ...]:
    by_id = {case.id: case for case in cases}
    seen: set[str] = set()
    scores = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        if case_id not in by_id:
            raise ValueError(f"run содержит неизвестный case_id: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"run содержит повтор case_id: {case_id}")
        seen.add(case_id)
        scores.append(
            score_response(
                by_id[case_id],
                record.get("response"),
                tuple(map(str, record.get("observed_commands", ()))),
                runner=str(record.get("runner", "unknown")),
                runner_error=str(record.get("error", "")),
                command_budget=(
                    int(record["command_budget"])
                    if record.get("command_budget") is not None
                    else None
                ),
            )
        )
    return tuple(scores)


def score_report(
    scores: Sequence[AgentScore], records: Sequence[dict[str, Any]] = ()
) -> str:
    passed = sum(score.passed for score in scores)
    average = sum(score.total for score in scores) / len(scores) if scores else 0.0
    lines = [
        "# Сквозной holdout агента",
        "",
        f"Случаев: **{len(scores)}**, прошли порог 8/10: **{passed}**, "
        f"средний балл: **{average:.2f}/10**.",
        "",
    ]
    if records:
        versions = sorted({str(record.get("runner_version", "unknown")) for record in records})
        models = sorted({str(record.get("model", "default")) for record in records})
        commits = sorted({str(record.get("repository_commit", "unknown")) for record in records})
        elapsed = sum(float(record.get("elapsed_s", 0)) for record in records)
        usage_keys = (
            "input_tokens",
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
        present_usage_keys = {
            key
            for record in records
            if isinstance(record.get("usage"), dict)
            for key in record["usage"]
        }
        usage = {
            key: sum(
                int(record.get("usage", {}).get(key, 0))
                for record in records
                if isinstance(record.get("usage"), dict)
            )
            for key in usage_keys
            if key in present_usage_keys
        }
        lines += [
            f"Runner: `{', '.join(versions)}`; model: `{', '.join(models)}`.",
            f"Коммит harness: `{', '.join(commits)}`; суммарное время: **{elapsed:.2f} с**.",
            "Usage: "
            + (", ".join(f"{key}={value}" for key, value in usage.items()) or "нет данных")
            + ".",
            "",
        ]
    lines += [
        "| случай | runner | балл | маршрут | итог |",
        "|---|---|---:|---:|---|",
    ]
    for score in scores:
        lines.append(
            f"| `{score.case_id}` | {score.runner} | {score.total}/10 | "
            f"{score.components['маршрут']}/2 | {'PASS' if score.passed else 'FAIL'} |"
        )
    lines += ["", "## Диагностика", ""]
    for score in scores:
        lines += [f"### `{score.case_id}` — {score.total}/10", ""]
        lines.append(
            ", ".join(f"{name}: {value}" for name, value in score.components.items())
        )
        lines.append("")
        if score.issues:
            lines.extend(f"- {issue}" for issue in score.issues)
        else:
            lines.append("- замечаний нет")
        lines.append("")
    return "\n".join(lines)
