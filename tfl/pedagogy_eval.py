"""Педагогический holdout режимов training/full.

Исторический ``agent_holdout`` измеряет полный предметный маршрут. Этот набор
сосредоточен на промежутке до основной теоремы: построить воспроизводимый путь
от зацепки в условии через малое действие и его результат к следующему шагу,
выбрать знакомый аналог и не заменить понятное решение необязательным тяжёлым
аппаратом.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

from tfl import agent_eval
from tfl.intake import analyse, load_seminar_index
from tfl.paths import ROOT


PEDAGOGY_ROOT = ROOT / "evals" / "pedagogy_holdout"
CASES_PATH = PEDAGOGY_ROOT / "cases.jsonl"
SCHEMA_PATH = PEDAGOGY_ROOT / "response.schema.json"
MANIFEST_PATH = PEDAGOGY_ROOT / "manifest.json"
COMMAND_BUDGET = 2


@dataclass(frozen=True)
class PedagogyCase:
    id: str
    mode: str
    hint: str
    source: str
    task_class: str
    seminar_analog: str
    statement: str
    oracle_modules: tuple[str, ...]
    signal_patterns: tuple[str, ...]
    micro_method_patterns: tuple[str, ...]
    method_patterns: tuple[str, ...]
    answer_patterns: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PedagogyCase":
        required = {
            "id",
            "mode",
            "hint",
            "source",
            "task_class",
            "seminar_analog",
            "statement",
            "oracle_modules",
            "signal_patterns",
            "micro_method_patterns",
            "method_patterns",
            "answer_patterns",
            "forbidden_patterns",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"pedagogy-case без полей: {', '.join(missing)}")
        mode = str(value["mode"])
        if mode not in {"training", "full"}:
            raise ValueError(f"неизвестный педагогический режим: {mode}")
        return cls(
            id=str(value["id"]),
            mode=mode,
            hint=str(value["hint"]),
            source=str(value["source"]),
            task_class=str(value["task_class"]),
            seminar_analog=str(value["seminar_analog"]),
            statement=str(value["statement"]),
            oracle_modules=tuple(map(str, value["oracle_modules"])),
            signal_patterns=tuple(map(str, value["signal_patterns"])),
            micro_method_patterns=tuple(map(str, value["micro_method_patterns"])),
            method_patterns=tuple(map(str, value["method_patterns"])),
            answer_patterns=tuple(map(str, value["answer_patterns"])),
            forbidden_patterns=tuple(map(str, value["forbidden_patterns"])),
        )


@dataclass(frozen=True)
class PedagogyScore:
    case_id: str
    components: dict[str, int]
    issues: tuple[str, ...]
    runner: str = "unknown"
    valid_response: bool = True
    infrastructure_error: bool = False

    @property
    def total(self) -> int:
        return sum(self.components.values())

    @property
    def content_passed(self) -> bool:
        """Прошла ли содержательная часть независимо от стоимости маршрута."""
        hard = (
            self.valid_response
            and self.components.get("режим", 0) == 1
            and self.components.get("класс", 0) == 1
            and self.components.get("аналог", 0) == 2
            and self.components.get("путь к решению", 0) == 1
            and self.components.get("малые приёмы", 0) == 2
            and self.components.get("метод", 0) == 1
            and self.components.get("выбор оракула", 0) == 1
            and self.components.get("запуск оракула", 0) == 1
            and self.components.get("ответ", 0) >= 1
            and self.components.get("границы", 0) == 1
        )
        return self.total >= 10 and hard

    @property
    def within_budget(self) -> bool:
        """Уложился ли маршрут в лимит наблюдаемых вызовов инструментов."""
        return not any(issue.startswith("превышен бюджет команд") for issue in self.issues)

    @property
    def passed(self) -> bool:
        return (
            not self.infrastructure_error
            and self.content_passed
            and self.within_budget
        )


def _json_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    canonical = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_cases(path: pathlib.Path = CASES_PATH) -> tuple[PedagogyCase, ...]:
    cases = tuple(PedagogyCase.from_dict(row) for row in _json_lines(path))
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("id педагогических случаев должны быть уникальны")
    if path.resolve() == CASES_PATH.resolve():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("case_count") != len(cases):
            raise ValueError("число педагогических случаев не совпало с manifest")
        if manifest.get("cases_sha256") != _content_hash(path):
            raise ValueError("cases.jsonl изменён без новой версии manifest")
    return cases


def choose_cases(
    cases: Sequence[PedagogyCase], selected: Sequence[str] = ()
) -> tuple[PedagogyCase, ...]:
    if not selected:
        return tuple(cases)
    wanted = set(selected)
    found = tuple(case for case in cases if case.id in wanted)
    missing = wanted - {case.id for case in found}
    if missing:
        raise ValueError(f"неизвестные случаи: {', '.join(sorted(missing))}")
    return found


def build_prompt(case: PedagogyCase) -> str:
    analysis = analyse(
        case.statement,
        case.hint,
        mode=case.mode,
        seminar_index=load_seminar_index(),
    )
    briefing = analysis.solver_briefing()
    proposed_class = (
        analysis.candidates[0].code
        if analysis.confident and analysis.candidates
        else "сверить кандидатов стартового пакета"
    )
    proposed_analog = (
        str(analysis.seminar_similar[0][1]["id"])
        if analysis.seminar_similar
        else "пустая строка"
    )
    return f"""Реши учебную задачу по контракту tfl-solver в режиме {case.mode}.
Не раскрывай скрытую цепочку рассуждений: нужен короткий
воспроизводимый учебный путь из наблюдаемых действий и результатов.

Это изолированный eval. Не читай evals/pedagogy_holdout/cases.jsonl,
manifest.json и прошлые отчёты: там закрытая рубрика. Все необходимые опоры
уже включены ниже; репозиторий не меняй и файлы не читай.

В training сначала проверь уже пройденные семинары. Переноси только шаги с
подтверждёнными предпосылками; если подходящего аналога нет, seminar_analog
оставь пустой строкой. В full также проверь аналоги, но рассмотри необходимые
альтернативные формализации. В discovery_path покажи два–пять связанных
учебных шагов: конкретная зацепка в условии, небольшое действие, его результат
и следующий выбор, который из результата следует. Не подменяй этот путь
перечнем готовых наблюдений и не привязывайся к фразе «заметим, что». Каждый
шаг должен быть сам по себе математически верным и согласованным с предыдущим;
правильный финальный ответ не исправляет ложную зацепку.
В micro_methods перечисли действия до основной теоремы. Выполни предметный
Python-оракул и честно запиши границы результата.

Ниже уже дан детерминированный стартовый пакет: не вызывай skill, intake,
read или glob, чтобы заново собрать те же сведения. Карточка в пакете — лишь
кандидат на перенос: явно сверь её предпосылки и формальный объект с условием.
Если объект тот же, выполни приведённую команду буквально. Если карточка
проверяет другие правила или другой объект, перенеси тип проверки, но подставь
объект текущей задачи; не выдавай запуск на старой задаче за проверку новой.
Для обычной SRS без переменных используй шаблон `from tfl.srs import
parse_srs; s=parse_srs('RULES'); print(s.terminates())`; для правил со
строковыми переменными используй `tfl.pattern`. Если команды или подходящего
шаблона нет, разрешён один точечный Python-вызов.
Бюджет — не более {COMMAND_BUDGET} вызовов инструментов суммарно. Не исследуй
исходники или синтаксис API. В Linux используй `python3`, в Windows — `py -3`.
Не реализуй формальную семантику самодельным скриптом. Не переноси из карточки
инварианты, нормальные формы или конфлюэнтность, если текущая задача спрашивает
только завершимость. В `oracle_calls.module` укажи импортированный модуль `tfl` без
префикса, например импорт `tfl.pattern` означает точное значение `pattern`;
записи `import tfl.srs` и `from tfl.srs import parse_srs` обе означают `srs`.
Только если команда не импортирует `tfl`, укажи `python`.

Работай последовательно, не выводя промежуточный черновик:

1. Сверка без инструментов. Сопоставь формальный объект, вопрос и предпосылки
   карточки с условием. Предложенный класс: `{proposed_class}`; предложенный
   аналог: `{proposed_analog}`. Не принимай их, если предпосылки расходятся.
   Если принимаешь, запиши в `task_class` и `seminar_analog` только эти точные
   идентификаторы без пояснений; пояснение переноса помести в путь решения.
2. Одна исполняемая проверка. Если строка «проверка» содержит тот же формальный
   объект, запусти её ровно. Если объект отличается, адаптируй только вход и
   подходящий предметный модуль по шаблонам выше; иначе составь один точечный
   Python-вызов по указанной проверке.
   После успешного stdout не вызывай больше инструменты. Конечный вывод
   оракула используй только в заявленных границах.
3. Общее доказательство. Построй два–пять связанных шагов
   clue → action → result → next_step. Для общего утверждения добавь формулу,
   инвариант или параметрическое семейство; конечной таблицы недостаточно.
   В `micro_methods` буквально перенеси все применимые пункты из строки
   «малые приёмы карточки» (если их больше трёх — минимум три), сохранив их
   ключевые слова; не заменяй их другими приёмами. В `chosen_method` назови и
   проверку оракулом, и общий свидетель.
4. Сериализация. Верни только один JSON-объект по переданной Schema. Перед
   отправкой проверь: `final_answer` и `chosen_method` — строки, не массивы;
   `discovery_path` — массив минимум из двух объектов со всеми четырьмя
   строковыми полями; `micro_methods`, `prerequisites_used`, `oracle_calls` и
   `limitations` — массивы; никаких дополнительных полей. Не используй bash,
   Python, `cat` или heredoc для построения и проверки JSON: сформируй
   финальный объект непосредственно в ответе.

{briefing}

case_id: {case.id}
Режим: {case.mode}
Контекст: {case.hint}

Условие:
{case.statement}
"""


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE | re.DOTALL) is not None
    except re.error as error:
        raise ValueError(f"ошибка регулярного выражения {pattern!r}: {error}") from error


def _module_name(value: Any) -> str:
    name = str(value).strip().lower().replace("\\", "/")
    name = name.removeprefix("tfl.").removeprefix("tfl/")
    if "python" in name:
        return "python"
    return re.split(r"[./:]", name, maxsplit=1)[0]


def _is_infrastructure_error(message: str) -> bool:
    """Отличить отказ runner/provider от некорректного ответа модели."""
    if not message.strip():
        return False
    return re.search(
        r"\b(?:401|403|429|5\d\d)\b|service unavailable|err_connect_fail|"
        r"connection (?:timed out|refused)|oauth_org_not_allowed|"
        r"executable .*(?:not found|не найден)",
        message,
        re.IGNORECASE,
    ) is not None


def validate_response(response: Any) -> tuple[str, ...]:
    if not isinstance(response, dict):
        return ("ответ не является JSON-объектом",)
    issues: list[str] = []
    for field in (
        "case_id",
        "mode",
        "task_class",
        "seminar_analog",
        "chosen_method",
        "final_answer",
    ):
        if not isinstance(response.get(field), str):
            issues.append(f"поле {field} должно быть строкой")
    for field in (
        "discovery_path",
        "micro_methods",
        "prerequisites_used",
        "oracle_calls",
        "limitations",
    ):
        if not isinstance(response.get(field), list):
            issues.append(f"поле {field} должно быть массивом")
    discovery_path = response.get("discovery_path", ())
    for number, step in enumerate(discovery_path, 1):
        if not isinstance(step, dict):
            issues.append(f"discovery_path[{number}] не объект")
            continue
        for field in ("clue", "action", "result", "next_step"):
            if not str(step.get(field, "")).strip():
                issues.append(f"discovery_path[{number}].{field} пусто")
    if isinstance(discovery_path, list) and len(discovery_path) < 2:
        issues.append("путь к решению должен содержать хотя бы два связанных шага")
    for number, call in enumerate(response.get("oracle_calls", ()), 1):
        if not isinstance(call, dict):
            issues.append(f"oracle_calls[{number}] не объект")
            continue
        for field in ("module", "operation", "input", "result"):
            if not str(call.get(field, "")).strip():
                issues.append(f"oracle_calls[{number}].{field} пусто")
    return tuple(issues)


def score_response(
    case: PedagogyCase,
    response: Any,
    observed_commands: Sequence[str] = (),
    *,
    runner: str = "unknown",
    runner_error: str = "",
    command_budget: int | None = None,
) -> PedagogyScore:
    components = {
        "режим": 0,
        "класс": 0,
        "аналог": 0,
        "путь к решению": 0,
        "малые приёмы": 0,
        "метод": 0,
        "выбор оракула": 0,
        "запуск оракула": 0,
        "ответ": 0,
        "границы": 0,
    }
    validation_issues = validate_response(response)
    issues = list(validation_issues)
    infrastructure_error = _is_infrastructure_error(runner_error)
    if runner_error:
        issues.append(f"ошибка runner: {runner_error}")
    if not isinstance(response, dict):
        return PedagogyScore(
            case.id,
            components,
            tuple(issues),
            runner,
            valid_response=False,
            infrastructure_error=infrastructure_error,
        )

    if response.get("case_id") != case.id:
        issues.append(f"case_id: ожидался {case.id!r}")
    if response.get("mode") == case.mode:
        components["режим"] = 1
    else:
        issues.append(f"режим: ожидался {case.mode}")
    if response.get("task_class") == case.task_class:
        components["класс"] = 1
    else:
        issues.append(f"класс: ожидался {case.task_class}")
    if str(response.get("seminar_analog", "")).strip() == case.seminar_analog:
        components["аналог"] = 2
    else:
        expected = case.seminar_analog or "пустая строка"
        issues.append(f"семинарский аналог: ожидался {expected}")

    discovery_path = response.get("discovery_path", [])
    discovery_text = "\n".join(
        " ".join(str(item.get(field, "")) for field in ("clue", "action", "result", "next_step"))
        for item in discovery_path
        if isinstance(item, dict)
    )
    missed_signals = [
        pattern for pattern in case.signal_patterns if not _matches(pattern, discovery_text)
    ]
    complete_steps = [
        item
        for item in discovery_path
        if isinstance(item, dict)
        and all(str(item.get(field, "")).strip() for field in ("clue", "action", "result", "next_step"))
    ]
    if len(complete_steps) >= 2 and not missed_signals:
        components["путь к решению"] = 1
    else:
        if missed_signals:
            issues.append("путь не объясняет признаки: " + ", ".join(missed_signals))
        if len(complete_steps) < 2:
            issues.append("нет двух полных связанных шагов пути к решению")

    micro_text = "\n".join(map(str, response.get("micro_methods", ())))
    matched_micro = sum(_matches(pattern, micro_text) for pattern in case.micro_method_patterns)
    if matched_micro == len(case.micro_method_patterns):
        components["малые приёмы"] = 2
    elif matched_micro:
        components["малые приёмы"] = 1
        issues.append(f"малые приёмы покрыты частично: {matched_micro}/{len(case.micro_method_patterns)}")
    else:
        issues.append("малые приёмы не совпали с задачей")

    method = str(response.get("chosen_method", ""))
    missed_method = [
        pattern for pattern in case.method_patterns if not _matches(pattern, method)
    ]
    if method.strip() and not missed_method:
        components["метод"] = 1
    else:
        issues.append("основной метод не покрыл: " + ", ".join(missed_method))

    calls = response.get("oracle_calls", [])
    used = {
        _module_name(call.get("module", ""))
        for call in calls
        if isinstance(call, dict)
    }
    allowed = {_module_name(module) for module in case.oracle_modules}
    if used & allowed:
        components["выбор оракула"] = 1
    else:
        issues.append("не выбран ожидаемый оракул: " + ", ".join(case.oracle_modules))
    if agent_eval._observed_oracle(observed_commands):
        components["запуск оракула"] = 1
    else:
        issues.append("в трассе нет фактического запуска Python-оракула")
    if command_budget is not None and len(observed_commands) > command_budget:
        issues.append(f"превышен бюджет команд: {len(observed_commands)} > {command_budget}")

    answer_value = response.get("final_answer", "")
    answer = answer_value if isinstance(answer_value, str) else ""
    matched_answer = sum(_matches(pattern, answer) for pattern in case.answer_patterns)
    if len(answer.strip()) >= 100 and matched_answer == len(case.answer_patterns):
        components["ответ"] = 2
    elif len(answer.strip()) >= 60 and matched_answer:
        components["ответ"] = 1
        issues.append(f"ответ покрыл {matched_answer}/{len(case.answer_patterns)} опор")
    else:
        issues.append(f"ответ покрыл {matched_answer}/{len(case.answer_patterns)} опор")

    limitations = [
        str(item).strip() for item in response.get("limitations", ()) if str(item).strip()
    ]
    oracle_text = "\n".join(
        " ".join(
            str(call.get(field, ""))
            for field in ("module", "operation", "input", "result")
        )
        for call in calls
        if isinstance(call, dict)
    )
    evaluated_text = "\n".join(
        (discovery_text, micro_text, method, oracle_text, answer, *limitations)
    )
    forbidden = [
        pattern for pattern in case.forbidden_patterns if _matches(pattern, evaluated_text)
    ]
    if not forbidden and limitations:
        components["границы"] = 1
    if forbidden:
        issues.append("недопустимое утверждение: " + ", ".join(forbidden))
    if not limitations:
        issues.append("не указана граница машинной проверки")

    return PedagogyScore(
        case.id,
        components,
        tuple(issues),
        runner,
        valid_response=not validation_issues,
        infrastructure_error=infrastructure_error,
    )


def run_cases(
    cases: Sequence[PedagogyCase],
    output: pathlib.Path,
    *,
    runner: str = "codex",
    executable: str | None = None,
    model: str | None = None,
    timeout: int = 900,
) -> tuple[dict[str, Any], ...]:
    if runner not in {"codex", "claude", "opencode"}:
        raise ValueError(f"неизвестный runner: {runner}")
    executable = executable or runner
    version = agent_eval._runner_version(executable)
    invoke = {
        "codex": agent_eval.run_codex_prompt,
        "claude": agent_eval.run_claude_prompt,
        "opencode": agent_eval.run_opencode_prompt,
    }[runner]
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            record = invoke(
                case_id=case.id,
                prompt=build_prompt(case),
                schema_path=SCHEMA_PATH,
                executable=executable,
                model=model,
                timeout=timeout,
                version=version,
                command_budget=COMMAND_BUDGET,
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
    return tuple(records)


def load_runs(path: pathlib.Path) -> tuple[dict[str, Any], ...]:
    return tuple(_json_lines(path))


def score_runs(
    cases: Sequence[PedagogyCase], records: Sequence[dict[str, Any]]
) -> tuple[PedagogyScore, ...]:
    by_id = {case.id: case for case in cases}
    seen: set[str] = set()
    scores: list[PedagogyScore] = []
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
                command_budget=int(record.get("command_budget", COMMAND_BUDGET)),
            )
        )
    return tuple(scores)


def score_report(
    scores: Sequence[PedagogyScore], records: Sequence[dict[str, Any]] = ()
) -> str:
    by_id = {str(record.get("case_id", "")): record for record in records}
    lines = ["# Педагогический holdout", ""]
    for score in scores:
        if score.infrastructure_error:
            status = "INFRA ERROR"
        elif score.passed:
            status = "PASS"
        elif score.content_passed:
            status = "CONTENT PASS / BUDGET FAIL"
        else:
            status = "FAIL"
        record = by_id.get(score.case_id, {})
        meta = []
        if record.get("model"):
            meta.append(f"model={record['model']}")
        if record.get("elapsed_s") is not None:
            meta.append(f"elapsed={record['elapsed_s']}s")
        suffix = f" ({', '.join(meta)})" if meta else ""
        lines.append(f"## {score.case_id}: {status}, {score.total}/13{suffix}")
        lines.append("")
        lines.append(
            ", ".join(f"{name}={value}" for name, value in score.components.items())
        )
        if score.issues:
            lines.append("")
            lines.extend(f"- {issue}" for issue in score.issues)
        lines.append("")
    passed = sum(score.passed for score in scores)
    content_passed = sum(score.content_passed for score in scores)
    infrastructure_errors = sum(score.infrastructure_error for score in scores)
    lines.append(
        f"Итого: {passed}/{len(scores)} PASS; "
        f"содержательно {content_passed}/{len(scores)}; "
        f"инфраструктурных ошибок {infrastructure_errors}"
    )
    return "\n".join(lines)
