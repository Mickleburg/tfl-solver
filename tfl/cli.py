"""Переносимая командная строка для детерминированной части агента ТФЯ."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from collections.abc import Sequence

from tfl.intake import BY_FORM, analyse, load_index


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _configure_windows_stdio() -> None:
    """Печатать UTF-8 и в терминал, и в pipe независимо от системной cp1251."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _subprocess_env() -> dict[str, str]:
    """Передать дочерним Python-процессам переносимую кодировку вывода."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _task_text(arguments: argparse.Namespace) -> str:
    if arguments.text is not None:
        text = arguments.text
    elif arguments.file is not None:
        text = pathlib.Path(arguments.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("пустой текст задачи")
    return text


def _intake(arguments: argparse.Namespace) -> int:
    try:
        text = _task_text(arguments)
    except (OSError, ValueError) as error:
        print(f"ошибка входа: {error}", file=sys.stderr)
        return 2
    print(analyse(text, arguments.hint, load_index(), arguments.limit).report())
    return 0


def _doctor(arguments: argparse.Namespace) -> int:
    checks = {
        "Python >= 3.11": sys.version_info >= (3, 11),
        "пакет tfl": (ROOT / "tfl" / "__init__.py").is_file(),
        "Codex skill": (ROOT / ".agents" / "skills" / "tfl-solver" / "SKILL.md").is_file(),
        "Claude adapter": (ROOT / ".claude" / "skills" / "tfl" / "SKILL.md").is_file(),
        "карта задач": (ROOT / "docs" / "02-TASK-TAXONOMY.md").is_file(),
        "индекс корпуса": (ROOT / "evals" / "tasks" / "index.jsonl").is_file(),
        "сквозной holdout": (
            (ROOT / "evals" / "agent_holdout" / "cases.jsonl").is_file()
            and (ROOT / "evals" / "agent_holdout" / "response.schema.json").is_file()
        ),
    }
    optional = {
        "первоисточник преподавателя": ROOT
        / "references"
        / "teacher"
        / "FormalLanguageTheory",
        "учебные материалы": ROOT / "references" / "course" / "TFL-IU9-claude",
        "архив чата": ROOT / "references" / "chat" / "chat-onTG-2025-tfl",
    }

    print(f"tfl-solver: {ROOT}")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    for name, passed in checks.items():
        print(f"{'OK' if passed else 'FAIL':4} {name}")
    for name, path in optional.items():
        print(f"{'OK' if path.exists() else 'WARN':4} {name}: {path.relative_to(ROOT)}")

    if arguments.tests:
        print("\nЗапуск pytest...")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            check=False,
            env=_subprocess_env(),
        )
        if completed.returncode:
            return completed.returncode
    return 0 if all(checks.values()) else 1


def _eval(arguments: argparse.Namespace) -> int:
    command = [sys.executable, str(ROOT / "tools" / "eval_suite.py")]
    if arguments.selected:
        command += ["--class", arguments.selected]
    if arguments.report is not None:
        command += ["--report"]
        if arguments.report:
            command.append(arguments.report)
    return subprocess.run(
        command, cwd=ROOT, check=False, env=_subprocess_env()
    ).returncode


def _repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else ROOT / path


def _holdout(arguments: argparse.Namespace) -> int:
    from tfl.agent_eval import (
        build_prompt,
        choose_cases,
        load_cases,
        load_runs,
        run_codex,
        score_report,
        score_runs,
    )

    cases = load_cases()
    if arguments.holdout_command == "list":
        for case in cases:
            head = case.statement.replace("\n", " ")[:90]
            print(f"{case.id:24} {case.year} {case.hint:8} {head}")
        return 0
    if arguments.holdout_command == "prompt":
        print(build_prompt(choose_cases(cases, [arguments.case])[0]))
        return 0
    if arguments.holdout_command == "run":
        if not arguments.all and not arguments.case:
            print("укажите --case ID (можно несколько раз) или --all", file=sys.stderr)
            return 2
        selected = choose_cases(cases, () if arguments.all else arguments.case)
        output = _repo_path(arguments.output)
        if output.exists():
            print(f"файл уже существует: {output}", file=sys.stderr)
            return 2
        print(f"Запуск {len(selected)} случаев через Codex; результат: {output}")
        records = run_codex(
            selected,
            output,
            executable=arguments.executable,
            model=arguments.model,
            timeout=arguments.timeout,
        )
        scores = score_runs(cases, records)
        print(score_report(scores, records))
        return 0 if all(score.passed for score in scores) else 1
    if arguments.holdout_command == "score":
        records = load_runs(_repo_path(arguments.input))
        scores = score_runs(cases, records)
        report = score_report(scores, records)
        print(report)
        if arguments.report:
            destination = _repo_path(arguments.report)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(report, encoding="utf-8")
            print(f"\nотчёт записан: {destination}")
        return 0 if scores and all(score.passed for score in scores) else 1
    raise AssertionError(arguments.holdout_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfl-agent",
        description=(
            "Детерминированный вход агента ТФЯ: диагностика, разбор условия "
            "и прогон исполняемых eval. Сам цикл рассуждений задаёт repo-skill."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="проверить готовность проекта")
    doctor.add_argument("--tests", action="store_true", help="также запустить pytest")
    doctor.set_defaults(handler=_doctor)

    intake = subparsers.add_parser("intake", help="разобрать полное условие задачи")
    source = intake.add_mutually_exclusive_group()
    source.add_argument("--text", help="текст задачи в командной строке")
    source.add_argument("--file", help="UTF-8 файл с условием")
    intake.add_argument("--hint", choices=sorted(BY_FORM), help="необязательная форма контроля")
    intake.add_argument("--limit", type=int, default=5, help="число похожих задач")
    intake.set_defaults(handler=_intake)

    evaluate = subparsers.add_parser("eval", help="прогнать исполняемый eval-набор")
    evaluate.add_argument("--class", dest="selected", help="класс или id случая")
    evaluate.add_argument(
        "--report",
        nargs="?",
        const="",
        help="обновить стандартный отчёт или записать указанный файл",
    )
    evaluate.set_defaults(handler=_eval)

    holdout = subparsers.add_parser(
        "holdout", help="сквозной eval маршрута, оракулов и итогового ответа"
    )
    holdout_actions = holdout.add_subparsers(dest="holdout_command", required=True)
    holdout_list = holdout_actions.add_parser("list", help="показать замороженные случаи")
    holdout_list.set_defaults(handler=_holdout)

    holdout_prompt = holdout_actions.add_parser("prompt", help="показать eval-промпт")
    holdout_prompt.add_argument("--case", required=True, help="id случая")
    holdout_prompt.set_defaults(handler=_holdout)

    holdout_run = holdout_actions.add_parser("run", help="запустить случаи через codex exec")
    holdout_selection = holdout_run.add_mutually_exclusive_group(required=False)
    holdout_selection.add_argument("--case", action="append", help="id случая; повторяется")
    holdout_selection.add_argument("--all", action="store_true", help="запустить весь набор")
    holdout_run.add_argument("--output", required=True, help="новый JSONL-файл результата")
    holdout_run.add_argument("--executable", default="codex", help="путь к Codex CLI")
    holdout_run.add_argument("--model", help="необязательная явная модель Codex")
    holdout_run.add_argument("--timeout", type=int, default=900, help="таймаут на случай, с")
    holdout_run.set_defaults(handler=_holdout)

    holdout_score = holdout_actions.add_parser("score", help="оценить сохранённый JSONL-run")
    holdout_score.add_argument("--input", required=True, help="JSONL-файл результата")
    holdout_score.add_argument("--report", help="необязательный Markdown-отчёт")
    holdout_score.set_defaults(handler=_holdout)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_windows_stdio()
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
