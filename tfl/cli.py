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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_windows_stdio()
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
