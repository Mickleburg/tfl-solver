"""Самодостаточные заготовки и аудит проектов лабораторных ТФЯ.

`tfl-solver` можно использовать при разработке как оракул, но сдаваемый
проект не должен импортировать его или другие сторонние пакеты. Этот модуль
создаёт нейтральную структуру и проверяет границу поставки статически и
запуском тестов стандартной библиотекой.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass


PLACEHOLDER = "LAB_TEMPLATE_TODO"
REQUIRED_FILES = (
    "README.md",
    "task.md",
    "solution.py",
    "tests/test_solution.py",
    "report.md",
)
DEPENDENCY_FILES = ("Pipfile", "Pipfile.lock", "poetry.lock", "uv.lock")
DEPENDENCY_DIRECTORIES = (".venv", "venv", "site-packages", "node_modules")


@dataclass(frozen=True)
class LabProjectAudit:
    """Результат проверки автономности и приёмочной готовности проекта."""

    root: pathlib.Path
    python_files: int
    issues: tuple[str, ...]
    tests_run: bool
    test_returncode: int | None = None
    test_output: str = ""

    @property
    def passed(self) -> bool:
        return not self.issues


def _one_line(value: str, fallback: str) -> str:
    cleaned = " ".join(value.split()).strip()
    return cleaned or fallback


def create_lab_scaffold(
    destination: str | pathlib.Path,
    *,
    title: str = "Лабораторная работа по ТФЯ",
    task_text: str | None = None,
) -> tuple[pathlib.Path, ...]:
    """Создать новую заготовку, не перезаписывая существующие файлы."""

    root = pathlib.Path(destination)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"каталог не пуст: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir()

    heading = _one_line(title, "Лабораторная работа по ТФЯ")
    condition = (task_text or "").strip()
    if not condition:
        condition = f"{PLACEHOLDER}: вставить полное неизменённое условие."

    files = {
        "README.md": f"""# {heading}

Самодостаточный проект лабораторной по ТФЯ. Для запуска нужен только
Python 3.11+; сеть, `tfl-solver` и сторонние пакеты не требуются.

## Запуск

```text
python solution.py < input.txt
python -S -m unittest discover -s tests -v
```

## Состав

- `task.md` — исходное условие без пересказа;
- `solution.py` — реализация алгоритма;
- `tests/` — проверка примеров, границ и контрпримеров;
- `report.md` — формализация, алгоритм, доказательство и описание ИИ.

## Готовность

{PLACEHOLDER}: заменить заготовку предметной реализацией и командами запуска,
затем выполнить `tfl-agent lab check .` из окружения tfl-solver.
""",
        "task.md": f"""# Условие

{condition}
""",
        "solution.py": f'''"""Самодостаточное решение: заменить заготовку реализацией из условия."""

from __future__ import annotations

import sys


def solve(source: str) -> str:
    """Решить один экземпляр в документированном текстовом формате."""
    _ = source
    raise NotImplementedError("{PLACEHOLDER}: реализовать алгоритм")


def main() -> int:
    try:
        answer = solve(sys.stdin.read())
    except (ValueError, NotImplementedError) as error:
        print(f"ошибка: {{error}}", file=sys.stderr)
        return 2
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        "tests/test_solution.py": f'''from __future__ import annotations

import unittest

from solution import solve


class SolutionTests(unittest.TestCase):
    @unittest.skip("{PLACEHOLDER}: заменить приёмочными примерами")
    def test_acceptance_example(self):
        self.assertEqual(solve("пример"), "ожидаемый ответ")


if __name__ == "__main__":
    unittest.main()
''',
        "report.md": f"""# Описание лабораторной

## Формализация

{PLACEHOLDER}: определить вход, выход, алфавиты, соглашения и ограничения.

## Алгоритм

{PLACEHOLDER}: описать шаги и выбранные структуры данных.

## Корректность

{PLACEHOLDER}: сформулировать инвариант и доказать завершимость/правильность.

## Тестирование

{PLACEHOLDER}: перечислить примеры, граничные случаи и машинные свидетельства.

## Использование ИИ

- Инструмент и модель: `{PLACEHOLDER}`
- Как использовался: `{PLACEHOLDER}`
- Фактические пользовательские промпты: `{PLACEHOLDER}`
- Что было исправлено или отвергнуто: `{PLACEHOLDER}`
- Независимая проверка: `{PLACEHOLDER}`

## Ограничения

{PLACEHOLDER}: честно перечислить непокрытые случаи либо написать «нет».
""",
        ".gitignore": "__pycache__/\n*.py[cod]\n",
    }

    created: list[pathlib.Path] = []
    for relative, content in files.items():
        path = root / relative
        path.write_text(content, encoding="utf-8", newline="\n")
        created.append(path)
    return tuple(created)


def _local_modules(root: pathlib.Path) -> set[str]:
    modules = {path.stem for path in root.glob("*.py")}
    modules.update(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    return modules


def _import_issues(root: pathlib.Path, python_files: list[pathlib.Path]) -> list[str]:
    allowed = set(sys.stdlib_module_names) | {"__future__"} | _local_modules(root)
    issues: list[str] = []
    for path in python_files:
        relative = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, UnicodeError, SyntaxError) as error:
            issues.append(f"{relative}: Python-файл не разбирается: {error}")
            continue
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module.split(".", 1)[0])
        for module in sorted(set(imported) - allowed):
            issues.append(f"{relative}: внешний импорт `{module}`")
    return issues


def _dependency_issues(root: pathlib.Path) -> list[str]:
    issues: list[str] = []
    for path in root.rglob("requirements*.txt"):
        issues.append(f"внешний файл зависимостей `{path.relative_to(root)}`")
    for name in DEPENDENCY_FILES:
        for path in root.rglob(name):
            issues.append(f"внешний файл зависимостей `{path.relative_to(root)}`")
    for path in root.rglob("*"):
        if path.is_dir() and path.name in DEPENDENCY_DIRECTORIES:
            issues.append(f"каталог внешнего окружения `{path.relative_to(root)}`")

    for pyproject in root.rglob("pyproject.toml"):
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            issues.append(f"{pyproject.relative_to(root)} не разбирается: {error}")
        else:
            project = data.get("project", {})
            if project.get("dependencies"):
                issues.append(
                    f"{pyproject.relative_to(root)} содержит project.dependencies"
                )
            if project.get("optional-dependencies"):
                issues.append(
                    f"{pyproject.relative_to(root)} содержит optional-dependencies"
                )
            if data.get("build-system", {}).get("requires"):
                issues.append(
                    f"{pyproject.relative_to(root)} требует внешнюю систему сборки"
                )
    return issues


def audit_lab_project(
    project: str | pathlib.Path,
    *,
    run_tests: bool = True,
    timeout: int = 120,
) -> LabProjectAudit:
    """Проверить структуру, импорты, заглушки и stdlib-тесты проекта."""

    root = pathlib.Path(project)
    issues: list[str] = []
    if not root.is_dir():
        return LabProjectAudit(root, 0, (f"каталог не найден: {root}",), False)

    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            issues.append(f"нет обязательного файла `{relative}`")

    text_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py"}
    )
    for path in text_files:
        try:
            if PLACEHOLDER in path.read_text(encoding="utf-8"):
                issues.append(f"{path.relative_to(root)}: осталась заглушка {PLACEHOLDER}")
        except (OSError, UnicodeError) as error:
            issues.append(f"{path.relative_to(root)}: не удалось прочитать: {error}")

    python_files = sorted(root.rglob("*.py"))
    issues.extend(_import_issues(root, python_files))
    issues.extend(_dependency_issues(root))

    test_returncode: int | None = None
    test_output = ""
    tests_run = run_tests and (root / "tests").is_dir()
    if tests_run:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tests",
                    "-v",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            issues.append(f"тесты превысили таймаут {timeout} с")
        else:
            test_returncode = completed.returncode
            test_output = (completed.stdout + completed.stderr).strip()
            if completed.returncode:
                issues.append(f"stdlib-тесты завершились с кодом {completed.returncode}")
            match = re.search(r"Ran\s+(\d+)\s+tests?", test_output)
            if match is None or int(match.group(1)) == 0:
                issues.append("stdlib-набор не выполнил ни одного теста")
            if re.search(r"skipped=\d+", test_output):
                issues.append("stdlib-набор содержит пропущенные тесты")

    return LabProjectAudit(
        root=root,
        python_files=len(python_files),
        issues=tuple(issues),
        tests_run=tests_run,
        test_returncode=test_returncode,
        test_output=test_output,
    )
