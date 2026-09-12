from __future__ import annotations

from tfl.lab_project import PLACEHOLDER, audit_lab_project, create_lab_scaffold


def test_scaffold_is_self_contained_and_never_overwrites(tmp_path):
    root = tmp_path / "lab"
    created = create_lab_scaffold(root, title="ЛР", task_text="Построить ДКА.")
    assert {path.relative_to(root).as_posix() for path in created} >= {
        "README.md",
        "task.md",
        "solution.py",
        "tests/test_solution.py",
        "report.md",
    }
    assert "Построить ДКА." in (root / "task.md").read_text(encoding="utf-8")
    assert not list(root.glob("requirements*.txt"))
    audit = audit_lab_project(root, run_tests=False)
    assert not audit.passed
    assert any(PLACEHOLDER in issue for issue in audit.issues)

    try:
        create_lab_scaffold(root)
    except FileExistsError:
        pass
    else:
        raise AssertionError("заполненный каталог нельзя перезаписывать")


def test_audit_accepts_stdlib_only_project_and_runs_tests(tmp_path):
    root = tmp_path / "ready"
    (root / "tests").mkdir(parents=True)
    (root / "README.md").write_text("# Ready\n", encoding="utf-8")
    (root / "task.md").write_text("# Условие\n\nУдвоить строку.\n", encoding="utf-8")
    (root / "report.md").write_text("# Описание\n\nПроверено unittest.\n", encoding="utf-8")
    (root / "solution.py").write_text(
        "from __future__ import annotations\n\n"
        "def solve(text: str) -> str:\n"
        "    return text * 2\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_solution.py").write_text(
        "import unittest\n\n"
        "from solution import solve\n\n"
        "class Tests(unittest.TestCase):\n"
        "    def test_solve(self):\n"
        "        self.assertEqual(solve('a'), 'aa')\n",
        encoding="utf-8",
    )

    audit = audit_lab_project(root)
    assert audit.passed, audit.issues
    assert audit.tests_run
    assert audit.test_returncode == 0


def test_audit_rejects_external_import_and_dependency_file(tmp_path):
    root = tmp_path / "external"
    create_lab_scaffold(root, task_text="Условие")
    (root / "solution.py").write_text("import requests\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests\n", encoding="utf-8")

    issues = audit_lab_project(root, run_tests=False).issues
    assert any("внешний импорт `requests`" in issue for issue in issues)
    assert any("requirements.txt" in issue for issue in issues)

    (root / "requirements.txt").unlink()
    (root / "vendor").mkdir()
    (root / "vendor" / "requirements-dev.txt").write_text("rich\n", encoding="utf-8")
    issues = audit_lab_project(root, run_tests=False).issues
    assert any("vendor" in issue and "requirements-dev.txt" in issue for issue in issues)


def test_audit_rejects_skipped_acceptance_tests(tmp_path):
    root = tmp_path / "skipped"
    create_lab_scaffold(root, task_text="Условие")
    audit = audit_lab_project(root)
    assert any("пропущенные тесты" in issue for issue in audit.issues)
