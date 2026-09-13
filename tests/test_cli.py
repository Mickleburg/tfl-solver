from __future__ import annotations

from tfl.cli import main
from tfl.paths import ROOT


def test_root_prints_runtime_path(capsys):
    assert main(["root"]) == 0
    assert capsys.readouterr().out.strip() == str(ROOT)


def test_setup_installs_and_checks_global_adapters(tmp_path, capsys):
    assert main(["setup", "--home", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "/tfl <запрос>" in output
    assert "$tfl-solver <запрос>" in output
    assert main(["setup", "--home", str(tmp_path), "--status"]) == 0
    assert "current" in capsys.readouterr().out


def test_doctor_is_independent_of_current_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "OK   общий skill" in output
    assert "OK   манифест Codex" in output
    assert "OK   манифест Claude" in output
    assert "OK   индекс корпуса" in output


def test_intake_accepts_whole_task_without_variant_number(capsys):
    task = "Дана SRS с правилами aa -> aba и ba -> a. Проверить завершимость."
    assert main(["intake", "--text", task, "--limit", "1"]) == 0
    output = capsys.readouterr().out
    assert "правила SRS" in output
    assert "заверш" in output


def test_intake_rejects_empty_input(capsys):
    assert main(["intake", "--text", "   "]) == 2
    assert "пустой текст задачи" in capsys.readouterr().err


def test_lab_cli_creates_scaffold_and_reports_placeholders(tmp_path, capsys):
    project = tmp_path / "lab"
    assert main(["lab", "init", str(project), "--text", "Построить ДКА."]) == 0
    assert (project / "solution.py").is_file()
    assert main(["lab", "check", str(project), "--skip-tests"]) == 1
    output = capsys.readouterr().out
    assert "автономная заготовка" in output
    assert "LAB_TEMPLATE_TODO" in output
