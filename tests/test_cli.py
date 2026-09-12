from __future__ import annotations

from tfl.cli import main


def test_doctor_is_independent_of_current_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "OK   Codex skill" in output
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
