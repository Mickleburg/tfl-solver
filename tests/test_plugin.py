from __future__ import annotations

import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_plugin_manifests_share_identity_and_license():
    portable = _json("plugin.json")
    codex = _json(".codex-plugin/plugin.json")
    claude = _json(".claude-plugin/plugin.json")

    assert portable["$schema"] == (
        "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    )
    for manifest in (portable, codex, claude):
        assert manifest["name"] == "tfl-solver"
        assert manifest["version"] == "1.0.0"
        assert manifest["license"] == "Apache-2.0"
        assert manifest["repository"] == "https://github.com/Mickleburg/tfl-solver"


def test_marketplaces_publish_the_same_plugin():
    codex = _json(".agents/plugins/marketplace.json")
    claude = _json(".claude-plugin/marketplace.json")

    assert codex["name"] == claude["name"] == "tfl-solver-marketplace"
    assert codex["plugins"][0]["name"] == "tfl-solver"
    assert claude["plugins"][0]["name"] == "tfl-solver"
    expected_source = {
        "source": "url",
        "url": "https://github.com/Mickleburg/tfl-solver.git",
        "ref": "main",
    }
    assert codex["plugins"][0]["source"] == expected_source
    assert claude["plugins"][0]["source"] == expected_source


def test_canonical_skill_is_portable_and_adapters_are_thin():
    skill = (ROOT / "skills" / "tfl" / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: tfl\n")
    assert "C:\\Users\\" not in skill
    assert "scripts/tfl_plugin.py" in skill

    for relative in (
        ".agents/skills/tfl-solver/SKILL.md",
        ".claude/skills/tfl/SKILL.md",
    ):
        adapter = (ROOT / relative).read_text(encoding="utf-8")
        assert "skills/tfl/SKILL.md" in adapter


def test_plugin_launcher_works_outside_checkout(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tfl_plugin.py"), "root"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert pathlib.Path(completed.stdout.strip()).resolve() == ROOT.resolve()


def test_plugin_launcher_exposes_stable_srs_command_outside_checkout(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "tfl_plugin.py"),
            "srs",
            "critical-pairs",
            "--rule",
            "aab -> ba",
            "--rule",
            "aaa -> ab",
            "--format",
            "json",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        encoding="ascii",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["count"] == 4
    assert payload["critical_pairs"][0]["word"] == "aaaab"
