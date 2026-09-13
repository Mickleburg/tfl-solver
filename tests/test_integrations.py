from __future__ import annotations

import pathlib

from tfl.integrations import (
    MANAGED_MARKER,
    adapter_status,
    install_adapters,
    selected_adapters,
)
from tfl.paths import ROOT


def test_installs_all_user_adapters_into_explicit_home(tmp_path):
    results = install_adapters(home=tmp_path)

    assert [result.status for result in results] == [
        "installed",
        "installed",
        "installed",
    ]
    expected = {
        tmp_path / ".claude" / "skills" / "tfl" / "SKILL.md",
        tmp_path / ".agents" / "skills" / "tfl-solver" / "SKILL.md",
        tmp_path / ".codex" / "prompts" / "tfl.md",
    }
    assert {result.adapter.path for result in results} == expected
    for result in results:
        content = result.adapter.path.read_text(encoding="utf-8")
        assert MANAGED_MARKER in content
    assert ROOT.as_posix() in results[0].adapter.content
    assert "$ARGUMENTS" in results[0].adapter.content


def test_install_is_idempotent_and_updates_managed_adapter(tmp_path):
    install_adapters("claude", home=tmp_path)
    current = install_adapters("claude", home=tmp_path)
    assert current[0].status == "current"

    path = current[0].adapter.path
    path.write_text(f"{MANAGED_MARKER}\nold", encoding="utf-8")
    updated = install_adapters("claude", home=tmp_path)
    assert updated[0].status == "updated"
    assert adapter_status(updated[0].adapter) == "current"


def test_install_preserves_unmanaged_file_unless_forced(tmp_path):
    adapter = selected_adapters("claude", tmp_path)[0]
    adapter.path.parent.mkdir(parents=True)
    adapter.path.write_text("my existing skill", encoding="utf-8")

    refused = install_adapters("claude", home=tmp_path)
    assert refused[0].status == "conflict"
    assert adapter.path.read_text(encoding="utf-8") == "my existing skill"

    forced = install_adapters("claude", home=tmp_path, force=True)
    assert forced[0].status == "updated"
    assert adapter_status(adapter) == "current"


def test_target_filters_adapters(tmp_path):
    assert len(selected_adapters("claude", tmp_path)) == 1
    assert len(selected_adapters("codex", tmp_path)) == 2
