"""Регрессии публичного обезличенного корпуса."""

from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".toml", ".tsv", ".json", ".jsonl", ".txt"}


def _public_text_files():
    excluded_roots = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "references",
        "venv",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in excluded_roots or relative.parts[0].endswith(
            ".egg-info"
        ):
            continue
        # Текстовое зеркало официальных PDF сохраняет библиографию дословно.
        if relative.parts[:2] == ("corpus", "txt"):
            continue
        yield path


def test_private_source_markers_do_not_return_to_public_tree():
    legacy_fragments = (
        "corpus/" + "chat",
        "references/" + "chat",
        "references/" + "students",
        "references/" + "vendor",
        "reports/" + "rk2-2026-photos",
        "reports/" + "vendor-repos",
        "chat_" + "AFA",
        "extract_" + "chat",
    )
    private_identifiers = tuple(
        part.lower()
        for part in (
            "boom" + "haa",
            "dm" + "800",
            "Prrro" + "manssss",
            "VD" + "EN5",
            "Useful" + "Tornado",
        )
    )

    violations: list[str] = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8").replace("\\", "/")
        lowered = text.lower()
        found = [item for item in legacy_fragments if item.lower() in lowered]
        found += [item for item in private_identifiers if item in lowered]
        if found:
            violations.append(f"{path.relative_to(ROOT)}: {', '.join(found)}")

    assert not violations, "обнаружены следы частных источников:\n" + "\n".join(
        violations
    )
