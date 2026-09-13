"""Runtime paths for a source checkout and an installed wheel."""

from __future__ import annotations

import pathlib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
BUNDLED_ROOT = PACKAGE_ROOT / "_runtime"


def _is_runtime_root(path: pathlib.Path) -> bool:
    """Return whether *path* contains the versioned agent runtime."""
    return (
        (path / "docs" / "02-TASK-TAXONOMY.md").is_file()
        and (path / "skills" / "tfl" / "SKILL.md").is_file()
        and (path / "evals" / "tasks" / "index.jsonl").is_file()
    )


if _is_runtime_root(SOURCE_ROOT):
    ROOT = SOURCE_ROOT
elif _is_runtime_root(BUNDLED_ROOT):
    ROOT = BUNDLED_ROOT
else:
    # Keep failures inspectable: ``doctor`` will report the missing assets.
    ROOT = SOURCE_ROOT


IS_BUNDLED = ROOT == BUNDLED_ROOT
