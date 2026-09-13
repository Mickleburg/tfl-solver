"""Build hook that embeds the agent knowledge base in distributable wheels."""

from __future__ import annotations

import fnmatch
import pathlib
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


RUNTIME_DIRECTORIES = (".agents", ".claude", "corpus", "docs", "evals", "tools")
RUNTIME_FILES = ("AGENTS.md", "README.md")
IGNORED_NAMES = ("__pycache__", "*.pyc", "*.pyo")


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if any(fnmatch.fnmatch(name, pattern) for pattern in IGNORED_NAMES)
    }


class build_py(_build_py):
    """Copy non-Python knowledge and eval assets next to the installed package."""

    def run(self) -> None:
        super().run()
        source = pathlib.Path(__file__).resolve().parent
        runtime = pathlib.Path(self.build_lib) / "tfl" / "_runtime"
        if runtime.exists():
            shutil.rmtree(runtime)
        runtime.mkdir(parents=True)
        for name in RUNTIME_DIRECTORIES:
            shutil.copytree(source / name, runtime / name, ignore=_ignore)
        for name in RUNTIME_FILES:
            shutil.copy2(source / name, runtime / name)


setup(cmdclass={"build_py": build_py})
