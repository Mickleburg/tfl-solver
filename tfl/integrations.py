"""Install thin global adapters for supported LLM command-line clients."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

from tfl.paths import ROOT


MANAGED_MARKER = "<!-- managed-by: tfl-solver -->"
TARGETS = ("claude", "codex")


@dataclass(frozen=True)
class Adapter:
    client: str
    kind: str
    path: pathlib.Path
    content: str


@dataclass(frozen=True)
class InstallResult:
    adapter: Adapter
    status: str


def _markdown_path(path: pathlib.Path) -> str:
    return path.resolve().as_posix()


def adapters(home: pathlib.Path | None = None) -> tuple[Adapter, ...]:
    """Describe the user-level adapters without changing the filesystem."""
    explicit_home = home is not None
    user_home = (home or pathlib.Path.home()).expanduser().resolve()
    codex_home = (
        user_home / ".codex"
        if explicit_home
        else pathlib.Path(os.environ.get("CODEX_HOME", user_home / ".codex"))
        .expanduser()
        .resolve()
    )
    runtime = _markdown_path(ROOT)
    canonical = _markdown_path(ROOT / "skills" / "tfl" / "SKILL.md")

    claude = f"""---
name: tfl
description: Решение задач, объяснение теории и автономные проекты лабораторных по ТФЯ
---

{MANAGED_MARKER}

# TFL Solver

Запрос пользователя: $ARGUMENTS

Рабочие материалы установлены в `{runtime}`. Полностью прочитай канонические
инструкции `{canonical}`, затем выполни запрос по ним. Все относительные пути
из канонических инструкций разрешай от `{runtime}`.
"""
    codex_skill = f"""---
name: tfl-solver
description: "Teach and solve theory of formal languages: explain theory, verify tasks, build standalone dependency-free lab projects and reports, and audit proposed solutions."
---

{MANAGED_MARKER}

# TFL Solver

Рабочие материалы установлены в `{runtime}`. Полностью прочитай канонические
инструкции `{canonical}`, затем выполни запрос пользователя по ним. Все
относительные пути из канонических инструкций разрешай от `{runtime}`.
"""
    codex_prompt = f"""{MANAGED_MARKER}
Используй навык `$tfl-solver` и выполни следующий запрос по ТФЯ:

$ARGUMENTS
"""
    return (
        Adapter(
            "claude",
            "skill /tfl",
            user_home / ".claude" / "skills" / "tfl" / "SKILL.md",
            claude,
        ),
        Adapter(
            "codex",
            "skill $tfl-solver",
            user_home / ".agents" / "skills" / "tfl-solver" / "SKILL.md",
            codex_skill,
        ),
        Adapter(
            "codex",
            "prompt /prompts:tfl",
            codex_home / "prompts" / "tfl.md",
            codex_prompt,
        ),
    )


def selected_adapters(
    target: str, home: pathlib.Path | None = None
) -> tuple[Adapter, ...]:
    available = adapters(home)
    if target == "all":
        return available
    if target not in TARGETS:
        raise ValueError(f"неизвестная интеграция: {target}")
    return tuple(adapter for adapter in available if adapter.client == target)


def adapter_status(adapter: Adapter) -> str:
    if not adapter.path.exists():
        return "missing"
    if not adapter.path.is_file():
        return "unreadable"
    try:
        current = adapter.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "unreadable"
    if current == adapter.content:
        return "current"
    if MANAGED_MARKER in current:
        return "outdated"
    return "conflict"


def install_adapters(
    target: str = "all",
    *,
    home: pathlib.Path | None = None,
    force: bool = False,
) -> tuple[InstallResult, ...]:
    """Install adapters, preserving unrelated user files unless forced."""
    results: list[InstallResult] = []
    for adapter in selected_adapters(target, home):
        before = adapter_status(adapter)
        if before == "current":
            results.append(InstallResult(adapter, "current"))
            continue
        if before == "unreadable":
            results.append(InstallResult(adapter, "unreadable"))
            continue
        if before == "conflict" and not force:
            results.append(InstallResult(adapter, "conflict"))
            continue
        adapter.path.parent.mkdir(parents=True, exist_ok=True)
        adapter.path.write_text(adapter.content, encoding="utf-8", newline="\n")
        action = "installed" if before == "missing" else "updated"
        results.append(InstallResult(adapter, action))
    return tuple(results)
