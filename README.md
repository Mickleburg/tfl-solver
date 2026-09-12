# tfl-solver

Агент и набор Python-оракулов для задач по теории формальных языков: РК,
лабораторных, экзамена и бонусных задач курса ИУ9 МГТУ.

Агент получает полное условие текстом или фотографией, сам определяет класс
задачи, подбирает рецепт и проверяет содержательные гипотезы исполняемыми
оракулами. Номер варианта для запуска не требуется.

## Быстрый запуск

Требуется Python 3.11+.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m tfl doctor
```

Для работы со сканами и SMT-поиском можно установить все дополнительные
зависимости:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,corpus,smt]"
```

### Codex

Запусти Codex из корня репозитория и передай задачу с упоминанием
`$tfl-solver`. Repo-skill находится в `.agents/skills/tfl-solver/` и
обнаруживается Codex автоматически. Список доступных навыков можно проверить
командой `/skills`.

### Claude Code

Совместимый вход сохранён в `.claude/skills/tfl/SKILL.md`. Он загружает тот же
канонический repo-skill, поэтому правила двух агентов не расходятся.

### Детерминированная командная строка

```powershell
py -3 -m tfl doctor
py -3 -m tfl intake --file task.txt
py -3 -m tfl intake --text "Проверить завершимость SRS ..." --hint ЛР1
py -3 -m tfl eval
py -3 -m tfl holdout list
```

После editable-установки те же команды доступны через `tfl-agent`.
Командная строка выполняет диагностику, классификацию и eval; рассуждающий
цикл выполняет LLM по repo-skill.

### Сквозная оценка агента

Обычный `eval` проверяет отдельные Python-оракулы. Замороженный holdout
проверяет весь маршрут агента на задачах 2021–2025 и читает реальные события
запуска инструментов Codex CLI или Claude Code:

```powershell
py -3 -m tfl holdout run `
  --runner codex `
  --case pharma-2022-A11 `
  --output reports/agent-holdout/codex-smoke.jsonl

py -3 -m tfl holdout score `
  --input reports/agent-holdout/codex-smoke.jsonl `
  --report reports/agent-holdout/codex-smoke.md
```

`--runner codex` (по умолчанию) запускает случай в read-only sandbox.
`--runner claude` использует non-interactive `stream-json`, отключает сохранение
сессии и оставляет только skill, чтение, поиск и Python-команды. Оба backend'а
требуют установленный и аутентифицированный CLI и запрашивают итог по одной
JSON Schema. Полный набор вызывается только явным `--all`, поскольку это
реальные LLM-запуски. Контракты сверены с официальной документацией
[Codex](https://developers.openai.com/codex/noninteractive) и
[Claude Code](https://code.claude.com/docs/en/headless).

## Устройство

- `.agents/skills/tfl-solver/SKILL.md` — канонический цикл решения;
- `tfl/` — проверяющие оракулы;
- `docs/recipes/` — рецепты 17 классов задач;
- `corpus/` — версионируемое текстовое зеркало источников;
- `evals/` — исполняемая оценка покрытия;
- `evals/agent_holdout/` — замороженные задачи и контракт сквозного eval;
- `docs/PROJECT-STATE.md` — единое актуальное состояние;
- `docs/OPEN-GAPS.md` — незакрытые задачи;
- `references/` — локальные исходные материалы, не входящие в Git.

Материалы преподавателя считаются первоисточником. Студенческие решения и
учебный чат используются только как примеры, гипотезы и тестовые данные.
