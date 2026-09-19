# tfl-solver

Агент и набор Python-оракулов для задач по теории формальных языков: РК,
лабораторных и экзамена курса ИУ9 МГТУ.

Агент получает полное условие текстом или фотографией, сам определяет класс
задачи, подбирает рецепт и проверяет содержательные гипотезы исполняемыми
оракулами. Номер варианта для запуска не требуется.

Основные сценарии — быстрые объяснения любой теории ТФЯ, проверяемые решения
задач, предварительная подготовка к контролю и создание законченных проектов
лабораторных. На РК и экзамене агент использоваться не предполагается: он
помогает заранее научиться решать и объяснять самостоятельно.

Форма контроля 2026 года описана в [`docs/COURSE-2026.md`](docs/COURSE-2026.md):
четыре лабораторные с обязательным раскрытием использования ИИ, карточки РК
под таймер и экзаменационный поиск ошибки в готовом ИИ-решении. Big Pharma в
текущем курсе нет; её старые задачи сохранены только как регрессии.

## Установка плагина из GitHub

Репозиторий одновременно является плагином для Codex и Claude Code. Отдельно
копировать skill-файлы или запускать `tfl setup` не требуется.

Codex CLI:

```text
codex plugin marketplace add Mickleburg/tfl-solver
codex plugin add tfl-solver@tfl-solver-marketplace
```

В новой сессии используй `$tfl <запрос>` или сформулируй задачу обычным
языком: skill допускает автоматический выбор.

Claude Code:

```text
/plugin marketplace add Mickleburg/tfl-solver
/plugin install tfl-solver@tfl-solver-marketplace
```

После перезапуска вызови `/tfl-solver:tfl <запрос>`. Claude Code добавляет к
командам плагина пространство имён, поэтому это имя намеренно длиннее `/tfl`.

Для объяснений и работы с текстовым корпусом достаточно установленного
плагина. Исполняемые оракулы и создание проектов ЛР требуют Python 3.11+;
в каталоге плагина агент запускает их через `scripts/tfl_plugin.py`. Полная
инструкция, обновление, локальная установка и перенос CLI — в
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

Для разработки самого проекта:

```powershell
git clone https://github.com/Mickleburg/tfl-solver.git
cd tfl-solver
py -3 -m pip install -e ".[dev]"
```

Старый вариант с `pip` и `tfl setup` сохранён: он устанавливает самостоятельную
CLI, ненеймспейсную команду Claude Code `/tfl`, Codex skill `$tfl-solver` и
prompt `/prompts:tfl`.

### Детерминированная командная строка

```powershell
tfl doctor
tfl intake --file task.txt
tfl intake --text "Проверить завершимость SRS ..." --hint ЛР
tfl srs critical-pairs --rule "aab -> ba" --rule "aaa -> ab"
tfl srs critical-pairs --file system.srs --format json
tfl lab init path/to/lab --file task.txt
tfl lab check path/to/lab
tfl eval
tfl holdout list
```

`lab init` создаёт неперезаписываемую заготовку с условием, кодом, `unittest`
и отчётом. `lab check` не принимает оставшиеся заглушки, сторонние импорты,
файлы зависимостей или упавшие/пропущенные тесты. Готовый проект использует
только Python 3.11+ и стандартную библиотеку; `tfl-solver` остаётся оракулом
разработки, а не зависимостью сдаваемого решения. Полный контракт —
[`docs/LAB-PROJECT-STANDARD.md`](docs/LAB-PROJECT-STANDARD.md).

Для коротких вопросов действует отдельный
[`docs/THEORY-ANSWER-STANDARD.md`](docs/THEORY-ANSWER-STANDARD.md): прямой
ответ, точное определение, минимальный пример и граница применимости без
ненужного запуска всего корпуса.

Если `tfl` не найден в `PATH`, из клона или распакованного плагина используй
`py -3 scripts/tfl_plugin.py <команда>`; из установленного Python-пакета —
`py -3 -m tfl <команда>`. Старое имя `tfl-agent` сохранено для совместимости.
Командная строка выполняет диагностику, классификацию и eval; рассуждающий
цикл выполняет LLM по repo-skill.

### Сквозная оценка агента

Обычный `eval` проверяет отдельные Python-оракулы. Замороженный holdout
проверяет весь маршрут агента на задачах 2021–2025 и читает реальные события
запуска инструментов Codex CLI или Claude Code:

```powershell
py -3 -m tfl holdout run `
  --runner codex `
  --case exam-2024-b15-q1 `
  --output reports/agent-holdout/codex-exam-smoke.jsonl

py -3 -m tfl holdout score `
  --input reports/agent-holdout/codex-exam-smoke.jsonl `
  --report reports/agent-holdout/codex-exam-smoke.md
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

- `skills/tfl/SKILL.md` — единый канонический цикл решения для обоих клиентов;
- `plugin.json`, `.codex-plugin/`, `.claude-plugin/` — переносимые манифесты;
- `.agents/plugins/marketplace.json` — каталог установки Codex;
- `tfl/` — проверяющие оракулы;
- `docs/recipes/` — рецепты 18 содержательных классов задач;
- `corpus/txt/` — версионируемое текстовое зеркало официальных источников;
- `corpus/knowledge/` — обезличенные вопросы, методы и типовые ошибки;
- `evals/` — исполняемая оценка покрытия;
- `evals/agent_holdout/` — замороженные задачи и контракт сквозного eval;
- `docs/COURSE-2026.md` — актуальная форма контроля и неизвестные детали;
- `docs/LAB-PROJECT-STANDARD.md` — автономный код, тесты и описание ЛР;
- `docs/LAB-AI-DISCLOSURE.md` — обязательный раздел лабораторного отчёта;
- `docs/THEORY-ANSWER-STANDARD.md` — быстрые и проверяемые объяснения теории;
- `docs/INSTALLATION.md` — установка из GitHub и глобальные команды клиентов;
- `docs/PROJECT-STATE.md` — единое актуальное состояние;
- `docs/OPEN-GAPS.md` — незакрытые задачи;
- `references/` — локальные исходные материалы, не входящие в Git.

Официальные материалы курса считаются первоисточником. Вторичные заметки
используются только как гипотезы и тестовые данные и проверяются независимо.
Текстовые зеркала и производные конспекты официальных материалов сохраняют
лицензию CC BY-SA 4.0; точная область действия, ссылка на источник и описание
изменений приведены в [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
