# Установка и перенос tfl-solver

Версия 1.0 распространяется тремя согласованными способами:

1. плагин Codex;
2. плагин Claude Code;
3. Python-пакет с самостоятельной CLI `tfl`.

Во всех вариантах используется один канонический skill `skills/tfl/SKILL.md`
и одна база рецептов. Локальные тяжёлые оригиналы из `references/` не входят
в дистрибутив; для обычного решения достаточно версионируемого `corpus/`.

## Codex: установка плагина

Добавь GitHub-репозиторий как marketplace и установи плагин:

```text
codex plugin marketplace add Mickleburg/tfl-solver
codex plugin add tfl-solver@tfl-solver-marketplace
```

Открой новую сессию Codex и вызови skill:

```text
$tfl Объясни разницу между детерминированным и однозначным КС-языком
$tfl Реши задачу из приложенного изображения и проверь ключевые гипотезы
$tfl Создай автономный проект для этой ЛР и подготовь описание с промптами
$tfl Найди первую ошибку в этом решении и предъяви контрпример
```

Можно не писать ключевое слово: описание skill позволяет Codex выбрать его
автоматически по обычному запросу о ТФЯ. Установку проверяют команды:

```text
codex plugin marketplace list
codex plugin list
```

Для обновления сначала выполни
`codex plugin marketplace upgrade tfl-solver-marketplace`. Если установленная
копия не обновилась, переустанови её командами `codex plugin remove tfl-solver`
и `codex plugin add tfl-solver@tfl-solver-marketplace`.

## Claude Code: установка плагина

В интерактивной сессии Claude Code:

```text
/plugin marketplace add Mickleburg/tfl-solver
/plugin install tfl-solver@tfl-solver-marketplace
```

После установки перезапусти сессию и используй namespaced-команду:

```text
/tfl-solver:tfl Объясни теорему Майхилла — Нероуда
/tfl-solver:tfl Реши задачу и проверь доказательство
/tfl-solver:tfl Создай автономный проект для этой ЛР
```

Пространство имён `tfl-solver:` добавляет сам формат плагинов Claude Code;
в marketplace-плагине получить глобальную ненеймспейсную `/tfl` нельзя.
Обновление выполняется командой:

```text
/plugin marketplace update tfl-solver-marketplace
/plugin update tfl-solver@tfl-solver-marketplace
```

Те же операции доступны из оболочки через `claude plugin marketplace ...` и
`claude plugin ...`.

## Python и исполняемые оракулы

Для ответа по теории и чтения корпуса отдельная Python-установка не нужна.
Для машинной проверки гипотез, создания и аудита проектов ЛР требуется
Python 3.11+. Агент запускает CLI прямо из каталога распакованного плагина:

```powershell
py -3 scripts/tfl_plugin.py doctor
py -3 scripts/tfl_plugin.py intake --file task.txt
py -3 scripts/tfl_plugin.py lab init path/to/lab --file task.txt
py -3 scripts/tfl_plugin.py lab check path/to/lab
```

На Linux/macOS вместо `py -3` используется `python3`. Скрипт не устанавливает
зависимости и работает с исходниками, уже входящими в плагин.

Если нужна глобальная команда `tfl`, установи Python-пакет:

Windows PowerShell:

```powershell
py -3 -m pip install --user --upgrade "git+https://github.com/Mickleburg/tfl-solver.git"
py -3 -m tfl doctor
```

Linux/macOS:

```bash
python3 -m pip install --user --upgrade "git+https://github.com/Mickleburg/tfl-solver.git"
python3 -m tfl doctor
```

После этого доступны:

```text
tfl doctor
tfl intake --file task.txt
tfl lab init path/to/lab --file task.txt
tfl lab check path/to/lab
tfl eval
```

Форма `python -m tfl` эквивалентна глобальной команде. Имя `tfl-agent`
сохранено для обратной совместимости.

## Совместимость со старой ручной установкой

Команда `tfl setup` остаётся доступной, но для plugin-установки она не нужна.
Она создаёт три маленьких пользовательских адаптера и не перезаписывает чужой
файл без `--force`:

- `~/.claude/skills/tfl/SKILL.md` — Claude Code `/tfl`;
- `~/.agents/skills/tfl-solver/SKILL.md` — Codex `$tfl-solver`;
- `~/.codex/prompts/tfl.md` — Codex `/prompts:tfl`.

```powershell
py -3 -m tfl setup
py -3 -m tfl setup --status
```

Этот путь полезен для старых версий клиентов и даёт короткую команду `/tfl`
в Claude Code. После первой установки либо изменения пути перезапусти клиента.

## Локальная разработка и проверка плагина

```powershell
git clone https://github.com/Mickleburg/tfl-solver.git
cd tfl-solver
py -3 -m pip install -e ".[dev]"
py -3 -m pytest -q
py -3 -m tfl doctor
```

Структуру текущего каталога без публикации можно проверить напрямую:

```text
codex plugin marketplace add .
claude --plugin-dir ./
```

Codex при этом проверяет repo-marketplace; его запись установки намеренно
указывает на GitHub, поэтому устанавливаемая копия соответствует опубликованной
ветке `main`, а не незакоммиченному рабочему дереву. Claude Code с
`--plugin-dir` загружает именно текущие файлы. Манифесты перед релизом
проверяются так:

```powershell
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

## Другие агенты

Корень репозитория следует переносимому формату Agent Plugins: основной
`plugin.json`, skills в `skills/`, документация, корпус и исполняемые скрипты
лежат рядом и используют относительные пути. Клиент с поддержкой этого формата
может загрузить репозиторий напрямую. В другом локальном агенте достаточно
передать ему каталог и попросить прочитать `skills/tfl/SKILL.md`.

В веб-интерфейсе без доступа к локальным файлам можно загрузить нужный рецепт,
skill и условие, но Python-оракулы тогда запускаются отдельно.
