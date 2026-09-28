# OpenCode и локальные модели

OpenCode использует уже существующий адаптер
`.agents/skills/tfl-solver/SKILL.md`: отдельная копия предметных инструкций не
нужна. По официальному контракту OpenCode ищет project skills в
`.agents/skills/*/SKILL.md`, а модель для одного запуска задаётся как
`provider/model`.

## Обычная работа

После установки OpenCode открой корень репозитория:

```powershell
opencode .
```

Сформулируй задачу обычным языком или явно попроси загрузить skill
`tfl-solver`. `AGENTS.md`, канонический skill, корпус и Python-оракулы остаются
теми же, что в Codex и Claude Code.

Доступные модели нужно узнавать у самого клиента, не угадывать по имени:

```powershell
opencode models
```

OpenCode поддерживает локальные OpenAI-совместимые серверы и Ollama. Конкретный
provider и model ID настраиваются пользователем в OpenCode; репозиторий не
фиксирует модель и не хранит ключи доступа. Официальные инструкции:

- <https://opencode.ai/docs/providers/>;
- <https://opencode.ai/docs/models/>;
- <https://opencode.ai/docs/skills>.

## Измеримый smoke для слабой модели

Сначала запускается один тренировочный случай:

```powershell
py -3 -m tfl pedagogy run `
  --runner opencode `
  --model ollama/<model-id> `
  --case training-unbounded-delay `
  --timeout 900 `
  --output reports/pedagogy/opencode-smoke.jsonl
```

После ручной проверки математики можно запускать остальные случаи. Для
OpenCode используется `.opencode/agents/tfl-eval.md`: он запрещает изменение
репозитория, сеть и подагентов, разрешая чтение и Python-проверки. Runner читает
JSONL-события `tool_use`, `text` и `step_finish`, поэтому scorer видит реальные
вызовы оракулов и расход токенов.

Полный набор не следует запускать первым: Codex baseline потребовал более
миллиона входных токенов даже с кэшированием. На локальной модели сначала нужно
измерить один случай, затем исправлять только наблюдаемые сбои. Если tool calls
обрываются, официальный provider guide рекомендует проверить размер контекста;
для Ollama отправная точка — 16–32K.

## Граница текущей проверки

Адаптер, JSONL-parser, схема ответа и read-only профиль покрыты unit-тестами.
На текущей машине OpenCode и локальный model server не установлены, поэтому
реальный OpenCode baseline ещё не снят. Это состояние отражается в
`docs/OPEN-GAPS.md`, а не маскируется синтетическим PASS.
