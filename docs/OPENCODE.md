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

### Изолированный клон в WSL

Для Debian клон размещается в `~/Projects/tfl-solver`. Личный GitHub не должен
менять рабочую конфигурацию GitLab: отдельный ключ хранится в
`~/Projects/.ssh/github_personal_ed25519`, а `user.name`, `user.email`,
`core.sshCommand` и SSH-адрес `origin` задаются в локальном `.git/config`
клона. Глобальные Git- и OpenCode-конфиги при этом не изменяются.

После создания ключа его открытая часть добавляется в личный GitHub. Проверка
доступа и обновление клона выполняются уже из `~/Projects/tfl-solver`:

```bash
ssh -T -i ~/Projects/.ssh/github_personal_ed25519 \
  -o IdentitiesOnly=yes git@github.com
git fetch origin
```

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
В WSL Debian найден OpenCode 1.18.32 и выполнен реальный smoke на
`pt/positive-llm-qwen36`. Модель дала правильный итог, но сделала ложное
промежуточное наблюдение и использовала 20 вызовов вместо 8. Рубрика revision 6
отдельно ловит содержательную ошибку пути и превышение бюджета; результат не
маскируется как PASS. Подробности — в отчёте
`reports/pedagogy/opencode-qwen36-smoke-2026-09-28.md`.
