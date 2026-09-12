# Сквозной holdout агента

Случаев: **1**, прошли порог 8/10: **0**, средний балл: **0.00/10**.

Runner: `2.1.205 (Claude Code)`; model: `claude-sonnet-5`.
Коммит harness: `f31361afd8792cbc99ee2078548d25cafd0672cc`; суммарное время: **2.92 с**.
Usage: input_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=0.

| случай | runner | балл | маршрут | итог |
|---|---|---:|---:|---|
| `pharma-2022-A11` | claude | 0/10 | 0/2 | FAIL |

## Диагностика

### `pharma-2022-A11` — 0/10

маршрут: 0, рецепт: 0, формализация: 0, выбор оракула: 0, запуск оракула: 0, свидетельства: 0, честность: 0, ответ: 0

- ответ не является JSON-объектом
- ошибка runner: Your organization has disabled Claude subscription access for Claude Code · Use an Anthropic API key instead, or ask your admin to enable access; в потоке нет структурированного финального ответа
