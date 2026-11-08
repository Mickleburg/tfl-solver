"""Build a portable RK1 teaching bundle; never overwrite an existing destination."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "study/rk1-2026/trainer"
OLD = ROOT / "study/rk1-2026/final-sprint"
CARD = re.compile(r"^## ((?:T\d{2}-[A-Z]|[PRG]\d{2}))(?:\.|$)", re.M)
TYPES = (
    "Критические пары SRS", "Завершимость SRS", "Конечность моноида",
    "Автомат → выражение", "Выражение → автомат", "Минимальность ДКА",
    "Детерминизация НКА", "Пересечение ДКА", "Дополнение языка",
    "Регулярность языка", "Неоднозначность выражения", "Эквивалентность выражений",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def sections(text: str) -> dict[str, str]:
    matches = list(CARD.finditer(text))
    if len({m[1] for m in matches}) != len(matches):
        raise ValueError("Повторяющиеся ID карточек")
    return {m[1]: text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
            for i, m in enumerate(matches)}


def documents() -> dict[str, str]:
    tasks = read(OLD / "TASKS.md") + "\n\n" + read(SOURCE / "EXTRA-TASKS.md")
    keys = read(OLD / "KEYS.md") + "\n\n" + read(SOURCE / "EXTRA-KEYS.md")
    task_sections, key_sections = sections(tasks), sections(keys)
    if task_sections.keys() != key_sections.keys():
        raise ValueError("Условия и ключи имеют разные ID")
    by_type: dict[int, list[str]] = {i: [] for i in range(1, 13)}
    for card_id, body in task_sections.items():
        kind = re.search(r"Тип: (\d+)\.", body)
        if not kind or int(kind[1]) not in by_type or "Требуется" not in body:
            raise ValueError(f"Неполное условие {card_id}")
        if not all(word in key_sections[card_id] for word in ("Подсказка:", "Решение:")):
            raise ValueError(f"Неполный ключ {card_id}")
        by_type[int(kind[1])].append(card_id)
    progress = ["# Прогресс подготовки к РК1", "", "current: T10-B", "",
                "Это пустой шаблон. Старый личный прогресс переносится отдельно;",
                "просмотр ключа не считается самостоятельным решением.", "",
                "## Блоки", "", "| Тип | Тема | Карточки |", "|---|---|---|"]
    for kind, ids in by_type.items():
        progress.append(f"| {kind} | {TYPES[kind - 1]} | {', '.join(ids)} |")
    progress += ["", "## Результаты", "", "| ID | Статус | Время | Помощь | Ошибка / вывод |",
                 "|---|---|---|---|---|"]
    progress += [f"| {card_id} | не начата | — | — | — |" for card_id in task_sections]
    progress += ["", "## Прежняя история", "", "Не приложена; готовность не измерена.",
                 "", "## Новые карточки N", "", "Нет. Условия и ключи переносятся отдельно."]
    result = {
        "TASKS.md": tasks, "KEYS.md": keys,
        "METHODS.md": read(OLD / "METHODS.md") + "\n\n" + read(SOURCE / "METHODS-EXTRA.md"),
        "PROGRESS.md": "\n".join(progress),
    }
    for name in ("START-PROMPT.md", "GENERATION.md", "README.md", "SOURCE-AUDIT.md"):
        result[name] = read(SOURCE / name)
    return result


def build(destination: Path, *, archive: bool = False) -> Path:
    destination = destination.resolve()
    zip_path = destination.with_name(destination.name + ".zip")
    if destination.exists() or (archive and zip_path.exists()):
        raise FileExistsError("Выбери новый каталог: существующий комплект/архив не перезаписывается")
    payload = documents()  # Validate everything before creating output.
    destination.mkdir(parents=True)
    for name, body in payload.items():
        with (destination / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(body + "\n")
    if archive:
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in payload:
                bundle.write(destination / name, arcname=name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", action="store_true", dest="archive")
    args = parser.parse_args()
    try:
        result = build(args.output, archive=args.archive)
    except (FileExistsError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(result)


if __name__ == "__main__":
    main()
