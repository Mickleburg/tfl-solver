#!/usr/bin/env python3
"""Разбор экспорта учебного чата: отделить предметное от бытового.

    py -3 tools/extract_chat.py

Читает `references/chat/chat-onTG-2025-tfl/result.json` и складывает в `corpus/chat/`:

* `messages.md` — сообщения по делу, хронологически, с автором и датой;
* `photos.tsv` — каталог фотографий с контекстом (что писали в самом
  сообщении и в том, на которое отвечали), чтобы понимать, что смотреть;
* `dropped.tsv` — что и почему отброшено. Отбрасывать молча нельзя:
  фильтр может ошибаться, и проверить его должно быть можно.

**Персональные данные третьих лиц не извлекаются.** В чате есть списки
групп с ФИО, баллами и допусками к экзамену; к предмету они отношения
не имеют, а распространять их мы не вправе. Такие сообщения отсеиваются
по образцу «три и более строки вида ФИО + число» и по спискам @ников,
и в `dropped.tsv` попадают без текста — только с причиной.

Предметность определяется теми же признаками, что и в приёме задачи
(`tfl.intake.FEATURES` и `ASKS`): если в сообщении есть формальный объект
или вопрос по предмету, оно по делу.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tfl.intake import find_asks, find_features

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPORT = ROOT / "references" / "chat" / "chat-onTG-2025-tfl"
OUT = ROOT / "corpus" / "chat"

#: ФИО с номером и/или баллом — строка списка группы.
FIO_LINE = re.compile(r"^\s*\d*\s*[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+", re.M)
NICKS = re.compile(r"@[A-Za-z0-9_]{3,}")
#: Слова курса. Не признаки объекта, а просто указание, что речь о предмете.
TOPIC = re.compile(
    r"автомат|грамматик|язык|накачк|регуляр|детермин|переписыван|SRS|TRS|PDA|"
    r"лаб|РК|билет|экзамен|минимиз|вывод|стек|нетерминал|ДКА|НКА|КС\b",
    re.I,
)
JUNK_MEDIA = {"sticker", "animation", "video_message"}


def plain(message: dict) -> str:
    text = message.get("text")
    if isinstance(text, str):
        return text
    return "".join(
        part if isinstance(part, str) else part.get("text", "") for part in text
    )


def personal(text: str) -> str | None:
    """Причина, по которой сообщение считается персональными данными."""
    if len(FIO_LINE.findall(text)) >= 3:
        return "список ФИО"
    if len(NICKS.findall(text)) >= 4:
        return "список ников"
    if re.search(r"допущен|пересдач|списк\w+ групп|зачётк", text, re.I) and (
        FIO_LINE.search(text) or len(NICKS.findall(text)) >= 2
    ):
        return "список допущенных"
    return None


def relevance(text: str) -> int:
    """Насколько сообщение предметное. Порог подобран по выдаче, а не теоретически."""
    score = 0
    if find_features(text):
        score += 3
    if find_asks(text):
        score += 2
    if TOPIC.search(text):
        score += 2
    if len(text) > 250:
        score += 1
    if re.search(r"→|::=|\{.*\|.*\}|\|w\|", text):
        score += 2
    return score


def main() -> None:
    data = json.loads((EXPORT / "result.json").read_text(encoding="utf-8"))
    messages = data["messages"]
    by_id = {m["id"]: m for m in messages}

    OUT.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    photos: list[dict] = []
    dropped: list[tuple[int, str, str]] = []

    for message in messages:
        text = plain(message).strip()
        media = message.get("media_type")
        has_photo = "photo" in message

        reason = personal(text)
        if reason:
            dropped.append((message["id"], reason, ""))
            continue
        if media in JUNK_MEDIA and not text:
            dropped.append((message["id"], f"медиа-{media}", ""))
            continue

        if has_photo:
            parent = by_id.get(message.get("reply_to_message_id"))
            photos.append(
                {
                    "id": message["id"],
                    "date": message["date"][:16],
                    "from": message.get("from", "?"),
                    "path": message["photo"],
                    "caption": text.replace("\n", " ")[:200],
                    "reply_to": (plain(parent).replace("\n", " ")[:200] if parent else ""),
                }
            )
            continue

        score = relevance(text)
        candidates.append(
            {
                "id": message["id"],
                "date": message["date"][:16],
                "from": message.get("from", "?"),
                "text": text,
                "score": score,
                "reply_to": message.get("reply_to_message_id"),
            }
        )

    # Разговор ценен веткой, а не отдельной репликой: «так всё-таки в T′ мы
    # проверяем, можно ли w→w′?» само по себе бессодержательно, а вместе
    # с вопросом, на который отвечает, — вполне. Поэтому к предметным
    # сообщениям добавляется один шаг связи в обе стороны.
    good = {c["id"] for c in candidates if c["score"] >= 3 and len(c["text"]) >= 15}
    replies_to_good = {
        c["id"] for c in candidates if c["reply_to"] in good and len(c["text"]) >= 15
    }
    parents_of_good = {
        c["reply_to"] for c in candidates if c["id"] in good and c["reply_to"]
    }
    keep_ids = good | replies_to_good | parents_of_good
    kept = [c for c in candidates if c["id"] in keep_ids]
    for c in candidates:
        if c["id"] not in keep_ids:
            dropped.append((c["id"], f"не по делу (счёт {c['score']})", c["text"][:60]))

    lines = [
        "# Учебный чат ТФЯ — предметные сообщения",
        "",
        f"Из {len(messages)} сообщений оставлено {len(kept)}; фотографий "
        f"{len(photos)} (каталог — `photos.tsv`), отброшено {len(dropped)} "
        "(`dropped.tsv`).",
        "",
        "> Чат — **не источник истины**. Здесь говорят и преподаватель, и студенты, ",
        "> и отличить одно от другого по имени аккаунта нельзя: сообщения ",
        "> преподавателя пересылаются со старостиного аккаунта. Всё, что отсюда ",
        "> берётся в решение, проверяется оракулом или сверяется с репозиторием ",
        "> преподавателя.",
        "",
    ]
    for item in kept:
        lines.append(f"## [{item['id']}] {item['date']} — {item['from']}")
        lines.append("")
        lines.append(item["text"])
        lines.append("")

    (OUT / "messages.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT / "photos.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "date", "from", "path", "caption", "reply_to"])
        for photo in photos:
            writer.writerow(
                [photo["id"], photo["date"], photo["from"], photo["path"],
                 photo["caption"], photo["reply_to"]]
            )

    with (OUT / "dropped.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "причина", "начало"])
        writer.writerows(dropped)

    print(f"сообщений по делу: {len(kept)}, фотографий: {len(photos)}, "
          f"отброшено: {len(dropped)}")
    print(f"каталог: {OUT}")


if __name__ == "__main__":
    main()
