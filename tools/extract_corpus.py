#!/usr/bin/env python3
"""Извлечение текстового корпуса из PDF курса ТФЯ.

Использование:
    py -3 tools/extract_corpus.py            # переизвлечь всё в corpus/txt
    py -3 tools/extract_corpus.py --render FILE.pdf 1 5   # отрендерить стр. 1..5 в PNG

Зачем: почти все материалы курса лежат в PDF. Grep по PDF невозможен,
поэтому агент работает с плоским текстовым зеркалом в corpus/txt/.

Подводные камни, которые скрипт лечит:
  * pdftotext отдаёт кириллицу LaTeX-овских PDF байтами CP1251 -> используем PyMuPDF;
  * часть PDF всё равно приходит "мохибейком" (Âàðèàíò) -> посимвольная
    перекодировка cp1251 с проверкой по доле кириллицы;
  * сканы (рукописные решения, конспекты) текстового слоя не имеют
    либо имеют мусорный OCR -> помечаются в INDEX.tsv как SCAN,
    их читают только рендером в PNG.

Требуется: pip install pymupdf
"""
import hashlib
import pathlib
import sys

import pymupdf

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "txt"
# Метка сохраняет стабильные имена corpus/txt, а путь отражает единый
# локальный каталог references/. Это позволяет перемещать весь проект без
# абсолютных путей и не пересобирать индекс задач из-за смены раскладки.
SOURCES = (
    ("FormalLanguageTheory", pathlib.Path("references/teacher/FormalLanguageTheory")),
    ("TFL-IU9-claude", pathlib.Path("references/course/TFL-IU9-claude")),
    ("Supplement", pathlib.Path("references/course/supplemental")),
)
# Персональные данные третьих лиц: списки групп с ФИО. В корпус не идут —
# к предмету отношения не имеют, а распространять их мы не вправе.
SKIP = ("Группа ИУ9", "Список группы")
CYR = set("абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
SCAN_CHARS_PER_PAGE = 120  # ниже порога считаем страницу сканом


def demojibake(s: str) -> str:
    """CP1251-байты, прочитанные как latin-1, обратно в кириллицу."""
    out = []
    for c in s:
        if ord(c) < 256:
            try:
                out.append(bytes([ord(c)]).decode("cp1251"))
            except Exception:
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def cyr_count(s: str) -> int:
    return sum(c in CYR for c in s)


def extract(pdf: pathlib.Path) -> tuple[str, int]:
    doc = pymupdf.open(pdf)
    text = "".join(
        f"\n----- [page {i + 1}] -----\n" + p.get_text("text") for i, p in enumerate(doc)
    )
    fixed = demojibake(text)
    if cyr_count(fixed) > cyr_count(text) * 1.5 + 20:
        text = fixed
    return text, len(doc)


def render(pdf: pathlib.Path, first: int, last: int, dpi: int = 150) -> None:
    """Сканы читаются только глазами: рендерим страницы в PNG рядом с PDF."""
    doc = pymupdf.open(pdf)
    outdir = ROOT / "corpus" / "png" / pdf.stem
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(first - 1, min(last, len(doc))):
        pix = doc[i].get_pixmap(dpi=dpi)
        target = outdir / f"p{i + 1:02d}.png"
        pix.save(target)
        print(target)


def main() -> None:
    if "--render" in sys.argv:
        idx = sys.argv.index("--render")
        pdf = pathlib.Path(sys.argv[idx + 1])
        first = int(sys.argv[idx + 2]) if len(sys.argv) > idx + 2 else 1
        last = int(sys.argv[idx + 3]) if len(sys.argv) > idx + 3 else first
        render(pdf, first, last)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: dict[str, str] = {}
    for label, base in SOURCES:
        for pdf in sorted((ROOT / base).rglob("*.pdf")):
            if any(mark in pdf.name for mark in SKIP):
                continue
            # Один и тот же PDF может встречаться в нескольких наборах.
            # Побайтовые дубли только засоряют текстовый поиск.
            digest = hashlib.md5(pdf.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen[digest] = pdf.name
            rel = pdf.relative_to(ROOT)
            inside = pdf.relative_to(ROOT / base)
            stem = str(inside.with_suffix(""))
            name = label + "_" + stem.replace("\\", "_").replace("/", "_").replace(" ", "_") + ".txt"
            try:
                text, pages = extract(pdf)
            except Exception as exc:  # повреждённый или защищённый PDF
                text, pages = f"EXTRACTION ERROR: {exc}", 0
            (OUT / name).write_text(text, encoding="utf-8")
            per_page = len(text) / pages if pages else 0
            kind = "SCAN" if per_page < SCAN_CHARS_PER_PAGE else "TEXT"
            rows.append((kind, pages, len(text), name, str(rel)))

    index = ROOT / "corpus" / "INDEX.tsv"
    with index.open("w", encoding="utf-8") as fh:
        fh.write("kind\tpages\tchars\ttxt\tpdf\n")
        for kind, pages, chars, name, rel in rows:
            fh.write(f"{kind}\t{pages}\t{chars}\t{name}\t{rel}\n")
    scans = sum(1 for r in rows if r[0] == "SCAN")
    print(f"{len(rows)} PDF -> {OUT} ({scans} сканов, читать рендером)")
    print(f"индекс: {index}")


if __name__ == "__main__":
    main()
