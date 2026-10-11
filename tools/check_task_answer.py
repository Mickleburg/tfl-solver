"""Check the explicit task-answer Markdown template, not its mathematics.

Uses only the standard library. Hint quality, Russian usage and preservation
of the original statement require a separate human/editorial review.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re


def _without_fences(markdown: str) -> str:
    """Do not interpret HTML-looking text in fenced code as real spoilers."""
    result = []
    fence = ""
    for line in markdown.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if not fence and marker:
            fence = marker[1]
            result.append("\n")
        elif fence:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = ""
            result.append("\n")
        else:
            result.append(line)
    return "".join(result)


class _Spoilers(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict[str, object]] = []
        self.before: list[str] = []
        self.outside: list[str] = []
        self.issues: list[str] = []
        self.depth = 0
        self.in_summary = False

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            if self.depth:
                self.issues.append("nested-details")
            else:
                self.blocks.append({"summary": [], "body": [], "summaries": 0})
            self.depth += 1
            if any(key == "open" for key, _ in attrs):
                self.issues.append("open-details")
        elif tag == "summary":
            if not self.depth or self.in_summary:
                self.issues.append("misplaced-summary")
            else:
                block = self.blocks[-1]
                block["summaries"] += 1
                self.in_summary = True

    def handle_endtag(self, tag):
        if tag == "summary":
            if not self.in_summary:
                self.issues.append("misplaced-summary")
            self.in_summary = False
        elif tag == "details":
            if not self.depth:
                self.issues.append("unmatched-details-close")
            else:
                if self.in_summary:
                    self.issues.append("unclosed-summary")
                    self.in_summary = False
                self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            field = "summary" if self.in_summary else "body"
            self.blocks[-1][field].append(data)
        elif self.blocks:
            self.outside.append(data)
        else:
            self.before.append(data)


def check_task(markdown: str) -> list[str]:
    """Return structural issues for one task in the documented template."""
    parser = _Spoilers()
    text = _without_fences(markdown)
    # HTMLParser is not a Markdown/MathJax parser: '<n' in a formula must not
    # swallow the next closing tag, nor should inline code create a spoiler.
    def protect(match):
        return match[0].replace("<", "&lt;").replace(">", "&gt;")

    text = re.sub(r"(?s)(`+)(.*?)\1", protect, text)
    text = re.sub(r"(?s)\$\$.*?\$\$|\$(?:\\.|[^$])*\$", protect, text)
    parser.feed(text)
    parser.close()
    issues = list(parser.issues)
    if parser.depth:
        issues.append("unclosed-details")
    before = "".join(parser.before)
    condition = re.search(r"(?m)^### Условие\s*$", before)
    if not condition:
        issues.append("missing-condition")
    elif not re.search(r"\bТребуется\s+\S", before[condition.end():]):
        issues.append("missing-requirement")
    if len(parser.blocks) != 2:
        issues.append("expected-two-spoilers")
    for index, block in enumerate(parser.blocks):
        expected = ("Подсказка к решению", "Решение")
        summary = " ".join("".join(block["summary"]).split())
        if (index >= len(expected) or block["summaries"] != 1
                or summary != expected[index]):
            issues.append("wrong-spoiler-label-or-order")
        if not "".join(block["body"]).strip():
            issues.append("empty-spoiler")
    if "".join(parser.outside).strip():
        issues.append("text-outside-spoilers")
    return sorted(set(issues))


def check_document(markdown: str) -> list[str]:
    """Check each '## Задача …' block, or a single unnumbered task."""
    # Remove fences before splitting so example headings in code are ignored.
    text = _without_fences(markdown)
    headings = list(re.finditer(r"(?m)^## Задача\b[^\n]*", text))
    if not headings:
        return check_task(text)
    issues = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        for issue in check_task(text[heading.start():end]):
            issues.append(f"{heading[0]}: {issue}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        issues = check_document(args.file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as error:
        print(f"ERROR: {error}")
        return 2
    for issue in issues:
        # ASCII escaping keeps diagnostics usable on Windows terminals.
        print("FAIL: " + issue.encode("ascii", "backslashreplace").decode("ascii"))
    if not issues:
        print("PASS: structure only; meaning, hint quality and language need review")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
