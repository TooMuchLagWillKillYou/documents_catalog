"""
cleaner.py cleans text, one step at the time
"""

import re
import unicodedata
from collections import Counter
from reader import Document

def _repeated_lines(text: str) -> set[str]:
    """
    Detects repeated line that will be removed later
    """
    counter = Counter()

    for r in text.split("\n"):
        r = r.strip()
        if r:
            counter[r] += 1

    to_remove = set()
    for row, count in counter.items():
        if count >= 3 and len(row) <= 80:
            to_remove.add(row)
    return to_remove


def _remove_repeated_lines(text: str, to_remove: set[str]) -> str:
    to_keep = []
    for r in text.split("\n"):
        if r.strip() not in to_remove:
            to_keep.append(r)
    return "\n".join(to_keep)


PAGE_NUMBER = re.compile(r"\s*(pagina\s+)?\d+(\s+di\s+\d+)?\s*", re.I)


def _remove_page_numbers(page: str, repeated: set[str]) -> str:
    """
    Removes the page number from a single page. From the top and from the bottom,
    blank and 'repeated' rows (headers, footers) are skipped: the first other row
    is removed only if it is a page number. Numbers in the rest of the page
    (years, amounts, table cells) are content and are kept.

    BE AWARE: this mthod is weak since it only remove page numbers that
    match the 'PAGE_NUMBER' regex
    """
    rows = page.split("\n")
    to_remove = set()

    for order in (range(len(rows)), reversed(range(len(rows)))):
        for i in order:
            r = rows[i].strip()
            if not r or r in repeated:
                continue
            if PAGE_NUMBER.fullmatch(r):
                to_remove.add(i)
            break

    return "\n".join(r for i, r in enumerate(rows) if i not in to_remove)


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w+)[ \t]*-[ \t]*\n(\w+)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    """
    Replaces tabs with a whitespace (" ") and threeor more consecutives
    newlines (\n) with two (\n\n)
    """
    text = unicodedata.normalize("NFKC",text)
    text = re.sub(r"[ \t]+", " ", text)                           # remove spaces and tabs
    text = re.sub(r"\n{3,}", "\n\n", text)
    rows = [r.strip() for r in text.split("\n")]
    return "\n".join(rows).strip()


def _reflow_paragraphs(text: str) -> str:
    """
    Concats text rows unless the previous one ends 
    with one of these (".", "!", "?", ":", ";")
    """
    rows = text.split("\n")
    out = []

    for r in rows:
        r = r.strip()
        if out and out[-1] and r and not out[-1].endswith((".", "!", "?", ":", ";")):
            out[-1] = out[-1] + " " + r
        else:
            out.append(r)

    return "\n".join(out)


def _clean_pdf(pages: list[str]) -> str:
    repeated = _repeated_lines("\n".join(pages))
    content = "\n".join(_remove_page_numbers(p, repeated) for p in pages)
    content = _remove_repeated_lines(content, repeated)
    content = _dehyphenate(content)
    content = _normalize_whitespace(content)
    content = _reflow_paragraphs(content)
    return content


def _clean_txt(content: str) -> str:
    content = _normalize_whitespace(content)
    content = _reflow_paragraphs(content)
    return content


def clean(files: list[Document]):
    result: list[Document] = []

    for f in files:
        if f.filename.endswith(".pdf"):
            f.content = _clean_pdf([p["content"] for p in f.metadata["pages"]])
        elif f.filename.endswith(".txt"):
            f.content = _clean_txt(f.content)
        else:
            f.content = _normalize_whitespace(f.content)
        result.append(f)

    print(f"FILES CLEANED")
    print(f"\t{"FILENAME":<20}{"CHARS":>10}")
    for f in result:
        print(f"\t{f.filename[:20]:<20}{len(f.content):>10}")

    return result