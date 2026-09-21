"""
cleaner.py cleans text, one step at the time
"""

import re
import unicodedata
from collections import Counter
from reader import Document

def _remove_repeated_lines(text: str) -> str:
    counter = Counter()

    for r in text.split("\n"):
        r = r.strip()
        if r:
            counter[r] += 1

    to_remove = set()
    for row, count in counter.items(): 
        if count >= 3 and len(row) <= 80:
            to_remove.add(row)

    to_keep = []
    for r in text.split("\n"):
        if r.strip() not in to_remove:
            to_keep.append(r)
    return "\n".join(to_keep)


def _remove_page_numbers(text: str) -> str:
    to_keep = []
    for r in text.split("\n"):
        if re.fullmatch(r"\s*(pagina\s+)?\d+(\s+di\s+\d+)?\s*", r, re.I):
            continue
        to_keep.append(r)
    return "\n".join(to_keep)


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC",text)
    text = re.sub(r"[ \t]+", " ", text)                           # remove spaces and tabs
    text = re.sub(r"\n{3,}", "\n\n", text)
    rows = [r.strip() for r in text.split("\n")]
    return "\n".join(rows).strip()


def _reflow_paragraphs(text: str) -> str:
    rows = text.split("\n")
    out = []

    for r in rows:
        r = r.strip()
        if out and r and not out[-1].endswith((".", "!", "?", ":", ";")):
            out[-1] = out[-1] + " " + r
        else:
            out.append(r)

    return "\n".join(out)


def _clean_pdf(content: str) -> str:
    content = _remove_repeated_lines(content)
    content = _remove_page_numbers(content)
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
            f.content = _clean_pdf(f.content)
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