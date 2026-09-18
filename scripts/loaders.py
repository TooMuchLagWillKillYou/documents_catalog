"""
loaders.py reads documents in different formats and normalize them in a Document class
"""

import os
import pypdf
import logging
import csv
import docx
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Document:
    filename: str
    text: str
    metadata: dict = field(default_factory=dict)

logging.getLogger("pypdf").setLevel(logging.ERROR)

NOISE = ["script", "style", "header", "nav", "aside", "footer"]

def _load_pdf(path: str) -> list[Document]:

    reader = pypdf.PdfReader(path)
    text = ""
    pages = []
    
    for i, page in enumerate(reader.pages):
        text += page.extract_text()
        pages.append(
            { "page": i + 1, "content":  page.extract_text()}
        )
    return Document(Path(path).name, text, {"source": str(path), "pages": pages, "type": "PDF"})

def _load_txt(path: str) -> Document:
    with open(path, encoding="utf-8") as f:
        return Document(Path(path).name, f.read(), {"source": str(path), "type": "Text Document"})

def _load_csv(path: str) -> Document:
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter=" ", quotechar="|")
        result = ""
        for r in reader:
            result += ", ".join(r)
        return Document(Path(path).name, result, {"source": path, "type": "CSV"})

def _load_docx(path: str) -> Document:
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text)
    return Document(Path(path).name, text, {"source": str(path), "type": "Word Document"})

def _load_html(path: str) -> Document:
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(NOISE):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return Document(Path(path).name, text, {"source": str(path), "type": "HTML"})

def _load_markdown(path: str) -> Document:
    source = Path(path).read_text(encoding="utf-8")
    html = MarkdownIt().render(source)
    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    return Document(Path(path).name, text, {"source": str(path), "type": "Markdown Document"})

def load(path: str) -> Document:

    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".txt":
        return _load_txt(path)
    if ext == ".csv":
        return _load_csv(path)
    if ext in (".doc", ".docx"):
        return _load_docx(path)
    if ext in (".html", ".htm"):
        return _load_html(path)
    if ext in (".md", ".markdown"):
        return _load_markdown(path)
    else:
        raise ValueError(f"[{__name__}]\tUnsupported file type: {ext}")