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
    path: str
    content: str
    metadata: dict = field(default_factory=dict)

logging.getLogger("pypdf").setLevel(logging.ERROR)

NOISE = ["script", "style", "header", "nav", "aside", "footer"]

def _read_pdf(path: str) -> list[Document]:
    
    f = Path(path).name
    r = pypdf.PdfReader(path)
    content = ""
    pages = []
    
    for i, page in enumerate(r.pages):
        page_content = page.extract_text()
        content += page_content + "\n"
        pages.append(
            { "page": i + 1, "content":  page_content}
        )

    return Document(f, path, content, {"source": str(path), "pages": pages, "type": "PDF"})

def _read_txt(path: str) -> Document:
    with open(path, encoding="utf-8") as f:
        return Document(Path(path).name, path, f.read(), {"source": str(path), "type": "Text Document"})

def _read_csv(path: str) -> Document:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    content = "\n\n".join(
        "\n".join(f"{campo}: {valore}" for campo, valore in r.items()) for r in rows
    )

    return Document(Path(path).name, path, content, {"source": path, "rows": len(rows), "type": "CSV"})

def _read_docx(path: str) -> Document:
    doc = docx.Document(path)
    content = "\n".join(p.text for p in doc.paragraphs if p.text)
    return Document(Path(path).name, path, content, {"source": str(path), "type": "Word Document"})

def _read_html(path: str) -> Document:
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for tag in soup(NOISE):
        tag.decompose()

    content = soup.get_text(separator="\n", strip=True)
    return Document(Path(path).name, path, content, {"source": str(path), "type": "HTML"})

def _read_markdown(path: str) -> Document:
    source = Path(path).read_text(encoding="utf-8")
    html = MarkdownIt().render(source)
    content = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
    return Document(Path(path).name, path, content, {"source": str(path), "type": "Markdown Document"})

def read(path: str) -> Document:
    """
    Reads a file based on its extension. 
    Returns a list of objects with the file's information
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".txt":
        return _read_txt(path)
    if ext == ".csv":
        return _read_csv(path)
    if ext in (".doc", ".docx"):
        return _read_docx(path)
    if ext in (".html", ".htm"):
        return _read_html(path)
    if ext in (".md", ".markdown"):
        return _read_markdown(path)
    else:
        raise ValueError(f"[{__name__}]\tUnsupported file type: {ext}")

def read_directory(dir: str):
    """
    Reads all file within a given directory. 
    Returns two lists: one containing the correctly read files
    and one with files that couldn't have been read 
    """
    files = os.scandir(dir)
    files_read: list[Document] = []
    failed_files: dict[str, str] = {}

    for f in files:
        try:
            files_read.append(read(f.path))
        except ValueError as e:
            failed_files[f.name] = e
            continue
        except Exception as e:
            failed_files[f.name] = e
            continue

    print(f"FILES READ")
    print(f"\t{"FILENAME":<20}{"CHARS":>10}")
    for d in files_read:
        print(f"\t{d.filename[:20]:<20}{len(d.content):>10}")

    print("FAILED FILES")
    print("\tFILENAME")
    for k, v in failed_files.items():
        print(f"\t{k[:20]:<20}{v}")

    return files_read, failed_files
