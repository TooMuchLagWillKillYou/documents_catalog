import os
import pypdf
from pathlib import Path
from dataclasses import dataclass, field

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)

def _load_pdf(path: str) -> list[Document]:
    reader = pypdf.PdfReader(path)

    pages = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"\t\tText: {text}")
        pages.append(
            Document(text, {"source": str(path), "page": i + 1, "type": "pdf"})
        )
    return pages

def _load_txt(path: str) -> Document:
    pass

def _load_csv(path: str) -> Document:
    pass

def load_documents() -> list[Document]:

    files_found = os.scandir(DOCS_DIRECTORY)
    docs = []
    unhandled_docs = []

    for f in files_found:
        print(f"\tHandling: {Path(f.path).name}")
        extension = Path(f.path).suffix.lower()

        if extension == ".pdf":
            docs.extend(_load_pdf(f.path))
        elif extension == ".txt":
            docs.append(_load_txt(f.path))
        elif extension == ".csv":
            docs.append(_load_csv(f.path))
        else:
            unhandled_docs.append(f.path)

    return docs
    

if __name__ == "__main__":
    load_documents()