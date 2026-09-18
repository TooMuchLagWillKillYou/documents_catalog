import os
from pathlib import Path
from dataclasses import dataclass, field

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"

@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)

def load_documents() -> list[Document]:
    files = os.scandir(DOCS_DIRECTORY)

    for f in files:
        print(f"File: {f}")

if __name__ == "__main__":
    load_documents()