import os
from pathlib import Path
from loaders import load, Document
from cleaner import clean

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"

def _load_documents():
    files = os.scandir(DOCS_DIRECTORY)
    docs: list[Document] = []
    docs_with_errors: dict[str, str] = {}

    for f in files:
        try:
            docs.append(load(f.path))
        except ValueError as e:
            docs_with_errors[f.name] = e
        except Exception as e:
            docs_with_errors[f.name] = e

    return docs, docs_with_errors

if __name__ == "__main__":

    docs, docs_with_errors = _load_documents()
    cleaned_docs = [clean(d) for d in docs]

    print("LOADED DOCUMENTS")
    for d in cleaned_docs:
        print(f"\tname: {d.filename[:15]} | chars: {len(d.text):>20}\n"
              f"\tcontent: {d.text[:100]}")

    print("\n")

    print("FAILED DOCUMENTS")
    for d in docs_with_errors:
        print(f"\tname: {d}")