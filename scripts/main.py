import os
from pathlib import Path
from loaders import load

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"

if __name__ == "__main__":

    files = os.scandir(DOCS_DIRECTORY)
    docs_with_errors: dict[str, str] = {}

    for f in files:
        try:
            load(f.path)
        except ValueError as e:
            docs_with_errors[f.name] = e
        except Exception as e:
            docs_with_errors[f.name] = e

    for k, v in docs_with_errors.items():
        print(f"{k}{v}")