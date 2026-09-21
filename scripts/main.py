import os
from pathlib import Path
from reader import read, read_directory ,Document
from cleaner import clean

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"

if __name__ == "__main__":

    files, _ = read_directory(DOCS_DIRECTORY)
    cleaned_files = clean(files)
