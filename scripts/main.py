from pathlib import Path
from reader import read_directory
from cleaner import clean
from chunker import semantic_chunk
from dotenv import load_dotenv

load_dotenv()

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"
THRESHOLD = 0.4

if __name__ == "__main__":

    files, _ = read_directory(DOCS_DIRECTORY)
    cleaned_files = clean(files)

    print(f"FILES CHUNKED")
    print(f"\t{"FILENAME":<20}{"CHUNKS":>10}")
    # for f in cleaned_files:
    f = cleaned_files[0]
    chunks = semantic_chunk(f.content, THRESHOLD)
    print(f"\t{f.filename:<20}{len(chunks):>10}")