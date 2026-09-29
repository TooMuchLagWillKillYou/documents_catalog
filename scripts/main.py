from config import OUTPUT_DIR, DOCS_DIR, THRESHOLD
from reader import read_directory
from cleaner import clean
from chunker import semantic_chunk, split_phrases
from dotenv import load_dotenv
from embedder import embed
from vectorstore import create_collection, index

load_dotenv()

def _output(files: list[dict[str, str]], order: int, step: str):
    for file in files:
        with open(OUTPUT_DIR / f"_{order}_output_{step}_{file["filename"]}.md", "w", encoding="utf-8") as f:
            f.write(file["content"])


if __name__ == "__main__":

    files, _ = read_directory(DOCS_DIR)
    _output([{"filename": f.filename, "content": f.content} for f in files], 1, "reading")
    cleaned_files = clean(files)
    _output([{"filename": f.filename, "content": f.content} for f in cleaned_files], 2, "cleaning")

    print(f"FILES CHUNKED")
    print(f"\t{"FILENAME":<20}{"CHUNKS":>10}")

    ids, documents, vectors, metadatas = [], [], [], []
    for f in cleaned_files:

        phrases = split_phrases(f.content)
        _output([{"filename": f.filename, "content": "\n\n".join(phrases)}], 3, "splitting")
        chunks = semantic_chunk(phrases, THRESHOLD, embed(phrases)) if phrases else []
        _output([{"filename": f.filename, "content": "\n\n".join(chunks)}], 4, "chunking")

        print(f"\t{f.filename[:20]:<20}{len(chunks):>10}")
        if not chunks:
            continue

        vectors.extend(embed(chunks))
        for i, c in enumerate(chunks):
            ids.append(f"{f.filename}-{i:03d}")
            documents.append(c)
            metadatas.append({"doc": f.filename})

    collection = create_collection("my_collection")
    index(collection, ids, documents, vectors, metadatas)
