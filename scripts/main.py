from pathlib import Path
from reader import read_directory
from cleaner import clean
from chunker import semantic_chunk
from dotenv import load_dotenv
from embedder import embed, embed_for_semantic_chuking
from vectorstore import create_collection, index

load_dotenv()

SCRIPTS = Path(__file__).parent
DOCS_DIRECTORY = SCRIPTS / "docs"
THRESHOLD = 0.4

if __name__ == "__main__":

    files, _ = read_directory(DOCS_DIRECTORY)
    cleaned_files = clean(files)

    print(f"FILES CHUNKED")
    print(f"\t{"FILENAME":<20}{"CHUNKS":>10}")

    ids, documents, vectors, metadatas = [], [], [], []
    for f in cleaned_files:

        sentence_vectors = embed_for_semantic_chuking(f.content)
        chunks = semantic_chunk(f.content, THRESHOLD, sentence_vectors)
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
