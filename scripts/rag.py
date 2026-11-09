import chromadb

from config import CHROMADB_DIR, DOCS_DIR, COLLECTION_NAME, MAX_CHARS
from loaders import load
from clean import clean, pipeline_for
from chunk import chunk_recursive
from embeddings import embed

def create_collection():
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def index():
    """The ingest: load → clean → chunk → embed → upsert, over the whole corpus.

    Recreates the collection from SCRATCH (create_collection): re-running
    does not duplicate the chunks. Prints the per-document count and returns
    the total number of indexed chunks.
    """
    collection = create_collection()
    ids, texts, metadatas = [], [], []
    indexed_documents = 0

    for path in sorted(DOCS_DIR.glob("*")):

        try:
            documents = load(str(path))
            raw_content = "\n".join(d.content for d in documents)
            clean_content = clean(raw_content, pipeline_for(str(path)))
            chunks = chunk_recursive(clean_content, MAX_CHARS)

            for i, chunk in enumerate(chunks):
                ids.append(f"{path.stem}-{i:03d}")
                texts.append(chunk)
                metadatas.append({"source": path.name})

            indexed_documents += 1
            print(f"\t{path.stem}: {len(chunks)} chunks")
        except ValueError as e:
            print(e)
    
    vectors = embed(texts)
    collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=metadatas)

    print(f"indexed {len(texts)} chunks from {indexed_documents} "
          f"documents into '{COLLECTION_NAME}'")
    return len(texts)


if __name__ == "__main__":
    index()