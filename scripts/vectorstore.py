from config import CHROMADB_DIR, COLLECTION_NAME
from chromadb.api.models.Collection import Collection
from embeddings import embed

import chromadb

def open_collection(name: str) -> Collection:
    """The on-disk collection: creates it the first time, then reopens it."""

    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    return client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )

def create_collection():
    """Like open_collection(), but from SCRATCH: if the collection exists, it is deleted.

    Used by `prepara_collection.py` to rebuild the store and, in Module 2 · Lesson 12,
    by `reindicizza_chunks` (Lab 2) when changing strategy/size. The
    retrieval step (`search`) does NOT call it: the collection must be reopened with
    `open_collection`, not recreated on every question.
    """
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def search(collection, query, k, backend="openai"):
    """The k documents most similar to the query, most similar first.

    Each result is a dict {"id", "text", "score", ...metadata}: the query
    is embedded in the same space as the corpus (rule from Module 2 · Lesson 8) and the DB
    answers in DISTANCES — the score is 1 - distance, as in Module 2 · Lesson 9. The "id"
    (e.g. "manuale-lumadesk-007") is the source to cite in the RAG prompt.
    """
    embedded_query = embed([query], backend=backend)[0]
    response = collection.query(query_embeddings=[embedded_query], n_results=k)

    results = []

    for i in range(len(response["ids"][0])):
        result = {
            "id": response["ids"][0][i],
            "text": response["documents"][0][i],
            "score": 1 - response["distances"][0][i]
        }
        if response["metadatas"][0][i]:
            result.update(response["metadatas"][0][i])
        results.append(result)
    return results
# TODO: copy remaining functions
