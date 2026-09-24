import chromadb 
from embedder import embed
from pathlib import Path

CHROMA_DIR: str = Path(__file__).parent.parent / "chroma"
UPSERT_BATCH_SIZE = 5000                  # chroma rejects upserts larger than ~5461 records

def create_collection(name: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def open_collection(name: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def index(collection, ids, documents, vectors, metadatas=None):
    for i in range(0, len(ids), UPSERT_BATCH_SIZE):
        batch = slice(i, i + UPSERT_BATCH_SIZE)
        collection.upsert(
            ids=ids[batch],
            documents=documents[batch],
            embeddings=vectors[batch],
            metadatas=metadatas[batch] if metadatas else None,
        )


def search(collection, query, k, backend="openai"):
    q = embed([query], backend=backend)[0]
    response = collection.query(query_embeddings=[q], n_results=k)
    results = []

    for i in range(len(response["ids"][0])):
        r =  {
            "id": response["ids"][0][i],
            "document": response["documents"][0][i],
            "score": 1 - response["distances"][0][i]
        }
        if response["metadatas"][0][i]:
            r.update(response["metadatas"][0][i])
        results.append(r)

    return results