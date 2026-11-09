"""embeddings.py — from text to vector, with a selectable backend.

The module that powers the whole retrieval block:

    from embeddings import embed, cosine_similarity

    vecs = embed(["text one", "text two"], backend="openai")
    print(cosine_similarity(vecs[0], vecs[1]))

Two backends behind the same signature:
  - "openai"  → text-embedding-3-small API (paid, the texts leave your machine)
  - "local"   → sentence-transformers on your machine (free, offline)

Re-running the same query pays for the embedding again — we're talking $0.0000002.
"""
from config import LOCAL_MODEL, OPENAI_MODEL, OPENAI_PRICE_PER_M, OPENAI_BATCH_SIZE, OPENAI_BATCH_CHARS
from dotenv import load_dotenv

import numpy as np

load_dotenv()

def cosine_similarity(vector_a, vector_b):
    """Cosine similarity between two vectors: cos(θ) = a·b / (|a| |b|).

    It's 1.0 for parallel vectors (same meaning), ~0 for orthogonal
    vectors (no relation), -1.0 for opposite vectors. What counts is the ANGLE,
    not the length: that's why there's no need to normalize first.
    """
    vector_a = np.asanyarray(vector_a, dtype=np.float64)
    vector_b = np.asanyarray(vector_b, dtype=np.float64)

    return float((vector_a @ vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))

def _openai_batches(texts):
    """Yield batches of at most OPENAI_BATCH_SIZE inputs and OPENAI_BATCH_CHARS
    chars, so each request stays under the API's per-request limits."""
    batch, chars = [], 0

    for text in texts:
        if batch and (len(batch) == OPENAI_BATCH_SIZE or chars + len(text) > OPENAI_BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)

    if batch:
        yield batch

def _embed_openai(texts):
    """Embedding via the OpenAI API. The texts are SENT to the service."""
    from openai import OpenAI

    client = OpenAI()
    vectors, tot_tokens = [], 0

    for batch in _openai_batches(texts):
        response = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        tot_tokens += response.usage.total_tokens
        vectors.extend(d.embedding for d in response.data)

    cost = tot_tokens / 1_000_000 * OPENAI_PRICE_PER_M

    print(f"\t[openai] {len(texts)} texts, {tot_tokens} token → ${cost:.6f}")

    return np.asarray(vectors, dtype=np.float32)

_local_model = None

def _embed_local(texts):
    """Embedding with sentence-transformers, on YOUR machine.

    The first use downloads the model (~88 MB) into the Hugging Face cache;
    from then on everything is offline and free.
    """
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(LOCAL_MODEL)
    return np.asarray(_local_model.encode(texts), dtype=np.float32)

def embed(texts, backend="openai"):
    """Embed a list of texts with the chosen backend."""
    if backend == "openai":
        return _embed_openai(texts)
    if backend == "local":
        return _embed_local(texts)
    raise ValueError(f"Unknown {backend!r} backend: 'openai' or 'local'")