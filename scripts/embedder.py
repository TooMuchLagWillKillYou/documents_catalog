import numpy as np
from config import CLOUD_EMBEDDING_MODEL, OPENAI_PRICE_PER_M, OPENAI_BATCH_SIZE, OPENAI_BATCH_CHARS, LOCAL_EMBEDDING_MODEL

def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=np.float32)
    vector_b = np.array(vector_b, dtype=np.float32)

    result = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    return float(result)

def _openai_batches(content: list[str]):
    """
    Groups 'content' in batches of at most OPENAI_BATCH_SIZE inputs and OPENAI_BATCH_CHARS chars
    """
    batch, chars = [], 0
    for text in content:
        if batch and (len(batch) == OPENAI_BATCH_SIZE or chars + len(text) > OPENAI_BATCH_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch

def embed_openai(content: list[str]):
    from openai import OpenAI

    client = OpenAI()
    vectors, tokens = [], 0
    for batch in _openai_batches(content):
        r = client.embeddings.create(model=CLOUD_EMBEDDING_MODEL, input=batch)
        tokens += r.usage.total_tokens
        vectors.extend(d.embedding for d in r.data)

    cost_usd = tokens / 1_000_000 * OPENAI_PRICE_PER_M
    print(f"\t[openai] Embedded {len(content)} inputs, {tokens} tokens -> ${cost_usd:.6f}")

    return np.array(vectors, dtype=np.float32)

_local_embedding_model = None

def embed_local(content: list[str]):
    global _local_embedding_model

    if _local_embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return np.asarray(_local_embedding_model.encode(content), dtype=np.float32)    


def embed(content: list[str], backend: str = "openai") -> np.ndarray[np.floating]:
    if backend == "openai":
        return embed_openai(content)
    elif backend == "local":
        return embed_local(content)
    else:
        raise ValueError(f"{__file__}: unkown backend \"{backend}\"")