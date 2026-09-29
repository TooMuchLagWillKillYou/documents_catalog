import numpy as np

OPENAI_MODEL = "text-embedding-3-large"   # 3072 dim — $0.13 / 1M token
OPENAI_PRICE_PER_M = 0.13
OPENAI_BATCH_SIZE = 300                    # max inputs per request
OPENAI_BATCH_CHARS = 250_000               # max chars per request: a token is at least ~1 char, so this stays under the 300k token limit
LOCAL_MODEL = "all-MiniLM-L6-v2"          # 384 dim — ~88 MB, runs on CPU

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
        r = client.embeddings.create(model=OPENAI_MODEL, input=batch)
        tokens += r.usage.total_tokens
        vectors.extend(d.embedding for d in r.data)

    cost_usd = tokens / 1_000_000 * OPENAI_PRICE_PER_M
    print(f"\t[openai] Embedded {len(content)} inputs, {tokens} tokens -> ${cost_usd:.6f}")

    return np.array(vectors, dtype=np.float32)

_local_model = None

def embed_local(content: list[str]):
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(LOCAL_MODEL)
    return np.asarray(_local_model.encode(content), dtype=np.float32)    


def embed(content: list[str], backend: str = "openai") -> np.ndarray[np.floating]:
    if backend == "openai":
        return embed_openai(content)
    elif backend == "local":
        return embed_local(content)
    else:
        raise ValueError(f"{__file__}: unkown backend \"{backend}\"")