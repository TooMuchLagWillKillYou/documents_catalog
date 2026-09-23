import numpy as np

OPENAI_MODEL = "text-embedding-3-small"   # 1536 dim — $0.02 / 1M token
LOCAL_MODEL = "all-MiniLM-L6-v2"          # 384 dim — ~88 MB, runs on CPU

def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=np.float32)
    vector_b = np.array(vector_b, dtype=np.float32)

    result = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

    return float(result)

def embed_openai(content: str):
    from openai import OpenAI

    r = OpenAI().embeddings.create(model=OPENAI_MODEL, input=content)
    tokens = r.usage.total_tokens
    cost_usd = tokens / 1_000_000 * 0.02
    print(f"\t[openai] Embedded {len(content)} chars, {tokens} tokens -> ${cost_usd:.6f}")

    return np.array([d.embedding for d in r.data], dtype=np.float32)

_local_model = None

def embed_local(content: str):
    global _local_model

    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(LOCAL_MODEL)
    return np.asarray(_local_model.encode(content), dtype=np.float32)    


def embed(content: str, backend: str = "openai") -> np.ndarray[np.floating]:
    if backend == "openai":
        return embed_openai(content)
    elif backend == "local":
        return embed_local(content)
    else:
        raise ValueError(f"{__file__}: unkown backend \"{backend}\"")