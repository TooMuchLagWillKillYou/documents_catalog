from embedder import embed, cosine_similarity 

def semantic_chunk(content: str, threshold: float, vectors, max_chars: int = 2000):
    """
    Splits the 'content' string in chunks on every "\n" and every "."
    The string is chunked everytime the similarity between a phrase and
    the previous one is lower than the threshold, or when adding the
    phrase would make the chunk longer than 'max_chars'
    """
    phrases = []

    for row in content.split("\n"):
        for phrase in row.split(". "):
            if phrase.strip():
                phrases.append(phrase.strip())

    if not phrases:
        return []

    # vectors = embed(phrases)
    chunks = []
    current = [phrases[0]]
    length = len(phrases[0])

    for i in range(1, len(phrases)):
        similarity = cosine_similarity(vectors[i - 1], vectors[i])
        too_long = length + 1 + len(phrases[i]) > max_chars
        if similarity < threshold or too_long:
            chunks.append(" ".join(current))
            current = []
            length = 0
        current.append(phrases[i])
        length += len(phrases[i]) + (1 if length else 0)
    chunks.append(" ".join(current))

    # a single phrase can still exceed the limit: hard-split it
    return [c[j:j + max_chars] for c in chunks for j in range(0, len(c), max_chars)]