from embedder import embed, cosine_similarity 

def semantic_chunk(content: str, threshold: float):
    """
    Splits the 'content' string in chunks on every "\n" and every "." 
    The string is chunked everytime the similarity between a phrase and 
    the previous one is lower than the threshold
    """
    phrases = []

    for row in content.split("\n"):
        for phrase in row.split(". "):
            if phrase.strip():
                phrases.append(phrase.strip())

    if not phrases:
        return []

    vectors = embed(phrases)
    chunks = []
    current = [phrases[0]]

    for i in range(1, len(phrases)):
        similarity = cosine_similarity(vectors[i - 1], vectors[i])
        if similarity < threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(phrases[i])
    chunks.append(" ".join(current))

    return chunks