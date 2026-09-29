import re
from embedder import cosine_similarity
from config import MAX_CHARS, SENTENCE_END


def split_phrases(content: str, max_chars: int = MAX_CHARS) -> list[str]:
    """
    Splits the 'content' string in phrases on every "\n" and after every ".", "!" or "?"
    followed by a space. The punctuation stays in the phrase, and the last phrase of each
    row ends with "\n" so that _join can restore the paragraph break.
    Phrases longer than 'max_chars' are split on the last space before the limit,
    so every phrase can be embedded on its own
    """
    phrases = []

    for row in content.split("\n"):
        row_phrases = []
        for phrase in SENTENCE_END.split(row):
            phrase = phrase.strip()
            while len(phrase) > max_chars:
                cut = phrase.rfind(" ", 0, max_chars + 1)
                if cut <= 0:
                    cut = max_chars
                row_phrases.append(phrase[:cut].strip())
                phrase = phrase[cut:].strip()
            if phrase:
                row_phrases.append(phrase)
        if row_phrases:
            row_phrases[-1] += "\n"
            phrases.extend(row_phrases)

    return phrases

def _join(phrases: list[str]) -> str:
    """
    Joins phrases from the same row with a space and phrases that end a row with "\n"
    """
    return "".join(p if p.endswith("\n") else p + " " for p in phrases).strip()

def semantic_chunk(phrases: list[str], threshold: float, vectors, max_chars: int = MAX_CHARS):
    """
    Joins consecutive 'phrases' (from split_phrases) into chunks. 'vectors[i]' is the
    embedding of 'phrases[i]'. A new chunk starts everytime the similarity between a
    phrase and the previous one is lower than the threshold, or when adding the
    phrase would make the chunk longer than 'max_chars'
    """
    if not phrases:
        return []

    chunks = []
    current = [phrases[0]]
    length = len(phrases[0])

    for i in range(1, len(phrases)):
        similarity = cosine_similarity(vectors[i - 1], vectors[i])
        too_long = length + 1 + len(phrases[i]) > max_chars
        if similarity < threshold or too_long:
            chunks.append(_join(current))
            current = []
            length = 0
        current.append(phrases[i])
        length += len(phrases[i]) + (1 if length else 0)
    chunks.append(_join(current))

    return chunks
