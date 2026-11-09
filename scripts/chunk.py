"""chunk.py — the three chunking strategies.

These are the functions you wrote in Module 2 · Lesson 10 (fixed, recursive) plus the one
the instructor showed in the demo (semantic), promoted into aikit/ — the same
path taken by embeddings.py, search.py and vectorstore.py. They are NOT to be touched,
just imported:

    from aikit.chunk import chunk_fixed, chunk_recursive, carica_documenti
"""

from embeddings import embed, cosine_similarity

import sys

def chunk_fixed(text, size, overlap):
    """Blocks of `size` characters; each block starts `overlap` characters
    before the end of the previous one. Returns the list of pieces."""
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + size])
        start = start + size - overlap

    return chunks

def chunk_recursive(text, size):
    """Cuts at the LAST separator that fits within `size` characters, trying
    separators from strongest to weakest (paragraph, line, sentence); if
    none fits, hard cut at `size`. Then starts over on the rest. Returns
    pieces at most `size` long."""

    sys.setrecursionlimit(10000)

    if len(text) <= size:
        return [text]
    window = text[:size]

    for separator in ("\n\n", "\n", ". "):
        position = window.rfind(separator)

        if position > 0:
            chunk = text[:position]
            remaining = text[position + len(separator):]
            return [chunk] + chunk_recursive(remaining, size)

    return [window] + chunk_recursive(text[size:], size)

def chunk_semantic(text, threshold):
    """Cuts where the TOPIC changes: embeds the sentences one by one and closes
    the chunk when the similarity between a sentence and the next one drops below
    `threshold` — a drop in similarity means a change of topic."""

    phrases = []

    for row in text.split("\n"):
        for phrase in row.split(". "):
            if phrase.strip():
                phrases.append(phrase.strip())

    vectors = embed(phrases)

    chunks = []
    current = [phrases[0]]

    for i in range(1, len(phrases)):
        similar = cosine_similarity(vectors[i - 1], vectors[i])

        if similar < threshold:
            chunks.append(" ".join(current))
            current = []

        current.append(phrases[i])
    chunks.append(" ". join(current))
    
    return chunks

def show_chunks(chunks):
    """How the chopping turned out: size and boundaries of each chunk."""
    for i, chunk in enumerate(chunks):
        clean = " ".join(chunk.split())
        print(f"  {i:3d} · {len(chunk):4d} chars · "
              f"{clean[:34]} ⟨…⟩ {clean[-34:]}")
