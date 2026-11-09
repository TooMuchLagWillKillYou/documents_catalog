"""clean.py — clean raw text before chunking/embedding.

The text coming out of the loaders is **raw**: boilerplate, broken lines,
random whitespace, repeated headers, page numbers. If we feed it as-is to
chunking/embedding, the *noise* carries through the whole pipeline (dirty chunks
→ dirty embeddings → worse retrieval).

Core idea: **pure `str -> str` functions** (each step does ONE thing) + a small runner that composes them.
No framework: a pipeline is just a list of functions applied in order. The instructor picks different steps
for different documents (a Gutenberg `.txt` doesn't have the same noise as a PDF).

    from loaders import load
    from clean import clean, pipeline_for

    path = "book.txt"
    text = load(path)[0].content
    cleaned = clean(text, pipeline_for(path))

Parametric steps (thresholds, heading lists) are **factories**: they return
an already-configured `str -> str` step, so the pipeline stays a list of
uniform functions.
"""
from pathlib import Path
from collections import Counter
import unicodedata
import re


def _normalize_unicode(text: str) -> str:
    """Canonical Unicode form (NFC): 'e´' → 'é', composed accents made uniform."""
    return unicodedata.normalize("NFC", text)

def _normalize_whitespace(text: str) -> str:
    """Multiple spaces/tabs → one; no trailing spaces; at most one blank line."""
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def _dehyphenate(text: str) -> str:
    """Rejoin words hyphenated at the end of a line: 'lit-\\nerature' → 'literature'.

    Typical of PDFs with fixed-width justified layout. We only join when there
    are letters before and after the hyphen (we leave 'e-mail' or '2019-2020' alone).
    """
    return re.sub(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])", r"\1\2", text)

def _reflow_paragraphs(text: str) -> str:
    """Rebuild paragraphs that were wrapped at ~70 characters (Gutenberg & co.).

    A single '\\n' inside a paragraph is a layout line break, not a real
    break: we turn it into a space. The blank line (paragraph break) stays.
    This way the text "flows" again.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    recomposed = [" ".join(p.split()) for p in paragraphs if p.strip()]
    return "\n\n".join(recomposed)

def _remove_page_numbers(text: str) -> str:
    """Remove lines made up of just a number (PDF page numbers)."""
    return "\n".join(l for l in text.split("\n") if not l.strip().isdigit())

def _remove_repeated_lines(min_repeats: int = 3, max_len: int = 80):
    """Remove (short) lines that repeat: recurring headers/footers.

    E.g. 'Published as a conference paper at ICLR 2019' on every page. We count
    the lines: if a short line appears at least `min_repeats` times, it is almost
    certainly a layout element, not content.
    """
    def step(text: str) -> str:
        count = Counter(l.strip() for l in text.split("\n") if l.strip())
        repeated = {r for r, n in count.items() if n >= min_repeats and len(r) <= max_len}
        return "\n".join(l for l in text.split("\n") if l.strip() not in repeated)
    
    return step

def _remove_citation_markers():
    """Strip references like '[12]' (footnotes/citations in web pages)."""
    return lambda text: re.sub(r"\[\d+\]", "", text)

def _keep_between_markers(start: str, end: str):
    """Keep only the body between two markers (Project Gutenberg boilerplate).

    E.g. start='*** START', end='*** END': drops the license, credits and metadata
    that sit before/after the actual text.
    """
    def step(text: str) -> str:
        lines = text.split("\n")
        i = next((k for k, l in enumerate(lines) if start in l), None)
        j = next((k for k, l in enumerate(lines) if end in l), None)

        if i is None:
            i = -1
        if j is None:
            j = len(lines)

        return "\n".join(lines[i + 1:j])
    
    return step

def _drop_sections(headings: list):
    """Truncate the text at the first 'trailing' section.

    Sections like Notes, Bibliography, External links, See also…: typical
    Wikipedia boilerplate that adds nothing to the content.
    """

    def step(text: str) -> str:
        lines = text.split("\n")
        
        for k, l in enumerate(lines):
            if l.strip() in headings:
                return "\n".join(lines[:k]).strip()
        return text
    return step


PIPELINES = {
    ".txt": [
        _keep_between_markers("*** START", "*** END"), # TODO: understand these params
        _normalize_unicode,
        _reflow_paragraphs
    ],
    ".pdf": [
        _dehyphenate,
        _remove_repeated_lines(min_repeats=3),
        _remove_page_numbers,
        _normalize_whitespace
    ],
    ".html": [
        _remove_citation_markers,
        _drop_sections(["Note", "Bibliography", "Correlated links", "External links"]),
        _normalize_whitespace
    ]
}

DEFAULT_STEPS = [_normalize_unicode, _normalize_whitespace]

def clean(text: str, steps: list) -> str:
    """Apply the steps in order. A pipeline is just a list of functions."""
    for step in steps:
        text = step(text)
    
    return text

def pipeline_for(path: str) -> list:
    """The right cleaning recipe for the file's extension."""
    return PIPELINES.get(Path(path).suffix.lower(), DEFAULT_STEPS)   
