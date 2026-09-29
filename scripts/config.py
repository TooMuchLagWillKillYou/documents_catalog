from pathlib import Path

import re

SCRIPTS_DIR = Path(__file__).parent
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
OUTPUT_DIR = SCRIPTS_DIR.parent / "out"
THRESHOLD = 0.4

NOISE = ["script", "style", "header", "nav", "aside", "footer"]

PAGE_NUMBER = re.compile(r"\s*(pagina\s+)?\d+(\s+di\s+\d+)?\s*", re.I)

MAX_CHARS = 2000                                # ~500 tokens, well under the embedding model's 8,191-token input limit
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

OPENAI_MODEL = "text-embedding-3-large"         # 3072 dim — $0.13 / 1M token
OPENAI_PRICE_PER_M = 0.13
OPENAI_BATCH_SIZE = 300                         # max inputs per request
OPENAI_BATCH_CHARS = 250_000                    # max chars per request: a token is at least ~1 char, so this stays under the 300k token limit
LOCAL_MODEL = "all-MiniLM-L6-v2"                # 384 dim — ~88 MB, runs on CPU

COLLECTION_NAME = "documents_catalog"

K = 4