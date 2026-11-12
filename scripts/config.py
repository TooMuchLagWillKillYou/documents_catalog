from pathlib import Path

import re

# general
SCRIPTS_DIR = Path(__file__).parent
DOCS_DIR = SCRIPTS_DIR.parent / "docs"
OUTPUT_DIR = SCRIPTS_DIR.parent / "out"
SESSIONS_DIR = SCRIPTS_DIR.parent / "sessions"

# chromadb
CHROMADB_DIR = Path(__file__).parent.parent / "chroma"
COLLECTION_NAME = "documents_catalog"

# search
K = 4
RESPONSE_MODEL = "gpt-4o-mini"
CHAT_TOKEN_BUDGET = 350                                      # TODO: tweak this param
INSTRUCTIONS_TO_GENERATE_RESPONSE = """Answer the question using ONLY the information in the documents
in the message. Cite the source in square brackets, for example:
[policy-resi-garanzia-002]. If the answer is not in the documents, reply
exactly: "I can't find it in the documents." — without making anything up."""
MEMORY = "buffer",                                          # "buffer" or "summary"

# cleaning
NOISE = ["script", "style", "header", "nav", "aside", "footer"]
PAGE_NUMBER = re.compile(r"\s*(pagina\s+)?\d+(\s+di\s+\d+)?\s*", re.I)

# chunking
MAX_CHARS = 2000                                            # ~500 tokens, well under the embedding model's 8,191-token input limit
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
THRESHOLD = 0.4

# embeddings
CLOUD_EMBEDDING_MODEL = "text-embedding-3-large"            # 3072 dim — $0.13 / 1M token
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"                  # 384 dim — ~88 MB, runs on CPU
OPENAI_PRICE_PER_M = 0.13
OPENAI_BATCH_SIZE = 1000                                    # max inputs per request
OPENAI_BATCH_CHARS = 250_000                                # max chars per request: a token is at least ~1 char, so this stays under the 300k token limit
