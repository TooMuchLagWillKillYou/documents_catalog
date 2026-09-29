# documents-catalog

A small retrieval pipeline for a personal document collection. It reads documents in several formats, cleans the text, splits it into semantic chunks, embeds the chunks, and indexes them in a local [ChromaDB](https://www.trychroma.com/) store. The goal is a catalog you can query in natural language, with an LLM answering from the retrieved passages.

> **Status:** work in progress. Reading, cleaning, semantic chunking, embedding and indexing run end to end from `main.py`. There is no query step yet, and the question-answering step with an LLM doesn't exist.

## Pipeline

```
docs/*  ──►  reader  ──►  cleaner  ──►  chunker  ──►  embedder  ──►  vectorstore (./chroma)
           (to text)    (clean text)  (semantic chunks) (vectors)     (index / search)
```

| Module | Purpose |
| --- | --- |
| [scripts/reader.py](scripts/reader.py) | Loads each file into a `Document` (`filename`, `path`, `content`, `metadata`). Supports `.pdf`, `.txt`, `.csv`, `.docx`, `.html`/`.htm` and `.md`/`.markdown`. Files that can't be read are collected and reported, not raised. |
| [scripts/cleaner.py](scripts/cleaner.py) | Per-format text cleanup. For PDFs it removes repeated headers and footers, drops page numbers (`12`, `pagina 3 di 10`), fixes words hyphenated across lines, normalizes whitespace and rejoins broken paragraphs. |
| [scripts/chunker.py](scripts/chunker.py) | `split_phrases` splits text into sentences on line breaks and after `.`, `!` or `?` (phrases longer than `MAX_CHARS` are cut on a space). `semantic_chunk` then joins consecutive phrases into chunks, and starts a new chunk whenever the cosine similarity between two consecutive phrases falls below a threshold or the chunk would exceed `MAX_CHARS`. |
| [scripts/embedder.py](scripts/embedder.py) | Embeddings from two backends: `openai` (`text-embedding-3-large`, 3072 dimensions, sent in batches of at most 300 inputs and 250,000 characters, logs token cost) or `local` (`all-MiniLM-L6-v2` from sentence-transformers, 384 dimensions, runs on CPU). |
| [scripts/vectorstore.py](scripts/vectorstore.py) | A persistent ChromaDB store in `./chroma` using cosine distance. Provides `create_collection` (recreates the collection from scratch), `open_collection`, `index` (upsert) and `search` (returns `id`, `document`, `score` and the stored metadata). |
| [scripts/config.py](scripts/config.py) | Shared settings: directories, chunking threshold and size limit, embedding models and OpenAI batching limits. |
| [scripts/main.py](scripts/main.py) | Entry point that runs the whole pipeline on `docs/` and indexes the chunks into the `my_collection` collection. It embeds with the `openai` backend. |

## Setup

You need Python 3.12 or newer, because the code uses nested quotes inside f-strings.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r scripts/requirements.txt
```

To use the OpenAI backend, create a `.env` file in the project root. It's already in `.gitignore`.

```
OPENAI_API_KEY=sk-...
```

The local backend needs no key. It downloads the model (about 88 MB) the first time it runs.

## Usage

1. Put your documents in `docs/` at the project root. This folder is gitignored, so your documents stay out of the repo. Create the `out/` folder too (also gitignored), because the pipeline writes its intermediate files there.
2. Run the pipeline:

   ```bash
   python scripts/main.py
   ```

   You can also start **Python Debugger: main.py** from VS Code (configured in `.vscode/launch.json`).

Each stage prints a table of files with their character or chunk counts, then a list of the files that couldn't be read and why. After each step (reading, cleaning, splitting, chunking) the text of every file is also written to `out/` as `_<step number>_output_<step>_<filename>.md`, so you can inspect what each stage produced.

### Tuning

- `THRESHOLD` in `scripts/config.py` (default `0.4`) controls chunk size. A higher value splits more often, which gives smaller and more focused chunks.
- `MAX_CHARS` in `scripts/config.py` (default `2000`) is a hard limit on phrase and chunk length. It keeps them well under the embedding model's 8,191-token input limit.
- `embed(..., backend="openai" | "local")` selects the embedding backend, and `main.py` currently uses the default, `openai`. Always use the same backend to index and to query, because the vectors from the two models aren't compatible.

## Known limitations

- `.doc` files are read with `python-docx`. This works when the file is really in `.docx` format with a `.doc` extension, as some tools save them. True legacy Word 97–2003 `.doc` files can't be read and are listed under FAILED FILES.
- Scanned PDFs without a text layer produce empty content, because there is no OCR yet.
- Sentences are split naively on `.`, `!` or `?` followed by whitespace, and on line breaks, so abbreviations like "e.g. " create extra splits.
- `create_collection` deletes any existing collection with the same name, so every run rebuilds the index from scratch.
- `main.py` embeds every phrase to find chunk boundaries and then embeds the chunks again, so with the OpenAI backend each document is paid for twice.
- `out/` must exist before you run the pipeline.

## Roadmap

- Local models for both embedding and retrieval
- OCR fallback for files without extractable text
- Importing documents from cloud storage (OneDrive, Google Drive)
- A query interface that answers from the retrieved chunks with an LLM
