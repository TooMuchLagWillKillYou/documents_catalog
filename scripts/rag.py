import chromadb

from config import DOCS_DIR, COLLECTION_NAME, MAX_CHARS, K, RESPONSE_MODEL, INSTRUCTIONS_TO_GENERATE_RESPONSE
from loaders import load
from clean import clean, pipeline_for
from chunk import chunk_recursive
from embeddings import embed
from pathlib import Path
from vectorstore import create_collection, open_collection, search
from openai import OpenAI

client = OpenAI()

def print_chunks(chunks):
    """I chunk recuperati, uno per riga: posizione, id, score, anteprima."""
    for position, chunk in enumerate(chunks, start=1):
        preview = " ".join(chunk["text"].split())[:58]
        print(f"\t{position}. {chunk["id"][:24]}…\tscore={chunk["score"]:.3f}\t{preview}…")

def retrieve(query, k=K):
    """The retrieve step: the k most relevant chunks for the query, most similar first.

    Each chunk is a dict {"id", "text", "score", "source"}. If k is not
    given, it defaults to K from config. The collection is reopened in here with
    vectorstore.open_collection(COLLECTION_NAME).
    """

    collection = open_collection(COLLECTION_NAME)
    return search(collection, query, k)

def generate_response(chunks, question, chat_history=None):
    """The generate step on its own: builds the RAG prompt and calls the model, with
    the rules in INSTRUCTIONS_TO_GENERATE_RESPONSE. Returns the response text.

    chat_history is the list of previous messages (user/assistant):
    it goes in the input BEFORE the current turn's message, so the
    model sees the earlier turns. Without history it is single-turn.
    """
    documents = ""

    for chunk in chunks:
        documents += f"""
            <document source='{chunk["id"]}'>
                {chunk["text"]}
            </document>
        """

    prompt = f"""
        <documents>
            {documents}
        </documents>
        
        <question>
            {question}
        </question>
    """

    if chat_history is None:
        chat_history = []

    response = client.responses.create(
        model=RESPONSE_MODEL,
        instructions=INSTRUCTIONS_TO_GENERATE_RESPONSE,
        input = chat_history + [{"role": "user", "content": prompt}]
    )

    return response.output_text

def index():
    """The ingest: load → clean → chunk → embed → upsert, over the whole corpus.

    Recreates the collection from SCRATCH (create_collection): re-running
    does not duplicate the chunks. Prints the per-document count and returns
    the total number of indexed chunks.
    """
    collection = create_collection()
    ids, texts, metadatas = [], [], []
    indexed_documents = 0

    for path in sorted(DOCS_DIR.glob("*")):

        try:
            extension = Path(path).suffix.lower()

            documents = load(str(path))
            raw_content = "\n".join(d.content for d in documents)
            clean_content = clean(raw_content, pipeline_for(str(path)))
            chunks = chunk_recursive(clean_content, MAX_CHARS)

            for i, chunk in enumerate(chunks):
                ids.append(f"{path.stem}-{i:03d}")
                texts.append(chunk)
                metadatas.append({"source": path.name})

            indexed_documents += 1
            print(f"\t{path.stem[:24]:<28}{extension:<8}{len(chunks)} chunks")
        except ValueError as e:
            print(e)
    
    vectors = embed(texts)
    collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=metadatas)

    print(f"indexed {len(texts)} chunks from {indexed_documents} "
          f"documents into '{COLLECTION_NAME}'")
    return len(texts)

if __name__ == "__main__":
    index()