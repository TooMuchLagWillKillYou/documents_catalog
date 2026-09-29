from config import K, COLLECTION_NAME
import vectorstore
from dotenv import load_dotenv

load_dotenv()

def _find_chunks(q: str):
    collection = vectorstore.open_collection(COLLECTION_NAME)
    return vectorstore.search(collection, q, K)

if __name__ == "__main__":
    TEST_QUERIES = [
        "Who's running the FCAS program?",
        "What is the internet?"
    ]

    print("ASKING")
    # print(f"{"ID":<10}{}")
    for q in TEST_QUERIES:
        print(f"\tAsking \"{q}\"")
        chunks = _find_chunks(q)
        print(f"Chunks: {[c for c in chunks]}")
