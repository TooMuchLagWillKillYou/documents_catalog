from config import COLLECTION_NAME, SESSIONS_DIR, RESPONSE_MODEL, CHAT_TOKEN_BUDGET, MEMORY
from vectorstore import open_collection
from rag import index, retrieve, generate_response, print_chunks
from dotenv import load_dotenv
from openai import OpenAI

import json
import tiktoken

load_dotenv()

client = OpenAI() 
_encoder = tiktoken.get_encoding("o200k_base")

def _load_session(session_name):
    """Returns the history saved under that name, or an empty list if the
    session doesn't exist yet."""

    path = SESSIONS_DIR / f"{session_name}.json"

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _save_session(session_name, chat_history):
    """Scrive la storia in SESSIONS_DIR/<nome>.json (la cartella
    si crea se manca)."""

    dir = SESSIONS_DIR
    dir.mkdir(exist_ok=True)

    with open(dir / f"{session_name}.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def _token_count(chat_history):
    return sum(len(_encoder.encode(m["content"])) for m in chat_history)

def _compact(chat_history, user_question):
    """Query condensation: rewrites the question using the history so that
    it stands on its own (names instead of pronouns, the topic of the
    conversation inside the question). One call to the model; returns the
    rewritten question as a string. If the question is already
    self-contained, the model leaves it as is."""

    chat = ""

    for text in chat_history:
        chat += f"{text["role"]}: {text["content"]}\n"

    prompt = f"""
    Below is a conversation and a new question from the user.
    Rewrite the new question so that it is understandable ON ITS OWN, without the
    conversation: replace pronouns and references ("that one", "the second
    case", "and for the wide one?") with what they refer to. If the question is
    already self-contained, copy it unchanged. Reply ONLY with the rewritten question.

    <conversation>
        {chat}
    </conversation>

    <new_question>
        {user_question}
    </new_question>
    """

    response = client.responses.create(
        model=RESPONSE_MODEL,
        input=[{"role": "user", "content": prompt}]
    )

    return response.output_text.strip()

def _truncate(chat_history):
    """Buffer memory: removes the oldest messages
    until the history fits in CHAT_TOKEN_BUDGET. Modifies the list in
    place and returns how many messages it removed."""
    
    removed_items = 0

    while len(chat_history) > 2 and _token_count() > CHAT_TOKEN_BUDGET:
        chat_history.pop(0)
        removed_items += 1

    return removed_items

def _summarise(chat_history):
    """Summary memory: if the history exceeds CHAT_TOKEN_BUDGET, all
    messages except the last two (the last turn) are replaced by
    ONE summary message written by the model — a "user" message
    that starts with «Summary of the conversation so far: …», followed by an
    acknowledging "assistant" message to keep the alternation. A previous
    summary, if there is one, is absorbed into the new one. Modifies the list
    in place and returns the summary (or None if it wasn't needed)."""

    if _token_count(chat_history) > CHAT_TOKEN_BUDGET:
        chat = ""
        old_texts, last_texts = chat_history[:-2], chat_history[-2:]

        for text in old_texts:
            chat += f"{text["role"]}: {text["content"]}\n"

    prompt = f"""Summarise the conversation below in a few lines, in Italian,
        keeping every concrete fact that came up (products, codes, figures, times,
        conditions) and the sources cited in square brackets. No preamble:
        just the summary.

        <conversation>
            {chat}
        </conversation>
    """

    response = client.responses.create(
        model=RESPONSE_MODEL,
        input=[{"role": "user", "content": prompt}]
    )

    summary = response.output_text.strip()

    chat_history[:] = [
        {"role": "user", "content": f"Summary of the conversation so far: {summary}"},
        {"role": "assistant", "content": "Got it, I'll keep that in mind"}
    ] + last_texts

    return summary

def _answer_in_chat(chat_history, user_question):
    if chat_history:
        question_for_llm = _compact(chat_history, user_question)
    else:
        question_for_llm = user_question

    chunks = retrieve(question_for_llm)
    response = generate_response(chunks, question_for_llm, chat_history)

    if MEMORY == "summary":
        _summarise(chat_history)
    if MEMORY == "buffer":
        _truncate(chat_history)

    return response, chunks, question_for_llm

if __name__ == "__main__":
    # TEST_QUERIES = [
    #     "Who's running the FCAS program?",
    #     "What is the internet?"
    # ]

    # print("ASKING")
    # # print(f"{"ID":<10}{}")
    # for q in TEST_QUERIES:
    #     print(f"\tAsking \"{q}\"")
    #     chunks = _find_chunks(q)
    #     print(f"Chunks: {[c for c in chunks]}")


    if open_collection(COLLECTION_NAME).count() == 0:
        index()

    session_name = input("Session name ('Enter' to continue without saving): ").strip()
    chat_history = []

    if session_name:
        chat_history = _load_session(session_name)
        print(f"Session '{session_name}' loaded: {len(chat_history)} texts retrieved")

    while True:
        user_question = input("You> ").strip()
        
        if user_question.lower() in {"exit", "quit"}:
            break   # so that users can quit the chat

        if not user_question:
            continue

        try:
            llm_response, chunks, question_for_llm = _answer_in_chat(chat_history, user_question)
        except Exception as e:
            raise

        if question_for_llm != user_question:
            print(f"   ↳ asked to LLM: {question_for_llm}")

        print_chunks(chunks)
        print(f"\nBot> {llm_response}")


        if session_name:
            _save_session(session_name, chat_history)
        