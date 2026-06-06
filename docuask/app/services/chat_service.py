"""
chat_service.py — handles:
  - building prompts with context + conversation history
  - calling Groq for answers
  - streaming responses
  - storing conversation history in memory
"""

import uuid
from groq import Groq
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.document_service import retrieve_similar_chunks

settings = get_settings()

# In-memory conversation store
# Key = conversation_id, Value = list of message dicts
# In Week 4 we'll move this to the database
conversation_store: dict[str, list[dict]] = {}


def get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# ── Conversation history helpers ──────────────────────────────────────────────

def get_history(conversation_id: str) -> list[dict]:
    """Return conversation history for this session."""
    return conversation_store.get(conversation_id, [])


def add_to_history(conversation_id: str, role: str, content: str):
    """Append a message to the conversation history."""
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []
    conversation_store[conversation_id].append({
        "role": role,       # "user" or "assistant"
        "content": content,
    })


def clear_history(conversation_id: str):
    """Delete conversation history."""
    conversation_store.pop(conversation_id, None)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    """
    Build the full messages list to send to Groq.

    Structure:
      1. System message — tells the AI its role and rules
      2. Conversation history — previous Q&A turns
      3. Current question with document context injected
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful document assistant. "
            "Answer questions based strictly on the document context provided. "
            "If the answer is not in the context, say: "
            "'I don't have enough information in this document to answer that.' "
            "Be concise, clear, and accurate. "
            "When relevant, mention which part of the document your answer comes from."
        )
    }

    # Current question with context injected
    user_message = {
        "role": "user",
        "content": (
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )
    }

    # history = previous turns, injected between system and current question
    # This gives the AI memory of the conversation
    return [system_message] + history + [user_message]


# ── Main answer function ──────────────────────────────────────────────────────

def get_answer(
    question: str,
    document_id: int,
    document_name: str,
    conversation_id: str | None,
    db: Session,
) -> dict:
    """
    Full RAG pipeline:
    1. Find relevant chunks using semantic search
    2. Build prompt with context + history
    3. Call Groq
    4. Store result in conversation history
    5. Return answer + sources
    """
    # Generate a conversation ID if this is a new session
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    # Step 1 — retrieve relevant chunks
    chunks = retrieve_similar_chunks(question, document_id, db)
    if not chunks:
        return {
            "answer": "I could not find relevant information in this document.",
            "conversation_id": conversation_id,
            "sources": [],
            "document_name": document_name,
        }

    # Step 2 — build context string from chunks
    context = "\n\n---\n\n".join([chunk.content for chunk in chunks])

    # Step 3 — get conversation history for this session
    history = get_history(conversation_id)

    # Step 4 — build messages
    messages = build_messages(question, context, history)

    # Step 5 — call Groq
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.1,    # low = factual and consistent
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    # Step 6 — save this turn to history
    # We save the plain question (not the context-injected version)
    # so history stays readable for follow-up turns
    add_to_history(conversation_id, "user", question)
    add_to_history(conversation_id, "assistant", answer)

    # Step 7 — return result
    return {
        "answer": answer,
        "conversation_id": conversation_id,
        "document_name": document_name,
        "sources": [
            {
                "chunk_index": chunk.chunk_index,
                "preview": chunk.content[:200] + "..."
            }
            for chunk in chunks
        ],
    }


# ── Streaming version ─────────────────────────────────────────────────────────

def stream_answer(
    question: str,
    document_id: int,
    conversation_id: str | None,
    db: Session,
):
    """
    Same as get_answer but yields tokens as they arrive from Groq.
    Uses Groq's streaming API — the model sends words as it generates them.
    """
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    chunks = retrieve_similar_chunks(question, document_id, db)
    if not chunks:
        yield "I could not find relevant information in this document."
        return

    context = "\n\n---\n\n".join([chunk.content for chunk in chunks])
    history = get_history(conversation_id)
    messages = build_messages(question, context, history)

    client = get_groq_client()

    # stream=True tells Groq to send tokens as they are generated
    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_answer += token
            yield token   # send this token to the frontend immediately

    # Save the complete answer to history after streaming finishes
    add_to_history(conversation_id, "user", question)
    add_to_history(conversation_id, "assistant", full_answer)