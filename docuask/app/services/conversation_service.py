"""
conversation_service.py — replaces the in-memory store from Week 2.

Now every message is saved to the database permanently.
Users can come back days later and their conversations are still there —
exactly like ChatGPT's conversation sidebar.
"""

from sqlalchemy.orm import Session
from sqlalchemy import desc
from groq import Groq

from app.core.config import get_settings
from app.db.models import Conversation, ConversationMessage, Document, User
from app.services.document_service import retrieve_similar_chunks

settings = get_settings()


def get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


# ── Conversation CRUD ─────────────────────────────────────────────────────────

def create_conversation(user_id: int, document_id: int, title: str, db: Session) -> Conversation:
    """Start a new conversation for a user + document."""
    conv = Conversation(
        user_id=user_id,
        document_id=document_id,
        title=title,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversations_for_document(
    user_id: int, document_id: int, db: Session
) -> list[Conversation]:
    """Get all conversations a user has had about a specific document, newest first."""
    return (
        db.query(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Conversation.document_id == document_id,
        )
        .order_by(desc(Conversation.updated_at))
        .all()
    )


def get_all_conversations(user_id: int, db: Session) -> list[Conversation]:
    """Get all conversations for a user across all documents, newest first."""
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .all()
    )


def get_conversation(conversation_id: int, user_id: int, db: Session) -> Conversation | None:
    """Get a single conversation — verifies it belongs to the user."""
    return (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        .first()
    )


def delete_conversation(conversation_id: int, user_id: int, db: Session) -> bool:
    conv = get_conversation(conversation_id, user_id, db)
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True


def save_message(conversation_id: int, role: str, content: str, db: Session) -> ConversationMessage:
    """Save one message to the database."""
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages(conversation_id: int, db: Session) -> list[ConversationMessage]:
    """Get all messages for a conversation in chronological order."""
    return (
        db.query(ConversationMessage)
        .filter(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.created_at)
        .all()
    )


def auto_title(question: str) -> str:
    """
    Generate a short conversation title from the first question.
    Truncates to 50 chars so it fits in the sidebar nicely.
    """
    title = question.strip().rstrip('?').strip()
    return title[:50] + '...' if len(title) > 50 else title


# ── Main chat function ────────────────────────────────────────────────────────

def chat_with_document(
    question: str,
    document_id: int,
    user_id: int,
    conversation_id: int | None,
    db: Session,
) -> dict:
    """
    Full RAG pipeline with persistent history:
    1. Create or resume a conversation
    2. Load message history from DB
    3. Retrieve relevant chunks
    4. Build prompt with context + history
    5. Call Groq
    6. Save both messages to DB
    7. Return answer + conversation info
    """

    # Step 1 — get or create conversation
    if conversation_id:
        conv = get_conversation(conversation_id, user_id, db)
        if not conv:
            raise ValueError("Conversation not found.")
    else:
        # New conversation — title comes from first question
        conv = create_conversation(
            user_id=user_id,
            document_id=document_id,
            title=auto_title(question),
            db=db,
        )

    # Step 2 — load history from DB (last 10 messages to avoid huge prompts)
    history_msgs = get_messages(conv.id, db)
    history = [
        {"role": m.role, "content": m.content}
        for m in history_msgs[-10:]
    ]

    # Step 3 — retrieve relevant chunks from pgvector
    chunks = retrieve_similar_chunks(question, document_id, db)
    context = "\n\n---\n\n".join([c.content for c in chunks]) if chunks else ""

    # Step 4 — build messages for Groq
    system = {
        "role": "system",
        "content": (
            "You are a helpful document assistant. "
            "Answer questions based on the provided document context. "
            "If the answer is not in the context say: "
            "'I don't have enough information in this document to answer that.' "
            "Be concise, clear, and accurate."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}"
    }
    messages = [system] + history + [user_msg]

    # Step 5 — call Groq
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    # Step 6 — save both messages to DB permanently
    save_message(conv.id, "user", question, db)
    save_message(conv.id, "assistant", answer, db)

    # Update conversation timestamp
    from datetime import datetime
    conv.updated_at = datetime.utcnow()
    db.commit()

    return {
        "answer": answer,
        "conversation_id": conv.id,
        "conversation_title": conv.title,
        "sources": [
            {"chunk_index": c.chunk_index, "preview": c.content[:200] + "..."}
            for c in chunks
        ],
    }


def stream_chat_with_document(
    question: str,
    document_id: int,
    user_id: int,
    conversation_id: int | None,
    db: Session,
):
    """Streaming version — yields tokens then saves to DB when done."""

    # Create or resume conversation
    if conversation_id:
        conv = get_conversation(conversation_id, user_id, db)
        if not conv:
            yield "error: Conversation not found"
            return
    else:
        conv = create_conversation(
            user_id=user_id,
            document_id=document_id,
            title=auto_title(question),
            db=db,
        )

    history_msgs = get_messages(conv.id, db)
    history = [{"role": m.role, "content": m.content} for m in history_msgs[-10:]]

    chunks = retrieve_similar_chunks(question, document_id, db)
    context = "\n\n---\n\n".join([c.content for c in chunks]) if chunks else ""

    system = {
        "role": "system",
        "content": (
            "You are a helpful document assistant. "
            "Answer based strictly on the document context provided. "
            "If the answer is not in the context say you don't know."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}"
    }

    client = get_groq_client()
    stream = client.chat.completions.create(
        model=settings.chat_model,
        messages=[system] + history + [user_msg],
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    full_answer = ""
    # Yield conversation_id first so frontend knows which conversation this is
    import json
    yield f"meta:{json.dumps({'conversation_id': conv.id, 'title': conv.title})}\n"

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_answer += token
            yield token

    # Save both messages permanently after streaming completes
    save_message(conv.id, "user", question, db)
    save_message(conv.id, "assistant", full_answer, db)

    from datetime import datetime
    conv.updated_at = datetime.utcnow()
    db.commit()