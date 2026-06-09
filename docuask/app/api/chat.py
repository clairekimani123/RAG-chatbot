"""
Updated chat.py — Week 4 changes:
  - All endpoints require authentication (get_current_user)
  - Uses conversation_service instead of chat_service
  - Conversations are saved to DB permanently
  - New endpoints for listing/deleting conversations
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import json

from app.db.database import get_db
from app.db.models import User
from app.services.auth_service import get_current_user
from app.services.document_service import get_document
from app.services.conversation_service import (
    chat_with_document,
    stream_chat_with_document,
    get_conversations_for_document,
    get_all_conversations,
    get_messages,
    delete_conversation,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    document_id: int
    question: str
    conversation_id: Optional[int] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: int
    title: str
    document_id: int
    updated_at: str

    class Config:
        from_attributes = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask")
def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ask a question — saves conversation to DB permanently."""
    doc = get_document(request.document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    return chat_with_document(
        question=request.question,
        document_id=request.document_id,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        db=db,
    )


@router.post("/stream")
def ask_stream(
    request: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Streaming version — saves conversation to DB when done."""
    doc = get_document(request.document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    def generate():
        for chunk in stream_chat_with_document(
            question=request.question,
            document_id=request.document_id,
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            db=db,
        ):
            if chunk.startswith("meta:"):
                yield f"data: {chunk}\n\n"
            else:
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversations")
def list_all_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all conversations for the logged-in user across all documents."""
    convs = get_all_conversations(current_user.id, db)
    return [
        {
            "id": c.id,
            "title": c.title,
            "document_id": c.document_id,
            "document_name": c.document.original_name if c.document else "",
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]


@router.get("/conversations/document/{document_id}")
def list_document_conversations(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all conversations for a specific document."""
    convs = get_conversations_for_document(current_user.id, document_id, db)
    return [
        {
            "id": c.id,
            "title": c.title,
            "updated_at": c.updated_at.isoformat(),
            "message_count": len(c.messages),
        }
        for c in convs
    ]


@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Load full message history for a conversation."""
    messages = get_messages(conversation_id, db)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]


@router.delete("/conversations/{conversation_id}", status_code=204)
def remove_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a conversation and all its messages."""
    success = delete_conversation(conversation_id, current_user.id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")