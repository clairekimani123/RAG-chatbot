"""
/api/chat endpoints:

  POST /api/chat/ask       → ask a question about a document
  POST /api/chat/stream    → same but streams the answer word by word
  GET  /api/chat/history/{document_id} → get conversation history
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import json

from app.db.database import get_db
from app.db.models import Document
from app.services.document_service import retrieve_similar_chunks, get_document
from app.services.chat_service import (
    get_answer,
    stream_answer,
    get_history,
    clear_history,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    document_id: int
    question: str
    conversation_id: Optional[str] = None   # optional — tracks multi-turn conversations


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list[dict]
    document_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)):
    """
    Ask a question about an uploaded document.
    Returns the AI answer plus the source chunks it used.
    """
    # Verify document exists
    doc = get_document(request.document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = get_answer(
        question=request.question,
        document_id=request.document_id,
        document_name=doc.original_name,
        conversation_id=request.conversation_id,
        db=db,
    )
    return result


@router.post("/stream")
def ask_stream(request: AskRequest, db: Session = Depends(get_db)):
    """
    Same as /ask but streams the answer token by token.
    The frontend receives chunks of text as they are generated.
    """
    doc = get_document(request.document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def generate():
        for chunk in stream_answer(
            question=request.question,
            document_id=request.document_id,
            conversation_id=request.conversation_id,
            db=db,
        ):
            # Send each token as a server-sent event
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/history/{document_id}")
def conversation_history(document_id: int, conversation_id: str, db: Session = Depends(get_db)):
    """Get the full conversation history for a document session."""
    doc = get_document(document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return get_history(conversation_id)


@router.delete("/history/{conversation_id}")
def delete_history(conversation_id: str):
    """Clear conversation history for a session."""
    clear_history(conversation_id)
    return {"message": "History cleared."}