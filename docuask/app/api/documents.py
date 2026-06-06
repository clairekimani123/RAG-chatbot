"""
/api/documents endpoints:

  POST   /api/documents/upload   → upload + ingest a PDF
  GET    /api/documents           → list all documents
  GET    /api/documents/{id}      → get one document
  DELETE /api/documents/{id}      → delete document + its chunks
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db
from app.services.document_service import (
    ingest_document,
    list_documents,
    get_document,
    delete_document,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


# ── Response schemas (what the API sends back) ────────────────────────────────

class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    total_chunks: int
    created_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    message: str
    document: DocumentResponse


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF and run the full ingestion pipeline.
    Returns the saved document with chunk count.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file."
        )

    # Validate file size (max 20MB)
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 20MB."
        )

    try:
        document = ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return UploadResponse(
        message=f"Document ingested successfully into {document.total_chunks} chunks.",
        document=DocumentResponse.model_validate(document),
    )


@router.get("", response_model=list[DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    """List all uploaded documents, newest first."""
    return list_documents(db)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document_by_id(document_id: int, db: Session = Depends(get_db)):
    doc = get_document(document_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and all its chunks."""
    success = delete_document(document_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")