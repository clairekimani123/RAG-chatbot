"""
Updated documents.py — Week 4 changes:
  - Upload now accepts PDF, PNG, JPEG, TXT, DOCX
  - All endpoints require authentication
  - Users only see their own documents
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db
from app.db.models import User
from app.services.auth_service import get_current_user
from app.services.document_service import (
    ingest_document,
    list_documents,
    get_document,
    delete_document,
    SUPPORTED_TYPES,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# All accepted extensions flattened into one list
ALL_EXTENSIONS = [ext for exts in SUPPORTED_TYPES.values() for ext in exts]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    file_type: str
    total_chunks: int
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/upload", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload any supported file type — PDF, image, TXT, or DOCX.
    Automatically detects type and runs the right extraction pipeline.
    """
    # Check extension
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALL_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(ALL_EXTENSIONS)}"
        )

    file_bytes = await file.read()

    # 20MB limit
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")

    try:
        document = ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
            db=db,
            user_id=current_user.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {
        "message": f"File processed into {document.total_chunks} chunks.",
        "document": DocumentResponse.model_validate(document),
    }


@router.get("", response_model=list[DocumentResponse])
def get_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List only the logged-in user's documents."""
    return list_documents(db, user_id=current_user.id)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_one_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = get_document(document_id, db)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = get_document(document_id, db)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_document(document_id, db)