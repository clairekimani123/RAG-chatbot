"""
Two tables:
  Document — one row per uploaded PDF (metadata only)
  Chunk    — many rows per document (text + embedding vector)

The vector column uses pgvector's VECTOR type.
1536 dimensions = size of OpenAI text-embedding-3-small output.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    original_name = Column(String, nullable=False)
    total_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # One document → many chunks
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document id={self.id} name={self.original_name}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)          # the actual text of this chunk
    chunk_index = Column(Integer, nullable=False)   # position in the document
    embedding = Column(Vector(384))               # pgvector column — 1536 dims

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk id={self.id} doc={self.document_id} index={self.chunk_index}>"