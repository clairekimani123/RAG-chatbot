import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, Chunk

settings = get_settings()

# Load the embedding model once — runs locally, no API needed
# all-MiniLM-L6-v2 is small, fast, and produces 384-dimensional vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_groq_client() -> Groq:
    """Create Groq client — used for chat completions only."""
    return Groq(api_key=settings.groq_api_key)


# ── Step 1: Extract text from PDF ─────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    if not full_text.strip():
        raise ValueError("No text found in PDF. The file may be a scanned image.")

    return full_text


# ── Step 2: Split text into chunks ────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


# ── Step 3: Generate embeddings locally ───────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Use sentence-transformers to embed text locally — completely free.
    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    Runs on your CPU, no internet required.
    """
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Embed one string — used when embedding a user question."""
    return embed_texts([text])[0]


# ── Step 4: Save to database ──────────────────────────────────────────────────

def ingest_document(file_bytes: bytes, filename: str, db: Session) -> Document:
    """Full pipeline: PDF bytes → extract → chunk → embed → store."""
    raw_text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(raw_text)

    if not chunks:
        raise ValueError("Document produced no chunks after processing.")

    embeddings = embed_texts(chunks)

    document = Document(
        filename=filename,
        original_name=filename,
        total_chunks=len(chunks),
    )
    db.add(document)
    db.flush()

    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk = Chunk(
            document_id=document.id,
            content=text,
            chunk_index=i,
            embedding=embedding,
        )
        db.add(chunk)

    db.commit()
    db.refresh(document)
    return document


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_similar_chunks(question: str, document_id: int, db: Session) -> list[Chunk]:
    """Find top K chunks most similar to the question using cosine distance."""
    question_embedding = embed_single(question)

    results = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.embedding.cosine_distance(question_embedding))
        .limit(settings.top_k_chunks)
        .all()
    )
    return results


# ── Chat — used in Week 2 /ask endpoint ───────────────────────────────────────

def ask_question(question: str, document_id: int, db: Session) -> dict:
    """
    Full RAG query pipeline:
    1. Find relevant chunks
    2. Build a prompt with those chunks as context
    3. Send to Groq for the answer
    4. Return answer + source chunks
    """
    # Step 1 — retrieve relevant chunks
    chunks = retrieve_similar_chunks(question, document_id, db)

    if not chunks:
        return {"answer": "I could not find relevant information in this document.", "sources": []}

    # Step 2 — build context from chunks
    context = "\n\n---\n\n".join([chunk.content for chunk in chunks])

    # Step 3 — build the prompt
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.
If the answer is not in the context, say "I don't have enough information in this document to answer that."
Do not make up information.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Step 4 — call Groq
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.content[:200] + "..."  # preview only
            }
            for chunk in chunks
        ]
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def list_documents(db: Session) -> list[Document]:
    return db.query(Document).order_by(Document.created_at.desc()).all()


def get_document(document_id: int, db: Session) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()


def delete_document(document_id: int, db: Session) -> bool:
    doc = get_document(document_id, db)
    if not doc:
        return False
    db.delete(doc)
    db.commit()
    return True