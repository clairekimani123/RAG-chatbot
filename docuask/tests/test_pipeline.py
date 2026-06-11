"""

Tests chunking and validates the project structure is correct.

Usage:
  cd docuask
  python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_chunking():
    """Test that text chunking works correctly — no API key needed."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to the natural intelligence displayed by animals including humans.

    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.

    The term "artificial intelligence" had previously been used to describe 
    machines that mimic and display human cognitive skills associated with the 
    human mind, such as learning and problem-solving.
    """ * 10  # repeat to get enough content for multiple chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(sample_text)

    print(f"✅ Chunking works: {len(chunks)} chunks from sample text")
    print(f"   First chunk preview: {chunks[0][:80]}...")
    print(f"   Average chunk size: {sum(len(c) for c in chunks) // len(chunks)} chars")
    assert len(chunks) > 1, "Should produce multiple chunks"
    return True


def test_pdf_extraction():
    """Test PyMuPDF import works."""
    import fitz
    print(f"✅ PyMuPDF (fitz) loaded — version {fitz.version[0]}")
    return True




def test_pgvector_import():
    """Test pgvector SQLAlchemy extension imports."""
    from pgvector.sqlalchemy import Vector
    print("✅ pgvector importable — Vector column type ready")
    return True


def test_project_structure():
    """Verify all files exist."""
    required = [
        "app/main.py",
        "app/core/config.py",
        "app/db/database.py",
        "app/db/models.py",
        "app/services/document_service.py",
        "app/api/documents.py",
        "requirements.txt",
        ".env.example",
    ]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing = []
    for path in required:
        full = os.path.join(base, path)
        if not os.path.exists(full):
            missing.append(path)

    if missing:
        print(f"❌ Missing files: {missing}")
        return False

    print(f"✅ Project structure valid — all {len(required)} files present")
    return True


if __name__ == "__main__":
    print("\n=== DocuAsk Pipeline Tests ===\n")
    tests = [
        test_project_structure,
        test_pdf_extraction,
        test_pgvector_import,
        test_chunking,
    ]
    passed = sum(1 for t in tests if t())
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)