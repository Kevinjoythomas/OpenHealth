"""Ingestion-worker unit tests. Mocks S3, ChromaDB, and Ollama."""
import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document


def test_split_docs_chunk_size():
    """Chunks should be no larger than 800 chars."""
    from app.chunker import split_docs
    long_text = "word " * 400  # 2000 chars
    doc = Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})
    chunks = split_docs([doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 800


def test_calculate_chunk_ids_unique():
    from app.chunker import calculate_chunk_ids
    chunks = [
        Document(page_content=f"Chunk {i}", metadata={"source": "test.pdf", "page": 0})
        for i in range(5)
    ]
    result = calculate_chunk_ids(chunks)
    ids = [c.metadata["id"] for c in result]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_calculate_chunk_ids_idempotent():
    from app.chunker import calculate_chunk_ids
    chunks = [
        Document(page_content="Text", metadata={"source": "doc.pdf", "page": 1})
    ]
    first = calculate_chunk_ids(chunks)[0].metadata["id"]
    chunks2 = [
        Document(page_content="Text", metadata={"source": "doc.pdf", "page": 1})
    ]
    second = calculate_chunk_ids(chunks2)[0].metadata["id"]
    assert first == second


def test_ingest_document_task():
    """Full task flow with all external dependencies mocked."""
    from app.tasks import ingest_document

    fake_pdf = b"%PDF-1.4 fake content"
    fake_doc = Document(page_content="Patient has chest pain.", metadata={"source": "test.pdf", "page": 0})
    fake_chunk = Document(page_content="Patient has chest pain.", metadata={"source": "test.pdf", "page": 0, "id": "test.pdf:0:0"})

    with patch("app.tasks.download_bytes", return_value=fake_pdf), \
         patch("app.tasks.load_pdf_bytes", return_value=[fake_doc]), \
         patch("app.tasks.split_docs", return_value=[fake_chunk]), \
         patch("app.tasks.calculate_chunk_ids", return_value=[fake_chunk]), \
         patch("app.tasks._upsert_to_chroma", return_value=(1, 0)):
        result = ingest_document.run(s3_key="uploads/test.pdf", filename="test.pdf")

    assert result["chunks_added"] == 1
    assert result["chunks_skipped"] == 0
