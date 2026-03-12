"""Celery tasks for document ingestion.

Task flow:
1. Download PDF bytes from S3 (s3_key).
2. Load and split into chunks (chunk_size=800, overlap=80).
3. Assign deterministic chunk IDs (idempotent upsert).
4. Embed with nomic-embed-text via Ollama.
5. Upsert to ChromaDB — skipping chunks that already exist.
"""
import logging
import os

from celery import shared_task
from celery.utils.log import get_task_logger

from app.worker import celery_app
from app.chunker import load_pdf_bytes, split_docs, calculate_chunk_ids
from app.embedder import get_embeddings
from app.storage import download_bytes

log = get_task_logger(__name__)


@celery_app.task(
    name="ingestion_worker.tasks.ingest_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def ingest_document(self, s3_key: str, filename: str) -> dict:
    """Download, chunk, embed, and upsert a document to ChromaDB.

    Args:
        s3_key: S3 object key (e.g. "uploads/backpain.pdf").
        filename: Human-readable name used as metadata source.

    Returns:
        dict with keys: s3_key, chunks_added, chunks_skipped.
    """
    log.info("Starting ingestion: s3_key=%s filename=%s", s3_key, filename)

    try:
        pdf_bytes = download_bytes(s3_key)
    except Exception as exc:
        log.error("S3 download failed: %s", exc)
        raise self.retry(exc=exc)

    try:
        docs = load_pdf_bytes(pdf_bytes, source_name=filename or s3_key)
        chunks = split_docs(docs)
        chunks = calculate_chunk_ids(chunks)
    except Exception as exc:
        log.error("Chunking failed: %s", exc)
        raise self.retry(exc=exc)

    try:
        added, skipped = _upsert_to_chroma(chunks)
    except Exception as exc:
        log.error("ChromaDB upsert failed: %s", exc)
        raise self.retry(exc=exc)

    log.info(
        "Ingestion complete: s3_key=%s added=%d skipped=%d",
        s3_key, added, skipped,
    )
    return {"s3_key": s3_key, "chunks_added": added, "chunks_skipped": skipped}


def _upsert_to_chroma(chunks) -> tuple[int, int]:
    """Upsert chunks to ChromaDB, skipping those with existing IDs."""
    from langchain_chroma import Chroma

    chroma_path = os.getenv("CHROMA_PATH", "./chroma")
    embeddings = get_embeddings()

    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    skipped = len(chunks) - len(new_chunks)

    if new_chunks:
        new_ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        log.info("Added %d new chunks to ChromaDB", len(new_chunks))
    else:
        log.info("No new chunks to add")

    return len(new_chunks), skipped
