"""Document loading and chunking — ported from populate_db.py.

Chunk config preserved exactly:
    chunk_size=800, chunk_overlap=80
"""
import logging
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


def split_docs(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using the same config as populate_db.py."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    log.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Assign deterministic IDs to chunks so duplicate ingestion is idempotent.

    ID format: <source>:<page>:<chunk_index>
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        last_page_id = current_page_id

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"

    return chunks


def load_pdf_bytes(content: bytes, source_name: str) -> list[Document]:
    """Load a PDF from raw bytes and return a list of Documents."""
    import tempfile, os
    from langchain_community.document_loaders import PyPDFLoader

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = source_name
        log.info("Loaded %d pages from %s", len(docs), source_name)
        return docs
    finally:
        os.unlink(tmp_path)
