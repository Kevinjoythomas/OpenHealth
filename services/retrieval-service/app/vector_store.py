"""ChromaDB vector store wrapper.

Keeps the same configuration as the original app.py:
- persist_directory from CHROMA_PATH env var
- OllamaEmbeddings with "nomic-embed-text"
- MMR search type
"""
import logging
import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

log = logging.getLogger(__name__)

_chroma_db: Chroma | None = None


def get_chroma() -> Chroma:
    """Return a cached Chroma instance (created once per process)."""
    global _chroma_db
    if _chroma_db is None:
        chroma_path = os.getenv("CHROMA_PATH", "./chroma")
        embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        log.info("Initialising ChromaDB at %s with embed model %s", chroma_path, embed_model)
        embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_url)
        _chroma_db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    return _chroma_db


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Run MMR search against ChromaDB. Returns list of {content, metadata, score}."""
    db = get_chroma()
    results = db.max_marginal_relevance_search(query, k=top_k)
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": None,  # MMR doesn't return scores
        }
        for doc in results
    ]
