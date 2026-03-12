"""BM25 lexical search — Phase 4 placeholder.

In Phase 4 this will be backed by a pre-built BM25 index persisted to disk
(or loaded from ChromaDB document corpus on startup).

For now it returns an empty list so the hybrid ranker gracefully falls back
to pure vector search.
"""
import logging

log = logging.getLogger(__name__)


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Placeholder: returns empty results until BM25 index is built (Phase 4).

    Returns list of {content, metadata, score}.
    """
    log.debug("BM25 search not yet implemented — returning empty results")
    return []
