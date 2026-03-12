"""BM25 lexical search backed by ChromaDB document corpus.

Index is built lazily on first query and cached in memory.
Rebuilds on service restart — corpus is loaded from ChromaDB directly.
No additional embedding model calls; tokenisation is simple whitespace split.
"""
import logging

log = logging.getLogger(__name__)

_bm25_index = None
_corpus_docs: list[dict] = []


def _build_index() -> None:
    """Load all documents from ChromaDB and build a BM25Okapi index."""
    global _bm25_index, _corpus_docs

    try:
        from rank_bm25 import BM25Okapi
        from app.vector_store import get_chroma

        db = get_chroma()
        result = db._collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not documents:
            log.warning("BM25: ChromaDB corpus is empty — skipping index build")
            return

        _corpus_docs = [
            {"content": doc, "metadata": meta or {}}
            for doc, meta in zip(documents, metadatas)
        ]
        tokenized = [doc.lower().split() for doc in documents]
        _bm25_index = BM25Okapi(tokenized)
        log.info("BM25: built index with %d documents", len(documents))

    except Exception as exc:
        log.warning("BM25 index build failed: %s", exc)


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """BM25 keyword search over the ChromaDB corpus.

    Returns list of {content, metadata, score}.
    Only documents with score > 0 are returned (no spurious matches).
    """
    global _bm25_index

    if _bm25_index is None:
        _build_index()

    if _bm25_index is None or not _corpus_docs:
        log.debug("BM25: no index available — returning empty results")
        return []

    scores = _bm25_index.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "content": _corpus_docs[i]["content"],
            "metadata": _corpus_docs[i]["metadata"],
            "score": float(scores[i]),
        }
        for i in top_indices
        if scores[i] > 0
    ]
