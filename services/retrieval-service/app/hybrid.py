"""Hybrid search with Reciprocal Rank Fusion (RRF) — Phase 4 placeholder.

Currently passes through vector-only results. When BM25 is implemented (Phase 4),
this module will:
1. Run vector_search() and bm25_search() concurrently.
2. Normalise scores with RRF: score = Σ 1/(k + rank_i) for each retriever i.
3. Return the top_k results sorted by fused score.
"""
import logging

from app.vector_store import vector_search
from app.bm25 import bm25_search

log = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Fuse vector and BM25 results via RRF.

    Phase 4 TODO: run both retrievers concurrently with ThreadPoolExecutor.
    """
    vector_results = vector_search(query, top_k=top_k * 2)
    bm25_results = bm25_search(query, top_k=top_k * 2)

    if not bm25_results:
        # BM25 not yet implemented — return vector results directly
        return vector_results[:top_k]

    return _rrf_fuse(vector_results, bm25_results, top_k=top_k)


def _rrf_fuse(
    list_a: list[dict],
    list_b: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Reciprocal Rank Fusion across two ranked lists."""
    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for rank, item in enumerate(list_a):
        key = item["content"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        content_map[key] = item

    for rank, item in enumerate(list_b):
        key = item["content"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        content_map[key] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**content_map[key], "score": score}
        for key, score in ranked[:top_k]
    ]
