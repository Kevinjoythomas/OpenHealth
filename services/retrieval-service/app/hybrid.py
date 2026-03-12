"""Hybrid search with Reciprocal Rank Fusion (RRF).

Runs vector_search() and bm25_search() concurrently via ThreadPoolExecutor,
then fuses results with RRF: score = Σ 1/(k + rank_i) for each retriever.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.vector_store import vector_search
from app.bm25 import bm25_search

log = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant
_RETRIEVAL_TIMEOUT = 10.0  # seconds per retriever


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """Fuse vector and BM25 results via RRF using concurrent retrieval."""
    fetch_n = top_k * 2

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_vector = pool.submit(vector_search, query, fetch_n)
        fut_bm25 = pool.submit(bm25_search, query, fetch_n)

        vector_results: list[dict] = []
        bm25_results: list[dict] = []

        for fut in as_completed(
            {fut_vector: "vector", fut_bm25: "bm25"},
            timeout=_RETRIEVAL_TIMEOUT,
        ):
            name = {fut_vector: "vector", fut_bm25: "bm25"}[fut]
            try:
                result = fut.result()
                if name == "vector":
                    vector_results = result
                else:
                    bm25_results = result
            except Exception as exc:
                log.warning("%s retriever failed: %s — continuing without it", name, exc)

    if not bm25_results:
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
