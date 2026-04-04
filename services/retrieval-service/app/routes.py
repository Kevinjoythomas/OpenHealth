import logging
import time

from flask import Blueprint, request, jsonify

from app.hybrid import hybrid_search
from app.vector_store import vector_search
from app.bm25 import bm25_search

log = logging.getLogger(__name__)

retrieve_bp = Blueprint("retrieve", __name__, url_prefix="/v1")

_LATENCY_WARN_MS = 1000  # warn if retrieval exceeds sub-second SLO


@retrieve_bp.post("/retrieve")
def retrieve():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400
    if top_k < 1 or top_k > 20:
        return jsonify({"error": "top_k must be between 1 and 20"}), 400

    t0 = time.perf_counter()
    try:
        results = hybrid_search(query, top_k=top_k)
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        return jsonify({"error": "Retrieval failed"}), 500
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if elapsed_ms > _LATENCY_WARN_MS:
        log.warning("retrieve latency=%.1fms exceeded %dms SLO", elapsed_ms, _LATENCY_WARN_MS)
    else:
        log.info("retrieve latency=%.1fms count=%d", elapsed_ms, len(results))

    return jsonify({"results": results, "count": len(results), "latency_ms": round(elapsed_ms, 1)}), 200


@retrieve_bp.post("/retrieve/vector")
def retrieve_vector():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        results = vector_search(query, top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"results": results, "count": len(results)}), 200


@retrieve_bp.post("/retrieve/bm25")
def retrieve_bm25():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        results = bm25_search(query, top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"results": results, "count": len(results)}), 200


@retrieve_bp.post("/retrieve/hybrid")
def retrieve_hybrid():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        results = hybrid_search(query, top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"results": results, "count": len(results)}), 200


@retrieve_bp.post("/retrieve/compare")
def compare():
    """Debug endpoint: returns vector, BM25, and hybrid results side-by-side.
    Used to demonstrate search strategy differences."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "query is required"}), 400

    def _run(fn, *args):
        t0 = time.perf_counter()
        try:
            results = fn(*args)
        except Exception as exc:
            log.warning("%s failed: %s", fn.__name__, exc)
            results = []
        return results, round((time.perf_counter() - t0) * 1000, 1)

    vec_results, vec_ms = _run(vector_search, query, top_k)
    bm25_results, bm25_ms = _run(bm25_search, query, top_k)
    hybrid_results, hybrid_ms = _run(hybrid_search, query, top_k)

    def _fmt(results):
        return [
            {
                "snippet": r["content"][:300],
                "source": r.get("metadata", {}).get("source", ""),
                "page": r.get("metadata", {}).get("page"),
                "score": r.get("score"),
            }
            for r in results
        ]

    return jsonify({
        "query": query,
        "vector": {"results": _fmt(vec_results), "latency_ms": vec_ms, "count": len(vec_results)},
        "bm25":   {"results": _fmt(bm25_results),  "latency_ms": bm25_ms,  "count": len(bm25_results)},
        "hybrid": {"results": _fmt(hybrid_results), "latency_ms": hybrid_ms, "count": len(hybrid_results)},
    }), 200
