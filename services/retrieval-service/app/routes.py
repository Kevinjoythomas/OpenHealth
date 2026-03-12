import logging

from flask import Blueprint, request, jsonify

from app.hybrid import hybrid_search

log = logging.getLogger(__name__)

retrieve_bp = Blueprint("retrieve", __name__, url_prefix="/v1")


@retrieve_bp.post("/retrieve")
def retrieve():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400
    if top_k < 1 or top_k > 20:
        return jsonify({"error": "top_k must be between 1 and 20"}), 400

    try:
        results = hybrid_search(query, top_k=top_k)
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        return jsonify({"error": "Retrieval failed"}), 500

    return jsonify({"results": results, "count": len(results)}), 200
