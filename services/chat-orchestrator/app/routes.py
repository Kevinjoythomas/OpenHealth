"""Chat-orchestrator HTTP routes.

All routes are protected: X-User-Id header is injected by the api-gateway
after JWT validation, so this service trusts that header without re-verifying.

Route prefix: /v1/chat
"""
import logging

from flask import Blueprint, request, jsonify, current_app

from app import session as session_store
from app.orchestrator import run_chat
from app.schemas import CreateSessionRequest, SendMessageRequest, IngestDocumentRequest

log = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/v1/chat")
ingest_bp = Blueprint("ingest", __name__, url_prefix="/v1/ingest")


def _get_user_id() -> str | None:
    return request.headers.get("X-User-Id")


def _require_user_id():
    uid = _get_user_id()
    if not uid:
        return None, (jsonify({"error": "X-User-Id header required"}), 401)
    return uid, None


# ── Session endpoints ─────────────────────────────────────────────────────────

@chat_bp.post("/sessions")
def create_session():
    user_id, err = _require_user_id()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    req = CreateSessionRequest.from_dict(data)
    session = session_store.create_session(user_id, title=req.title)
    return jsonify(session.to_dict()), 201


@chat_bp.get("/sessions")
def list_sessions():
    user_id, err = _require_user_id()
    if err:
        return err

    sessions = session_store.list_sessions(user_id)
    return jsonify([s.to_dict() for s in sessions]), 200


@chat_bp.post("/sessions/<session_id>/messages")
def send_message(session_id: str):
    user_id, err = _require_user_id()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    req = SendMessageRequest.from_dict(data)
    if not req.message:
        return jsonify({"error": "message is required"}), 400

    # Verify session belongs to this user
    session = session_store.get_session(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    try:
        answer = run_chat(session_id, req.message)
    except Exception as exc:
        log.error("Chat pipeline error: %s", exc)
        return jsonify({"error": "Failed to process message"}), 502

    return jsonify({"session_id": session_id, "answer": answer}), 200


@chat_bp.get("/sessions/<session_id>/messages")
def get_messages(session_id: str):
    user_id, err = _require_user_id()
    if err:
        return err

    session = session_store.get_session(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    messages = session_store.get_messages(session_id)
    return jsonify([m.to_dict() for m in messages]), 200


@chat_bp.delete("/sessions/<session_id>")
def delete_session(session_id: str):
    user_id, err = _require_user_id()
    if err:
        return err

    deleted = session_store.delete_session(session_id, user_id)
    if not deleted:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"deleted": True}), 200


# ── Ingestion trigger ─────────────────────────────────────────────────────────

@ingest_bp.post("/document")
def ingest_document():
    """Enqueue a document ingestion task via Celery."""
    user_id, err = _require_user_id()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    req = IngestDocumentRequest.from_dict(data)

    if not req.s3_key:
        return jsonify({"error": "s3_key is required"}), 400

    try:
        # Import here to avoid Celery startup cost on every request
        from app.celery_client import enqueue_ingest
        task_id = enqueue_ingest(req.s3_key, req.filename)
        return jsonify({"task_id": task_id, "status": "queued"}), 202
    except Exception as exc:
        log.error("Failed to enqueue ingestion task: %s", exc)
        return jsonify({"error": "Failed to enqueue task"}), 503
