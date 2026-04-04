"""Proxy routes for the chat-orchestrator.

All chat routes require a valid JWT — the gateway validates it before forwarding,
and injects X-User-Id so the orchestrator doesn't need to re-decode the token.
"""
import os
import logging

import requests
from flask import Blueprint, request, Response, g

from app.middleware.auth import require_auth

log = logging.getLogger(__name__)
CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://chat-orchestrator:5002")

chat_proxy_bp = Blueprint("chat_proxy", __name__, url_prefix="/v1/chat")


def _forward(method: str, path: str, **kwargs) -> Response:
    url = f"{CHAT_SERVICE_URL}/v1/chat{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=_downstream_headers(),
            timeout=600,  # LLM calls can be slow — local GGUF can take several minutes
            **kwargs,
        )
    except requests.exceptions.ConnectionError:
        log.error("chat-orchestrator unreachable at %s", CHAT_SERVICE_URL)
        return Response('{"error":"chat-orchestrator unavailable"}', status=503, mimetype="application/json")
    return Response(
        resp.content,
        status=resp.status_code,
        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
    )


def _downstream_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if hasattr(g, "user_id"):
        headers["X-User-Id"] = g.user_id
    if hasattr(g, "claims"):
        headers["X-User-Role"] = g.claims.get("role", "")
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


@chat_proxy_bp.post("/sessions")
@require_auth
def create_session():
    return _forward("POST", "/sessions", json=request.get_json(silent=True))


@chat_proxy_bp.get("/sessions")
@require_auth
def list_sessions():
    return _forward("GET", "/sessions", params=request.args)


@chat_proxy_bp.post("/sessions/<session_id>/messages")
@require_auth
def send_message(session_id: str):
    return _forward("POST", f"/sessions/{session_id}/messages", json=request.get_json(silent=True))


@chat_proxy_bp.get("/sessions/<session_id>/messages")
@require_auth
def get_messages(session_id: str):
    return _forward("GET", f"/sessions/{session_id}/messages", params=request.args)


@chat_proxy_bp.delete("/sessions/<session_id>")
@require_auth
def delete_session(session_id: str):
    return _forward("DELETE", f"/sessions/{session_id}")
