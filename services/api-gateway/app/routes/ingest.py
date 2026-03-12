"""Proxy routes for the ingestion-worker trigger endpoint.

Clients POST a document reference; the gateway forwards it to the
chat-orchestrator's /ingest endpoint which enqueues a Celery task.
"""
import os
import logging

import requests
from flask import Blueprint, request, Response, g

from app.middleware.auth import require_auth

log = logging.getLogger(__name__)
CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://chat-orchestrator:5002")

ingest_proxy_bp = Blueprint("ingest_proxy", __name__, url_prefix="/v1/ingest")


def _downstream_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if hasattr(g, "user_id"):
        headers["X-User-Id"] = g.user_id
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


@ingest_proxy_bp.post("/document")
@require_auth
def ingest_document():
    url = f"{CHAT_SERVICE_URL}/v1/ingest/document"
    try:
        resp = requests.post(
            url,
            headers=_downstream_headers(),
            json=request.get_json(silent=True),
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        log.error("chat-orchestrator unreachable for ingest at %s", CHAT_SERVICE_URL)
        return Response('{"error":"chat-orchestrator unavailable"}', status=503, mimetype="application/json")
    return Response(
        resp.content,
        status=resp.status_code,
        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
    )
