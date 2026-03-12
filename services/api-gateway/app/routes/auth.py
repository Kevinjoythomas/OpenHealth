"""Proxy routes for the auth-service.

Signup and login are public. /refresh and /me require a valid JWT so
the gateway can validate before forwarding (saves a round-trip to auth-service
for obviously bad tokens).
"""
import os
import logging

import requests
from flask import Blueprint, request, Response

from app.middleware.auth import require_auth

log = logging.getLogger(__name__)
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:5001")

auth_proxy_bp = Blueprint("auth_proxy", __name__, url_prefix="/v1/auth")


def _forward(method: str, path: str, **kwargs) -> Response:
    """Forward a request to the auth-service and return its response verbatim."""
    url = f"{AUTH_SERVICE_URL}/v1/auth{path}"
    try:
        resp = requests.request(
            method,
            url,
            headers=_downstream_headers(),
            timeout=10,
            **kwargs,
        )
    except requests.exceptions.ConnectionError:
        log.error("auth-service unreachable at %s", AUTH_SERVICE_URL)
        return Response('{"error":"auth-service unavailable"}', status=503, mimetype="application/json")
    return Response(
        resp.content,
        status=resp.status_code,
        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
    )


def _downstream_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


@auth_proxy_bp.post("/signup")
def signup():
    return _forward("POST", "/signup", json=request.get_json(silent=True))


@auth_proxy_bp.post("/login")
def login():
    return _forward("POST", "/login", json=request.get_json(silent=True))


@auth_proxy_bp.post("/refresh")
def refresh():
    return _forward("POST", "/refresh", json=request.get_json(silent=True))


@auth_proxy_bp.get("/me")
@require_auth
def me():
    return _forward("GET", "/me")
