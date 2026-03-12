"""API gateway unit tests.

Uses unittest.mock to avoid real Redis / downstream HTTP calls.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.main import create_app


@pytest.fixture()
def app():
    application = create_app()
    application.config["TESTING"] = True
    # Disable rate limiter Redis calls in tests
    with patch("app.middleware.rate_limit.RateLimiter._get_redis") as mock_redis:
        redis_mock = MagicMock()
        redis_mock.pipeline.return_value.__enter__ = lambda s: s
        redis_mock.pipeline.return_value.__exit__ = MagicMock(return_value=False)
        redis_mock.pipeline.return_value.execute.return_value = [None, None, 1, None]
        mock_redis.return_value = redis_mock
        yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json["status"] == "ok"


def test_health_ready_degraded(client):
    with patch("requests.get", side_effect=ConnectionError("down")):
        resp = client.get("/health/ready")
    assert resp.status_code == 503


def test_signup_proxied(client):
    mock_response = MagicMock()
    mock_response.content = b'{"access_token":"tok"}'
    mock_response.status_code = 201
    mock_response.headers = {"Content-Type": "application/json"}
    with patch("requests.request", return_value=mock_response):
        resp = client.post(
            "/v1/auth/signup",
            json={"email": "doc@example.com", "password": "secret123", "name": "Dr Test"},
        )
    assert resp.status_code == 201


def test_chat_requires_auth(client):
    resp = client.post("/v1/chat/sessions", json={})
    assert resp.status_code == 401


def test_chat_with_valid_token(client):
    import jwt, os, time
    token = jwt.encode(
        {"sub": "user-1", "role": "doctor", "type": "access", "exp": time.time() + 300},
        os.getenv("JWT_SECRET", "CHANGE-ME-in-production"),
        algorithm="HS256",
    )
    mock_response = MagicMock()
    mock_response.content = b'{"id":"sess-1"}'
    mock_response.status_code = 201
    mock_response.headers = {"Content-Type": "application/json"}
    with patch("requests.request", return_value=mock_response):
        resp = client.post(
            "/v1/chat/sessions",
            json={"title": "My session"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 201
