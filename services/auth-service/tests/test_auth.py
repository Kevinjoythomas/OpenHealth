"""Auth-service unit tests using an in-memory SQLite database."""
import pytest
from unittest.mock import patch, MagicMock

from app.main import create_app


@pytest.fixture()
def app():
    application = create_app.__wrapped__() if hasattr(create_app, "__wrapped__") else None
    application = create_app()
    application.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    # Patch Redis so tests don't need a running Redis instance
    with patch("app.auth.get_redis") as mock_get_redis:
        redis_mock = MagicMock()
        redis_mock.exists.return_value = 1
        mock_get_redis.return_value = redis_mock
        with application.app_context():
            from app.db import db
            db.create_all()
        yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_signup_success(client):
    with patch("app.auth.get_redis") as mock_redis:
        redis_mock = MagicMock()
        mock_redis.return_value = redis_mock
        resp = client.post("/v1/auth/signup", json={
            "email": "doctor@example.com",
            "password": "securepass1",
            "name": "Dr Example",
        })
    assert resp.status_code == 201
    data = resp.get_json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_signup_duplicate_email(client):
    with patch("app.auth.get_redis", return_value=MagicMock()):
        client.post("/v1/auth/signup", json={
            "email": "dup@example.com",
            "password": "securepass1",
            "name": "Dr Dup",
        })
        resp = client.post("/v1/auth/signup", json={
            "email": "dup@example.com",
            "password": "securepass1",
            "name": "Dr Dup2",
        })
    assert resp.status_code == 409


def test_login_invalid_credentials(client):
    resp = client.post("/v1/auth/login", json={
        "email": "nobody@example.com",
        "password": "wrongpass",
    })
    assert resp.status_code == 401


def test_login_success(client):
    with patch("app.auth.get_redis", return_value=MagicMock()):
        client.post("/v1/auth/signup", json={
            "email": "login@example.com",
            "password": "securepass1",
            "name": "Dr Login",
        })
        resp = client.post("/v1/auth/login", json={
            "email": "login@example.com",
            "password": "securepass1",
        })
    assert resp.status_code == 200
    assert "access_token" in resp.get_json()


def test_me_requires_auth(client):
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401
