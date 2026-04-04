"""Chat-orchestrator unit tests.

Mocks Postgres (SQLite in-memory), Redis, retrieval-service HTTP call, and Ollama.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.main import create_app


@pytest.fixture()
def app():
    application = create_app()
    application.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    with patch("app.session.get_redis") as mock_redis:
        mock_redis.return_value = MagicMock(get=lambda k: None, set=MagicMock(), delete=MagicMock())
        with application.app_context():
            from app.db import db
            db.create_all()
        yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def _auth_headers(user_id: str = "user-test-1") -> dict:
    return {"X-User-Id": user_id}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_create_session(client):
    with patch("app.session.get_redis", return_value=MagicMock(
        get=lambda k: None, set=MagicMock(), delete=MagicMock()
    )):
        resp = client.post("/v1/chat/sessions", json={"title": "Test session"}, headers=_auth_headers())
    assert resp.status_code == 201
    assert "id" in resp.get_json()


def test_create_session_no_user(client):
    resp = client.post("/v1/chat/sessions", json={})
    assert resp.status_code == 401


def test_list_sessions_empty(client):
    with patch("app.session.get_redis", return_value=MagicMock(
        get=lambda k: None, set=MagicMock(), delete=MagicMock()
    )):
        resp = client.get("/v1/chat/sessions", headers=_auth_headers())
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_send_message(client):
    redis_mock = MagicMock(get=lambda k: None, set=MagicMock(), delete=MagicMock())

    with patch("app.session.get_redis", return_value=redis_mock):
        session_resp = client.post(
            "/v1/chat/sessions",
            json={"title": "Medical chat"},
            headers=_auth_headers(),
        )
    session_id = session_resp.get_json()["id"]

    with patch("app.session.get_redis", return_value=redis_mock), \
         patch("app.orchestrator._call_retrieval_service", return_value=[]), \
         patch("app.orchestrator._get_llm") as mock_llm:

        mock_llm.return_value.invoke = MagicMock(return_value="Take ibuprofen.")
        resp = client.post(
            f"/v1/chat/sessions/{session_id}/messages",
            json={"message": "I have a headache"},
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    assert "answer" in resp.get_json()
    assert resp.get_json()["title"] == "Medical chat"


def test_send_message_generates_meaningful_title(client):
    redis_mock = MagicMock(get=lambda k: None, set=MagicMock(), delete=MagicMock())

    with patch("app.session.get_redis", return_value=redis_mock):
        session_resp = client.post(
            "/v1/chat/sessions",
            json={"title": "New consultation"},
            headers=_auth_headers(),
        )
    assert session_resp.status_code == 201
    assert session_resp.get_json()["title"] is None
    session_id = session_resp.get_json()["id"]

    with patch("app.session.get_redis", return_value=redis_mock), \
         patch("app.orchestrator._call_retrieval_service", return_value=[]), \
         patch("app.orchestrator._get_llm") as mock_llm:

        mock_llm.return_value.invoke = MagicMock(return_value="Rest and hydrate.")
        resp = client.post(
            f"/v1/chat/sessions/{session_id}/messages",
            json={"message": "I have fever and cough for three days"},
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    assert resp.get_json()["title"] == "Fever and cough for three days"

    with patch("app.session.get_redis", return_value=redis_mock):
        sessions_resp = client.get("/v1/chat/sessions", headers=_auth_headers())
    assert sessions_resp.status_code == 200
    assert sessions_resp.get_json()[0]["title"] == "Fever and cough for three days"


def test_list_sessions_backfills_placeholder_title(app, client):
    redis_mock = MagicMock(get=lambda k: None, set=MagicMock(), delete=MagicMock())

    with app.app_context():
        from app.db import db
        from app.models import ChatSession, ChatMessage, MessageRole

        sess = ChatSession(user_id="user-test-1", title="New consultation")
        db.session.add(sess)
        db.session.flush()
        db.session.add(ChatMessage(
            session_id=sess.id,
            role=MessageRole.USER,
            content="My chest pain when breathing deeply",
        ))
        db.session.commit()

    with patch("app.session.get_redis", return_value=redis_mock):
        resp = client.get("/v1/chat/sessions", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.get_json()[0]["title"] == "Chest pain when breathing deeply"


def test_delete_session(client):
    redis_mock = MagicMock(get=lambda k: None, set=MagicMock(), delete=MagicMock())

    with patch("app.session.get_redis", return_value=redis_mock):
        sess = client.post("/v1/chat/sessions", json={}, headers=_auth_headers())
    session_id = sess.get_json()["id"]

    with patch("app.session.get_redis", return_value=redis_mock):
        resp = client.delete(f"/v1/chat/sessions/{session_id}", headers=_auth_headers())
    assert resp.status_code == 200
