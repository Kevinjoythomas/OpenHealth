"""Retrieval-service unit tests. Mocks ChromaDB so no Ollama/Chroma is needed."""
import pytest
from unittest.mock import patch, MagicMock

from app.main import create_app


@pytest.fixture()
def app():
    application = create_app()
    application.config["TESTING"] = True
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_retrieve_missing_query(client):
    resp = client.post("/v1/retrieve", json={})
    assert resp.status_code == 400


def test_retrieve_top_k_bounds(client):
    resp = client.post("/v1/retrieve", json={"query": "headache", "top_k": 0})
    assert resp.status_code == 400
    resp = client.post("/v1/retrieve", json={"query": "headache", "top_k": 99})
    assert resp.status_code == 400


def test_retrieve_success(client):
    mock_results = [
        {"content": "Headache is common.", "metadata": {"source": "pain.pdf"}, "score": None}
    ]
    with patch("app.routes.hybrid_search", return_value=mock_results):
        resp = client.post("/v1/retrieve", json={"query": "I have a headache", "top_k": 3})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["results"][0]["content"] == "Headache is common."
