import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_rejects_non_pdf():
    fake_txt = ("document.txt", b"some text content", "text/plain")
    response = client.post("/ingest", files={"file": fake_txt})
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_chat_requires_question():
    response = client.post("/chat", json={})
    assert response.status_code == 422
