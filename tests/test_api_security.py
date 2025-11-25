# tests/test_security.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from importlib import reload
from fastapi.testclient import TestClient
import app.main as main_module


def _get_client_with_token(token: str) -> TestClient:
    # نثبّت الـ API_TOKEN في البيئة
    os.environ["API_TOKEN"] = token
    reload(main_module)  # نعيد تحميل app.main بحيث يقرأ الـ token الجديد
    return TestClient(main_module.app)


def test_ingest_requires_api_key():
    client = _get_client_with_token("test-token-123")

    # بدون هيدر X-API-KEY
    resp = client.post(
        "/ingest",
        files={"file": ("doc.txt", b"Some test contract text", "text/plain")},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing API key."


def test_ingest_and_ask_with_valid_api_key():
    token = "test-token-123"
    client = _get_client_with_token(token)

    # 1) ingest
    resp_ingest = client.post(
        "/ingest",
        files={
            "file": (
                "contract.txt",
                (
                    b"SERVICE AGREEMENT\n"
                    b"3. Term and Termination\n"
                    b"This Agreement lasts 6 months. Either party may terminate "
                    b"with 30 days written notice.\n"
                ),
                "text/plain",
            )
        },
        headers={"X-API-KEY": token},
    )
    assert resp_ingest.status_code == 200
    data = resp_ingest.json()
    assert "doc_id" in data
    doc_id = data["doc_id"]

    # 2) ask
    resp_ask = client.post(
        "/ask",
        json={
            "question": "What are the termination conditions?",
            "doc_id": doc_id,
        },
        headers={"X-API-KEY": token},
    )
    assert resp_ask.status_code == 200

    body = resp_ask.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)
    assert "termination" in body["answer"].lower() or "notice" in body["answer"].lower()
