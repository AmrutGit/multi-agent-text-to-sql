import pytest

# Starlette's TestClient needs an extra HTTP-client package; skip rather than fail
# collection for everyone who hasn't installed it.
try:
    from fastapi.testclient import TestClient
except RuntimeError:
    pytest.skip("starlette TestClient dependency not installed", allow_module_level=True)

from src.api.main import app

pytestmark = pytest.mark.northwind


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_contract(fake_llm):
    fake_llm(["SELECT COUNT(*) AS n FROM Customers", "93."])
    r = client.post("/query", json={"question": "How many customers are there?"})
    assert r.status_code == 200
    assert set(r.json()) == {"sql", "rows", "answer", "valid", "retries"}


def test_rejects_empty_question():
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_llm_misconfiguration_is_503(monkeypatch):
    from src import llm
    from src.config import settings
    monkeypatch.setattr(settings, "google_api_key", "")
    monkeypatch.setattr(llm, "_env_file_values", dict)  # ignore a developer's local .env
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    r = client.post("/query", json={"question": "How many customers are there?"})
    assert r.status_code == 503
