"""API handlers called directly: hermetic, needs no HTTP test client or Northwind."""
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.main import Query, health, query
from src.config import settings


def test_health():
    assert health() == {"status": "ok"}


def test_query_returns_contract_keys(tiny_db, fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tiny_db)
    fake_llm(["SELECT COUNT(*) AS n FROM schools", "There are 3 schools."])
    out = query(Query(question="How many schools are there?"))
    assert {"sql", "rows", "answer", "valid", "retries"} <= set(out)
    assert out["rows"] == [{"n": 3}] and out["answer"] == "There are 3 schools."


def test_question_validation():
    with pytest.raises(ValidationError):
        Query(question="")
    with pytest.raises(ValidationError):
        Query(question="x" * 501)


def test_llm_misconfiguration_maps_to_503(tiny_db, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tiny_db)

    def boom(system, user):
        raise RuntimeError("GOOGLE_API_KEY is empty")
    monkeypatch.setattr("src.agents.sql_writer.complete", boom)
    with pytest.raises(HTTPException) as e:
        query(Query(question="How many schools are there?"))
    assert e.value.status_code == 503 and "GOOGLE_API_KEY" in e.value.detail


def test_unexpected_error_maps_to_500_without_leaking_details(tiny_db, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tiny_db)

    def boom(system, user):
        raise ValueError("secret internals")
    monkeypatch.setattr("src.agents.sql_writer.complete", boom)
    with pytest.raises(HTTPException) as e:
        query(Query(question="How many schools are there?"))
    assert e.value.status_code == 500 and "secret" not in e.value.detail
