import pytest

from src.agents.sql_writer import _clean
from src.orchestrator import run

pytestmark = pytest.mark.northwind


def test_clean_strips_markdown_fences_and_prose():
    assert _clean("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert _clean("Here you go:\nSELECT 1") == "SELECT 1"


def test_happy_path_no_retry(fake_llm):
    fake_llm(["SELECT COUNT(*) AS n FROM Customers", "There are 93 customers."])
    out = run("How many customers are there?")
    assert out["valid"] is True and out["retries"] == 0
    assert out["rows"] == [{"n": 93}]
    assert out["answer"] == "There are 93 customers."


def test_self_correction_loop_feeds_error_back(fake_llm):
    calls = fake_llm([
        "SELECT COUNT(*) FROM Customer",            # wrong table name
        "SELECT COUNT(*) AS n FROM Customers",      # fixed after seeing the error
        "93 customers.",
    ])
    out = run("How many customers are there?")
    assert out["valid"] is True and out["retries"] == 1
    # second writer call must contain the failed SQL and the SQLite error
    assert "no such table" in calls[1]["user"] and "FROM Customer" in calls[1]["user"]


def test_gives_up_after_max_retries(fake_llm):
    fake_llm(["SELECT x FROM Customers"] * 3)  # 1 try + 2 retries, all bad; no explain call
    out = run("anything")
    assert out["valid"] is False and out["retries"] == 2 and out["rows"] == []


def test_write_statement_never_executes(fake_llm):
    fake_llm(["DROP TABLE Customers"] * 3)
    out = run("drop the customers table")
    assert out["valid"] is False
    from src.db.schema import get_table_schemas
    assert "Customers" in get_table_schemas()


def test_empty_result_skips_llm_explain(fake_llm):
    calls = fake_llm(["SELECT * FROM Customers WHERE Country = 'Narnia'"])
    out = run("customers in Narnia")
    assert out["valid"] is True and out["rows"] == []
    assert "no matching rows" in out["answer"] and len(calls) == 1


def test_rows_are_capped(fake_llm, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "max_rows", 5)
    fake_llm(["SELECT * FROM Orders", "ok"])
    assert len(run("all orders")["rows"]) == 5


def test_runaway_query_is_aborted(fake_llm, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "query_timeout_s", 0.2)
    # accidental cross join of the 600k-row table with itself
    fake_llm(["SELECT COUNT(*) FROM \"Order Details\" a, \"Order Details\" b"] * 3)
    out = run("cross join")
    assert out["valid"] is False and "interrupted" in out["answer"].lower()
