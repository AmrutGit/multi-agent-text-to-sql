"""The pipeline on an arbitrary DB (not Northwind): proves nothing is hard-wired."""
from src.agents.schema_retriever import retrieve_schema_context, select_tables
from src.agents.validator import validate
from src.orchestrator import run


def test_validator_uses_given_db(tiny_db):
    assert validate("SELECT name FROM schools", tiny_db) == (True, "")
    ok, err = validate("SELECT name FROM Customers", tiny_db)
    assert not ok and "no such table" in err


def test_retriever_selects_and_bridges_on_other_db(tiny_db):
    assert select_tables("What is the average math score per county?", tiny_db)[0] in {"scores", "schools"}
    assert {"schools", "scores"} <= set(select_tables("math score of each school", tiny_db))


def test_full_mode_returns_every_table_and_no_northwind_fewshot(tiny_db):
    ctx = retrieve_schema_context("anything", tiny_db, mode="full")
    assert "audit_log" in ctx and "scores" in ctx and "Examples:" not in ctx


def test_run_on_other_db_with_evidence_and_no_explainer(tiny_db, fake_llm):
    calls = fake_llm(["SELECT name FROM schools WHERE enrollment > 500 ORDER BY name"])
    out = run("Which schools are big?", db_path=tiny_db, evidence="big means enrollment > 500",
              explain_answer=False)
    assert out["valid"] and [r["name"] for r in out["rows"]] == ["Alpha High", "Gamma Academy"]
    assert out["answer"] == "" and len(calls) == 1  # explainer skipped: no 2nd LLM call
    assert "Hint: big means enrollment > 500" in calls[0]["user"]
    assert "Revenue" not in calls[0]["system"]  # Northwind hints must not leak
    assert out["attempts"] == [{"sql": "SELECT name FROM schools WHERE enrollment > 500 ORDER BY name", "error": None}]


def test_zero_retries_means_single_shot(tiny_db, fake_llm):
    fake_llm(["SELECT bad FROM schools"])
    out = run("q", db_path=tiny_db, max_retries=0, explain_answer=False)
    assert out["valid"] is False and out["retries"] == 0 and len(out["attempts"]) == 1


def test_subword_matching_finds_table_named_by_abbreviation(tmp_path):
    # BIRD regression: question says "SAT test takers", table is `satscores`.
    import sqlite3
    path = str(tmp_path / "s.sqlite")
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE satscores (cds TEXT, takers INTEGER); CREATE TABLE frpm (cds TEXT, n INTEGER);")
    con.close()
    assert "satscores" in select_tables("How many SAT test takers are there?", path)


def test_auto_mode_small_schema_is_full(tiny_db):
    ctx = retrieve_schema_context("math score", tiny_db, mode="auto")
    assert "audit_log" in ctx  # irrelevant table kept: schema is tiny, retrieval would only add risk


def test_retrieval_is_deterministic_across_hash_seeds(tmp_path):
    """Regression: BFS over a set picked different equal-length join paths per process,
    making eval results irreproducible. Diamond schema: a->b->d and a->c->d."""
    import os
    import sqlite3
    import subprocess
    import sys

    path = str(tmp_path / "d.sqlite")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE alpha (id INTEGER PRIMARY KEY);
        CREATE TABLE bravo (id INTEGER PRIMARY KEY, a INTEGER REFERENCES alpha(id));
        CREATE TABLE charlie (id INTEGER PRIMARY KEY, a INTEGER REFERENCES alpha(id));
        CREATE TABLE delta (id INTEGER PRIMARY KEY, b INTEGER REFERENCES bravo(id), c INTEGER REFERENCES charlie(id));
    """)
    con.close()
    code = ("from src.agents.schema_retriever import select_tables;"
            f"print(select_tables('alpha and delta', {path!r}))")
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout
        for seed in range(6)
    }
    assert len(outs) == 1, outs
