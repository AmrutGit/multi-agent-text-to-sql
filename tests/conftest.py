"""Shared fixtures. Tests never call a real LLM: they patch the agents' `complete`."""
import os
import sqlite3

import pytest

from src.config import settings


def pytest_collection_modifyitems(config, items):
    # Only tests marked `northwind` need the downloaded DB; hermetic tests always run.
    if not os.path.exists(settings.db_path):
        skip = pytest.mark.skip(reason="run `make data` first")
        for item in items:
            if "northwind" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def fake_llm(monkeypatch):
    """Script LLM replies. `fake_llm(["sql1", "sql2", "answer"])` -> returns calls list."""
    def install(replies):
        calls, queue = [], list(replies)

        def fake(system, user):
            calls.append({"system": system, "user": user})
            return queue.pop(0)

        for mod in ("src.agents.sql_writer", "src.agents.explainer"):
            monkeypatch.setattr(f"{mod}.complete", fake)
        return calls
    return install


@pytest.fixture
def tiny_db(tmp_path):
    """A small hermetic DB (2 related tables) so most tests need no download."""
    path = str(tmp_path / "tiny.sqlite")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE schools (id INTEGER PRIMARY KEY, name TEXT, county TEXT, enrollment INTEGER);
        CREATE TABLE scores (school_id INTEGER, year INTEGER, math REAL,
                             FOREIGN KEY (school_id) REFERENCES schools(id));
        CREATE TABLE audit_log (id INTEGER PRIMARY KEY, note TEXT);
        INSERT INTO schools VALUES (1,'Alpha High','Marin',900),(2,'Beta Prep','Napa',400),
                                   (3,'Gamma Academy','Marin',650);
        INSERT INTO scores VALUES (1,2020,81.5),(2,2020,70.0),(3,2020,90.25),(1,2021,85.0);
    """)
    con.commit()
    con.close()
    return path
