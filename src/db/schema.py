import sqlite3
from functools import lru_cache

from src.config import settings


def _path(db_path: str | None) -> str:
    return db_path or settings.db_path


def _q(name: str) -> str:
    """Quote an identifier for use in SQL (BIRD has names with spaces and quotes)."""
    return '"' + name.replace('"', '""') + '"'


def _connect(db_path: str) -> sqlite3.Connection:
    # Read-only: schema introspection must never be able to modify the DB.
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")  # BIRD has dirty bytes
    return con


@lru_cache(maxsize=64)
def _table_schemas(db_path: str) -> dict[str, str]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        return {name: sql for name, sql in rows}
    finally:
        con.close()


def get_table_schemas(db_path: str | None = None) -> dict[str, str]:
    """Map table name -> CREATE TABLE DDL (base tables only).

    Views are deliberately excluded: in the Northwind build several are broken
    (return 0 rows, or use `+` for string concat, which is wrong in SQLite) and
    would teach the model bad patterns. Cached because the schema is static.
    """
    return _table_schemas(_path(db_path))


@lru_cache(maxsize=64)
def _foreign_keys(db_path: str) -> dict[str, set[str]]:
    con = _connect(db_path)
    try:
        graph: dict[str, set[str]] = {t: set() for t in _table_schemas(db_path)}
        for table in list(graph):
            for row in con.execute(f"PRAGMA foreign_key_list({_q(table)})"):
                other = row[2]  # referenced table
                if other in graph:
                    graph[table].add(other)
                    graph[other].add(table)
        return graph
    finally:
        con.close()


def get_foreign_keys(db_path: str | None = None) -> dict[str, set[str]]:
    """Undirected FK graph: table -> set of tables it is joined to by a foreign key."""
    return _foreign_keys(_path(db_path))


@lru_cache(maxsize=64)
def _empty_tables(db_path: str) -> frozenset[str]:
    con = _connect(db_path)
    try:
        return frozenset(
            t for t in _table_schemas(db_path)
            if con.execute(f"SELECT 1 FROM {_q(t)} LIMIT 1").fetchone() is None
        )
    finally:
        con.close()


def get_empty_tables(db_path: str | None = None) -> frozenset[str]:
    """Tables with zero rows: they can't answer anything, so retrieval down-weights them."""
    return _empty_tables(_path(db_path))


def get_full_schema(db_path: str | None = None) -> str:
    """Return the CREATE TABLE DDL for every base table."""
    return "\n\n".join(get_table_schemas(db_path).values())
