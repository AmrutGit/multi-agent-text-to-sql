import sqlite3
from src.config import settings


def get_full_schema() -> str:
    """Return the full CREATE TABLE DDL for every table. Plumbing — done for you."""
    con = sqlite3.connect(settings.db_path)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        return "\n\n".join(r[0] for r in rows)
    finally:
        con.close()
