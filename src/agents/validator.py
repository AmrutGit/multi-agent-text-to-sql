import sqlite3
from src.config import settings


def validate(sql: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty error when valid.

    YOUR WORK — where engineering judgment shows. Suggested checks:
      - read-only guard: reject INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA
      - syntax/plan check without running rows: `EXPLAIN <sql>` on a connection
      - (optional) confirm referenced tables exist
    A failing validation feeds back into write_sql(prior_error=...) — that retry
    loop is the single most impressive part of this repo. Build it deliberately.
    """
    raise NotImplementedError("Implement validate — see docstring.")
