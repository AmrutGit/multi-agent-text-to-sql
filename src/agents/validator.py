import re
import sqlite3

from src.config import settings

# Statements/keywords that can change state or reach outside the DB.
# REPLACE is intentionally absent: REPLACE() is a legit string function, and
# REPLACE INTO is already blocked by the must-start-with-SELECT/WITH check.
_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
    "pragma", "vacuum", "reindex", "truncate", "begin", "commit", "rollback",
    "savepoint", "release",
}

# One left-to-right pass over quoted text AND comments, so a '--' or ';' inside a
# string literal is not mistaken for a comment/terminator (and vice versa).
_TOKEN = re.compile(
    r"(?P<str>'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`|\[[^\]]*\])"
    r"|(?P<com>--[^\n]*|/\*.*?\*/)",
    re.DOTALL,
)


def _skeleton(sql: str) -> str:
    """SQL with comments and quoted text blanked out, for keyword inspection only."""
    return _TOKEN.sub(lambda m: "''" if m.group("str") else " ", sql)


def validate(sql: str, db_path: str | None = None) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty error when valid.

    Defence in depth, cheapest check first:
      1. static guard  - single statement, SELECT/WITH only, no write keywords
      2. planner check - `EXPLAIN` on a read-only connection catches unknown
         tables/columns and syntax errors WITHOUT executing the query
    The error text is written for an LLM: it is fed straight back to the SQL writer.
    The execution connection is also read-only (see orchestrator), so even if this
    guard had a hole the database could not be modified.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL."

    skeleton = _skeleton(sql).strip().rstrip(";").strip()
    if not skeleton:
        return False, "Empty SQL."
    if ";" in skeleton:
        return False, "Only a single SQL statement is allowed."

    first = skeleton.split(None, 1)[0].lower()
    if first not in ("select", "with"):
        return False, f"Only SELECT queries are allowed (statement starts with '{first.upper()}')."

    # Whole-word match so a column like "UpdatedAt" or "Discount" doesn't trip it.
    bad = sorted(_FORBIDDEN & set(re.findall(r"[a-z_]+", skeleton.lower())))
    if bad:
        return False, f"Forbidden keyword(s) in a read-only query: {', '.join(k.upper() for k in bad)}."

    con = sqlite3.connect(f"file:{db_path or settings.db_path}?mode=ro", uri=True)
    try:
        con.execute(f"EXPLAIN {sql.strip().rstrip(';')}")
    except sqlite3.Error as e:
        return False, f"SQLite rejected the query: {e}"
    finally:
        con.close()
    return True, ""
