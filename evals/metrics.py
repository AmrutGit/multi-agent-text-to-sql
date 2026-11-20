"""Execution accuracy + automatic failure classification.

Correctness follows the BIRD official metric: the predicted and gold queries are
executed and the results compared as SETS of rows (column order matters, row order and
duplicates do not). Floats are rounded to 4 d.p. so sum-order noise doesn't count as an
error. We never compare SQL text: many different queries are equally correct.
"""
import sqlite3
import time

MAX_COMPARE_ROWS = 100_000


def _norm(v):
    return round(v, 4) if isinstance(v, float) else v


def execute_sql(db_path: str, sql: str, timeout_s: float = 30.0):
    """Run `sql` read-only, UNCAPPED (the agent's own 200-row cap would corrupt scoring).

    Returns (rows | None, error | None).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        rows = con.execute(sql).fetchmany(MAX_COMPARE_ROWS)
        return [tuple(_norm(v) for v in r) for r in rows], None
    except sqlite3.Error as e:
        return None, str(e)
    finally:
        con.close()


def compare(pred: list, gold: list) -> tuple[bool, str]:
    """(is_correct, mismatch_kind). mismatch_kind is '' when correct."""
    p, g = set(pred), set(gold)
    if p == g:
        return True, ""
    if not pred and gold:
        return False, "empty_result"
    if pred and gold and len(pred[0]) != len(gold[0]):
        return False, "wrong_columns"
    if p < g:
        return False, "missing_rows"
    if p > g:
        return False, "extra_rows"
    return False, "wrong_values"


# Ordered: first match wins. Patterns are matched on the LAST error the pipeline saw.
_ERROR_TYPES = [
    ("no such column", "hallucinated_column"),
    ("no such table", "hallucinated_table"),
    ("no such function", "hallucinated_function"),
    ("ambiguous column", "ambiguous_column"),
    ("syntax error", "syntax_error"),
    ("incomplete input", "syntax_error"),
    ("interrupted", "timeout"),
    ("only select queries", "blocked_by_guard"),
    ("forbidden keyword", "blocked_by_guard"),
    ("single sql statement", "blocked_by_guard"),
    ("empty sql", "no_sql"),
    ("misuse of aggregate", "aggregate_misuse"),
    ("datatype mismatch", "type_error"),
]


def classify_error(error: str) -> str:
    low = (error or "").lower()
    for needle, label in _ERROR_TYPES:
        if needle in low:
            return label
    return "other_sql_error"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Essential at small n: with 50
    questions, 60% accuracy is really 'somewhere in 46-72%'."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, centre - half), min(1.0, centre + half)
