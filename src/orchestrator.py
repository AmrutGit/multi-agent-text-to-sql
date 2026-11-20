"""The agent graph, as explicit hand-rolled Python (no framework) so every step is
visible and debuggable. The CONTRACT below is what the API and evals depend on.

CONTRACT — run(question) MUST return a dict shaped like:
    {
      "sql": str,          # final SQL produced
      "rows": list,        # executed result rows (read-only)
      "answer": str,       # natural-language answer
      "valid": bool,       # did validation pass
      "retries": int,      # how many self-correction passes were used
    }

FLOW:
    question
      -> retrieve_schema_context(question)        # RAG
      -> write_sql(question, ctx)                 # SQL writer
      -> validate(sql) --invalid--> write_sql(..., prior_error)  # loop up to max_sql_retries
      -> execute (read-only) -> rows
      -> explain(question, rows) -> answer
"""
import logging
import sqlite3
import time

from src.agents.explainer import explain
from src.agents.schema_retriever import retrieve_schema_context
from src.agents.sql_writer import NORTHWIND_HINTS, write_sql
from src.agents.validator import validate
from src.config import settings

log = logging.getLogger(__name__)


def _execute(sql: str, db_path: str | None = None) -> list:
    """Run `sql` on a READ-ONLY connection with a row cap and a wall-clock timeout."""
    con = sqlite3.connect(f"file:{db_path or settings.db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + settings.query_timeout_s
    # SQLite calls this every N VM instructions; returning non-zero aborts the query.
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchmany(settings.max_rows)]
    finally:
        con.close()


def _default_hints(db_path: str | None) -> str | None:
    return NORTHWIND_HINTS if (db_path or settings.db_path).endswith("northwind.sqlite") else None


def run(question: str, db_path: str | None = None, evidence: str | None = None,
        hints: str | None = None, explain_answer: bool = True,
        max_retries: int | None = None, schema_mode: str | None = None) -> dict:
    """question -> schema RAG -> write SQL -> validate/execute (self-correcting) -> explain.

    Extra keyword args exist for evaluation: another database (`db_path`), a per-question
    `evidence` hint, skipping the explainer (`explain_answer=False`: it doesn't affect SQL
    accuracy but adds latency/cost), and ablations (`max_retries`, `schema_mode`).
    The result also carries `attempts`: the SQL + error of every try, for failure analysis.
    """
    # Evidence often names the very columns/tables needed, so it informs retrieval too.
    context = retrieve_schema_context(f"{question} {evidence}" if evidence else question,
                                      db_path, schema_mode)
    hints = hints if hints is not None else _default_hints(db_path)

    sql, error, rows = "", None, []
    attempts_log: list[dict] = []
    retries = settings.max_sql_retries if max_retries is None else max_retries
    attempts = retries + 1  # first try + N corrections
    for attempt in range(attempts):
        sql = write_sql(question, context, prior_error=error, prior_sql=sql or None,
                        evidence=evidence, hints=hints)
        ok, error = validate(sql, db_path)
        if ok:
            try:
                rows = _execute(sql, db_path)
            except sqlite3.Error as e:  # runtime failure the planner couldn't see
                ok, error = False, f"Execution failed: {e}"
        attempts_log.append({"sql": sql, "error": None if ok else error})
        if ok:
            return {
                "sql": sql,
                "rows": rows,
                "answer": explain(question, rows) if explain_answer else "",
                "valid": True,
                "retries": attempt,
                "attempts": attempts_log,
            }
        log.info("attempt %d failed: %s", attempt + 1, error)

    # Out of retries: report honestly instead of explaining non-existent results.
    return {
        "sql": sql,
        "rows": [],
        "answer": f"I couldn't produce a valid query for that question. Last error: {error}",
        "valid": False,
        "retries": attempts - 1,
        "attempts": attempts_log,
    }
