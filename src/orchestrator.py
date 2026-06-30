"""The agent graph. I've fixed the CONTRACT (so the API + evals work against it);
you implement the flow. Keep it explicit, hand-rolled Python for now — no framework
magic — so you can explain exactly what happens. (Optional later commit: port this
to Google ADK as a second variant — rare as a public repo, strong differentiator.)

CONTRACT — run(question) MUST return a dict shaped like:
    {
      "sql": str,          # final SQL produced
      "rows": list,        # executed result rows (read-only)
      "answer": str,       # natural-language answer
      "valid": bool,       # did validation pass
      "retries": int,      # how many self-correction passes were used
    }

INTENDED FLOW (your implementation):
    question
      -> retrieve_schema_context(question)        # RAG
      -> write_sql(question, ctx)                 # SQL writer
      -> validate(sql) --invalid--> write_sql(..., prior_error)  # loop up to max_sql_retries
      -> execute (read-only) -> rows
      -> explain(question, rows) -> answer
"""
import sqlite3
from src.config import settings
from src.agents.schema_retriever import retrieve_schema_context
from src.agents.sql_writer import write_sql
from src.agents.validator import validate
from src.agents.explainer import explain


def _execute(sql: str) -> list:
    con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def run(question: str) -> dict:
    raise NotImplementedError(
        "Implement run() following the CONTRACT and INTENDED FLOW above. "
        "_execute() is provided as read-only plumbing."
    )
