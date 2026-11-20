import re

from src.llm import complete

SYSTEM_PROMPT = """\
You are an expert SQLite analyst. Convert the user's question into ONE SQLite query.

Rules:
- Dialect is SQLite (not MySQL/Postgres): use strftime() for dates, || to concatenate
  strings, LIMIT (not TOP). There is no ILIKE; use LIKE (case-insensitive for ASCII).
- Output a single read-only SELECT (a WITH ... SELECT is fine). Never write
  INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA/ATTACH.
- Use ONLY the tables and columns in the schema below. Quote names containing spaces
  with double quotes or backticks. Do not invent columns.
- Select only the columns needed to answer the question; add ORDER BY when the question
  says top/most/least/highest/lowest, and LIMIT when it asks for a fixed number.
- Respond with the SQL ONLY: no explanation, no comments, no markdown.
{hints}
Schema:
{schema}
"""

# Dataset-specific knowledge lives OUTSIDE the generic prompt so it can't leak into
# other databases (it would actively mislead the model on, say, a BIRD schema).
NORTHWIND_HINTS = """\
Database notes:
- Revenue/sales amount = UnitPrice * Quantity * (1 - Discount) from "Order Details".
- Dates are text beginning 'YYYY-MM-DD'; filter years with strftime('%Y', col) = '2016'.
- Products.Discontinued is TEXT: compare with '1' / '0'.
"""

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _clean(text: str) -> str:
    """Strip markdown fences / stray prose some models add despite instructions."""
    text = _FENCE.sub("", text.strip()).strip()
    m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    return text[m.start():].strip() if m else text


def write_sql(question: str, schema_context: str, prior_error: str | None = None,
              prior_sql: str | None = None, evidence: str | None = None,
              hints: str | None = None) -> str:
    """Generate a SQLite SELECT for `question` using `schema_context`.

    `evidence` is an optional per-question hint (BIRD supplies one, e.g. a formula or
    value mapping). On a retry, the failed SQL and the error are included so the model
    repairs the specific problem rather than regenerating from scratch.
    """
    system = SYSTEM_PROMPT.format(schema=schema_context, hints=("\n" + hints) if hints else "")
    user = f"Question: {question}"
    if evidence:
        user += f"\nHint: {evidence}"
    if prior_error:
        user += (
            f"\n\nYour previous attempt failed.\nPrevious SQL:\n{prior_sql}\n"
            f"Error: {prior_error}\nReturn a corrected query."
        )
    return _clean(complete(system, user))
