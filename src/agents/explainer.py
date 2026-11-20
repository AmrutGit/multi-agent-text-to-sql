import json

from src.config import settings
from src.llm import complete

SYSTEM_PROMPT = """\
You answer a user's question using ONLY the query result rows provided.
- Be concise: 1-3 sentences. Lead with the answer.
- Use only facts present in the rows. Never guess or add outside knowledge.
- If the rows are a truncated sample, say the figures cover the sample shown.
"""


def explain(question: str, rows: list) -> str:
    """Turn the result `rows` into a short natural-language answer to `question`.

    Guards: empty results are answered without an LLM call (nothing to hallucinate
    from), and large results are truncated so context stays small and cheap.
    """
    if not rows:
        return "The query ran successfully but returned no matching rows."

    limit = settings.explain_row_limit
    sample = rows[:limit]
    note = ""
    if len(rows) > limit:
        note = f"\n(Showing the first {limit} of {len(rows)} rows.)"
    payload = json.dumps(sample, default=str)  # default=str handles dates/bytes
    user = f"Question: {question}\nRows (JSON):\n{payload}{note}"
    return complete(SYSTEM_PROMPT, user).strip()
