from src.llm import complete


def write_sql(question: str, schema_context: str, prior_error: str | None = None) -> str:
    """Generate a SQLite SELECT for `question` using `schema_context`.

    YOUR WORK — the prompt is the product here:
      - system prompt: role, dialect=SQLite, READ-ONLY (SELECT only), output = SQL only
      - inject schema_context; if prior_error is set, include it so the model self-corrects
      - strip markdown fences from the output before returning
    Every line of your prompt should be a choice you can defend out loud.
    """
    raise NotImplementedError("Implement write_sql — see docstring.")
