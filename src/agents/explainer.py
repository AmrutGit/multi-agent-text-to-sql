from src.llm import complete


def explain(question: str, rows: list) -> str:
    """Turn the result `rows` into a short natural-language answer to `question`.

    YOUR WORK: design a tight prompt. Guard the empty-result and large-result cases
    (don't dump 10k rows into the model). Keep the answer grounded in the rows only.
    """
    raise NotImplementedError("Implement explain — see docstring.")
