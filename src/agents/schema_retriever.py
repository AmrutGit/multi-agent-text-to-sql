from src.db.schema import get_full_schema


def retrieve_schema_context(question: str) -> str:
    """The RAG step. Return ONLY the schema slice relevant to `question`.

    YOUR WORK — and a real talking point. Northwind's full schema fits in a prompt,
    so the naive move is to dump get_full_schema(). Do better and be ready to say why:
      - rank/select tables by relevance to the question (keyword, embedding, or LLM pick)
      - optionally attach a few question->SQL examples (few-shot)
      - explain the tradeoff: full-schema = simple but bloats context & hurts accuracy
        as schemas grow; retrieval = scalable but can miss a needed table.
    Start simple (return get_full_schema()), then make it actually retrieve.
    """
    raise NotImplementedError("Implement retrieve_schema_context — see docstring.")
