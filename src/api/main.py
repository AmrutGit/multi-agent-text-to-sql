import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.orchestrator import run

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Multi-Agent Text-to-SQL")


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class QueryResponse(BaseModel):
    sql: str
    rows: list[dict]
    answer: str
    valid: bool
    retries: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(q: Query):
    try:
        return run(q.question)  # extra `attempts` key is dropped by response_model
    except RuntimeError as e:  # LLM misconfigured / blocked / unavailable
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logging.getLogger(__name__).exception("query failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.") from e
