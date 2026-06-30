from fastapi import FastAPI
from pydantic import BaseModel
from src.orchestrator import run

app = FastAPI(title="Multi-Agent Text-to-SQL")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: Query):
    try:
        return run(q.question)
    except NotImplementedError as e:
        return {"error": "not implemented yet", "detail": str(e)}
