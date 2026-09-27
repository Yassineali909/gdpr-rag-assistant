"""
Phase 10: FastAPI backend.

Exposes the RAG assistant as a REST API so it can be called by any front-end,
another service, or a test suite.

Run:  uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import ComplianceRAG

app = FastAPI(
    title="Compliance RAG Assistant",
    description="Ask questions about GDPR; get source-cited answers.",
    version="1.0.0",
)

# Build the RAG object once at startup (loads models into memory).
rag: ComplianceRAG | None = None


@app.on_event("startup")
def _startup():
    global rag
    rag = ComplianceRAG()


class Query(BaseModel):
    question: str


class Source(BaseModel):
    article: str
    score: float
    text: str


class Answer(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag is not None}


@app.post("/ask", response_model=Answer)
def ask(query: Query):
    result = rag.answer(query.question)
    return Answer(
        answer=result.answer,
        refused=result.refused,
        sources=[
            Source(article=a, score=s, text=t) for (a, s, t) in result.sources
        ],
    )
