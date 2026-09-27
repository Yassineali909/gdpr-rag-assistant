"""
Phase 5-6: Retrieve the most relevant chunks for a question.

Two upgrades over naive similarity search:

1. HYBRID SEARCH — combine dense vector search (good at meaning) with BM25 keyword
   search (good at exact terms like "Article 17" or "right to erasure"). Legal text
   needs both.

2. RERANKING — the initial retrievers are fast but rough. A cross-encoder reads the
   question and each candidate TOGETHER and scores true relevance, letting us reorder
   and keep only the best few chunks before they reach the LLM.
"""
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config
from src.ingest import build_documents
from src.embed_store import load_vector_store
from src.query_expansion import expand_query


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it."""
    return CrossEncoder(config.RERANKER_MODEL)


def build_hybrid_retriever() -> EnsembleRetriever:
    """
    Build an ensemble of BM25 (keyword) + Chroma (dense vector) retrievers.
    BM25 needs the raw documents in memory; the vector store is loaded from disk.
    """
    documents = build_documents()  # for BM25's in-memory keyword index

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = config.RETRIEVE_K

    vector_store = load_vector_store()
    dense = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVE_K})

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=config.HYBRID_WEIGHTS,
    )


def rerank(query: str, docs: list[Document], top_k: int = config.TOP_K):
    """
    Score each candidate against the query with the cross-encoder, sort by score,
    and return the top_k as (document, score) pairs.
    """
    if not docs:
        return []

    reranker = _get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def retrieve(query: str, retriever: EnsembleRetriever | None = None):
    """
    Full retrieval pipeline: hybrid search -> rerank.
    Returns a list of (document, relevance_score) tuples, best first.
    """
    if retriever is None:
        retriever = build_hybrid_retriever()

    expanded = expand_query(query)
    candidates = retriever.invoke(expanded)
    return rerank(expanded, candidates)


if __name__ == "__main__":
    # python -m src.retriever "right to be forgotten"
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How can someone delete their data?"
    results = retrieve(q)
    print(f"\nQuery: {q}\n" + "=" * 60)
    for doc, score in results:
        print(f"[score {score:.3f}] Article {doc.metadata.get('article')}")
        print(doc.page_content[:160], "...\n")
