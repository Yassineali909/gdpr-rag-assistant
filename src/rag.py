"""
Phase 4, 7, 8: The RAG chain that ties everything together.

Given a question:
  1. Retrieve + rerank the most relevant chunks (src/retriever.py)
  2. OUT-OF-SCOPE REFUSAL (Phase 8): if nothing is relevant enough, refuse instead
     of hallucinating.
  3. Build a grounded prompt and ask the local LLM.
  4. CITATIONS (Phase 7): the prompt forces the model to cite the Articles it used,
     and we also return the source chunks so the UI can show them.
"""
from dataclasses import dataclass, field

from langchain_ollama import OllamaLLM

import config
from src.retriever import build_hybrid_retriever, retrieve


SYSTEM_PROMPT = """You are a careful compliance assistant. Answer the user's question \
using ONLY the numbered context passages provided below. 

Rules:
- Base every statement on the context. Do NOT use outside knowledge.
- Cite the Article number in parentheses after each claim, e.g. "(Article 17)".
- If the context does not contain the answer, reply EXACTLY with:
  "I can't answer that from the provided regulation."
- Be concise and precise. Do not invent Article numbers.

Context passages:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResult:
    answer: str
    sources: list = field(default_factory=list)  # list of (article, score, text)
    refused: bool = False


class ComplianceRAG:
    """Reusable RAG object — build the retriever once, answer many questions."""

    def __init__(self):
        print("[rag] Initializing (loading retriever + LLM)...")
        self.retriever = build_hybrid_retriever()
        self.llm = OllamaLLM(model=config.LLM_MODEL, temperature=0.0, num_predict=256, base_url=config.OLLAMA_HOST)
        print("[rag] Ready.")

    def answer(self, question: str) -> RAGResult:
        ranked = retrieve(question, self.retriever)

        # --- Phase 8: out-of-scope refusal ---
        best_score = ranked[0][1] if ranked else -999
        if not ranked or best_score < config.MIN_RELEVANCE_SCORE:
            return RAGResult(
                answer="I can't answer that from the provided regulation.",
                sources=[],
                refused=True,
            )

        # --- Build the grounded context block with numbered, cited passages ---
        context_lines = []
        sources = []
        for i, (doc, score) in enumerate(ranked, start=1):
            article = doc.metadata.get("article", "N/A")
            context_lines.append(f"[{i}] (Article {article}) {doc.page_content}")
            sources.append((article, float(score), doc.page_content))
        context = "\n\n".join(context_lines)

        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        answer = self.llm.invoke(prompt).strip()

        refused = answer.lower().strip().startswith("i can't answer that from the provided regulation")
        return RAGResult(answer=answer, sources=sources, refused=refused)


def _cli():
    """Command-line entry point: python -m src.rag "your question" """
    import sys

    question = " ".join(sys.argv[1:]) or "How can someone request erasure of their data?"
    rag = ComplianceRAG()
    result = rag.answer(question)

    print("\n" + "=" * 60)
    print("Q:", question)
    print("=" * 60)
    print(result.answer)
    if result.sources and not result.refused:
        print("\nSources:")
        for article, score, _ in result.sources:
            print(f"  - Article {article}  (relevance {score:.3f})")


if __name__ == "__main__":
    _cli()
