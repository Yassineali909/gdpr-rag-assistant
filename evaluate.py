"""
Phase 9: Evaluate retrieval + answer quality.

Two layers:

1. A quick BUILT-IN check (no extra services): for each test question, did the right
   Article show up in the sources? And did out-of-scope questions get refused?
   This alone gives you a concrete metric to report and improve.

2. A RAGAS scaffold (optional, more advanced): measures faithfulness and answer
   relevance. RAGAS normally uses an LLM judge — the hook is provided; wire it to your
   local Ollama or an API when you're ready.

Run:  python evaluate.py
"""
import json
from pathlib import Path

from src.rag import ComplianceRAG

TEST_FILE = Path(__file__).parent / "evaluation" / "test_questions.json"


def run_builtin_eval():
    tests = json.loads(TEST_FILE.read_text())
    rag = ComplianceRAG()

    hits = 0
    refusal_correct = 0
    refusal_total = 0
    in_scope_total = 0

    print("\n" + "=" * 70)
    print("BUILT-IN EVALUATION")
    print("=" * 70)

    for t in tests:
        q = t["question"]
        expected = t["expected_article"]
        result = rag.answer(q)
        found_articles = [a for (a, _, _) in result.sources]

        if expected == "OUT_OF_SCOPE":
            refusal_total += 1
            ok = result.refused
            refusal_correct += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] (refusal) {q}")
        else:
            in_scope_total += 1
            ok = expected in found_articles
            hits += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] expect Art.{expected}, got {found_articles} | {q}")

    print("-" * 70)
    if in_scope_total:
        print(f"Retrieval hit rate : {hits}/{in_scope_total} "
              f"({100*hits/in_scope_total:.0f}%)")
    if refusal_total:
        print(f"Refusal accuracy   : {refusal_correct}/{refusal_total} "
              f"({100*refusal_correct/refusal_total:.0f}%)")
    print("=" * 70)
    print(
        "\nTip: this is the number you improve. Add reranking, tune HYBRID_WEIGHTS or "
        "MIN_RELEVANCE_SCORE in config.py, re-run, and watch it go up."
    )


# ---------------------------------------------------------------------------
# Optional: RAGAS scaffold. Uncomment and wire an LLM/embeddings judge to use.
# ---------------------------------------------------------------------------
def run_ragas_eval():
    """
    LLM-as-judge evaluation: faithfulness + answer-relevancy on the in-scope
    questions, scored by the local Ollama model (see src/llm_judge.py for why this
    replaces RAGAS, which does not import on this stack).

    Refusal cases are skipped: there is no answer to ground or rate when the
    assistant correctly declines.
    """
    from src.llm_judge import LLMJudge

    tests = json.loads(TEST_FILE.read_text())
    rag = ComplianceRAG()
    judge = LLMJudge()

    print(chr(10) + "=" * 70)
    print("LLM-AS-JUDGE EVALUATION  (faithfulness + answer-relevancy)")
    print("=" * 70)
    print("Judge model:", __import__("config").LLM_MODEL,
          "(local) — scores are DIRECTIONAL, not ground truth.")
    print("NOTE: small local judge. Relevancy misjudged correct answers until the")
    print("      prompt was hardened; faithfulness scores 1.00 and may be too lenient")
    print("      to catch subtle grounding errors. A stronger judge is the next step.")
    print("-" * 70)

    faith_scores = []
    rel_scores = []

    for t in tests:
        if t["expected_article"] == "OUT_OF_SCOPE":
            continue  # nothing to judge on a refusal

        q = t["question"]
        result = rag.answer(q)

        if result.refused:
            # In-scope question the pipeline refused (e.g. the multi-article case).
            # Not a judge failure — record as unscored so it doesn't fake a number.
            print(f"[refused, skipped] {q[:55]}")
            continue

        contexts = [text for (_, _, text) in result.sources]
        f = judge.faithfulness(result.answer, contexts)
        r = judge.answer_relevancy(q, result.answer)

        f_str = f"{f:.2f}" if f is not None else "  ??"
        r_str = f"{r:.2f}" if r is not None else "  ??"
        print(f"faith={f_str}  rel={r_str}  | {q[:50]}")

        if f is not None:
            faith_scores.append(f)
        if r is not None:
            rel_scores.append(r)

    print("-" * 70)
    if faith_scores:
        print(f"Mean faithfulness    : {sum(faith_scores)/len(faith_scores):.2f} "
              f"(n={len(faith_scores)})")
    if rel_scores:
        print(f"Mean answer-relevancy: {sum(rel_scores)/len(rel_scores):.2f} "
              f"(n={len(rel_scores)})")
    print("=" * 70)
    print("'??' = judge returned unparseable output for that item (marked unscored, "
          "not counted as 0).")



if __name__ == "__main__":
    run_builtin_eval()
    run_ragas_eval()
