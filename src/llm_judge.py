"""
LLM-as-judge evaluation: faithfulness + answer-relevancy, scored by the local
Ollama model already used by the RAG pipeline. No external eval service.

WHY THIS EXISTS INSTEAD OF RAGAS
RAGAS 0.4.3 (the newest version this Python 3.14 / langchain 1.x stack accepts)
fails at import: its llms/base.py does an unconditional
`from langchain_community.chat_models.vertexai import ChatVertexAI`, a module that
no longer exists after the langchain-community split. No newer RAGAS resolves on
this stack. So the two metrics are implemented directly — they are concepts, not a
library:

  faithfulness      : is every claim in the answer supported by the retrieved
                      context? (hallucination check — the metric that matters most
                      for a compliance tool that must not invent article content)
  answer_relevancy  : does the answer actually address the question asked?

HONESTY ABOUT THE JUDGE
The judge is Mistral-7B (local). A 7B judge is noisy: it will occasionally misjudge
a grounded claim, and scores can wobble run to run. These numbers are DIRECTIONAL,
not ground truth. A stronger judge (larger local model or an API model used only for
eval) would tighten them. The harness is judge-agnostic: swap the model in
config.LLM_MODEL and rerun.
"""
import json
import re

from langchain_ollama import OllamaLLM

import config


FAITHFULNESS_PROMPT = """You are evaluating whether an ANSWER is fully supported by \
the CONTEXT. Judge ONLY grounding, not whether the answer is good.

Break the ANSWER into its factual claims. A claim is "supported" if the CONTEXT
states or directly implies it. Count claims that are unsupported or contradicted.

Return ONLY a JSON object, no other text:
{{"supported": <int>, "total": <int>, "reason": "<one short sentence>"}}

CONTEXT:
{context}

ANSWER:
{answer}

JSON:"""

RELEVANCY_PROMPT = """You are evaluating whether an ANSWER addresses the QUESTION. \
Judge ONLY relevance (does it respond to what was asked), NOT correctness or grounding.

IMPORTANT: paraphrases and synonyms count as relevant. If the question asks about
"reporting to the authority" and the answer discusses "notifying the supervisory
authority", that is the SAME topic and is fully relevant. Do not invent distinctions
between synonymous terms. Judge the substance, not the wording.

Return ONLY a JSON object, no other text:
{{"score": <float 0.0-1.0>, "reason": "<one short sentence>"}}

Scoring guide:
- 1.0 : the answer responds to what the question asked (even if paraphrased)
- 0.5 : the answer is partly on-topic but misses part of the question
- 0.0 : the answer is genuinely off-topic or a non-answer

EXAMPLE:
QUESTION: How quickly must a data breach be reported to the authority?
ANSWER: The controller must notify the supervisory authority without undue delay and
where feasible within 72 hours (Article 33).
CORRECT JUDGMENT: {{"score": 1.0, "reason": "Directly answers the timing of breach notification"}}

QUESTION:
{question}

ANSWER:
{answer}

JSON:"""


def _extract_json(text: str) -> dict | None:
    """
    Pull the first JSON object out of the model's reply. A 7B model often wraps
    JSON in prose or ```json fences, so we don't trust the whole string to parse.
    Returns None on failure — caller marks the item unscored rather than fake a 0.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text)
    # Grab the first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMJudge:
    """Scores faithfulness + answer-relevancy with the local Ollama model."""

    def __init__(self):
        # temperature 0 for reproducibility; judge should be as stable as a 7B allows
        self.llm = OllamaLLM(model=config.LLM_MODEL, temperature=0.0, num_predict=200, base_url=config.OLLAMA_HOST)

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        context = "\n\n".join(contexts)
        raw = self.llm.invoke(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        )
        data = _extract_json(raw)
        if not data or "supported" not in data or "total" not in data:
            return None
        total = data["total"]
        if not total:  # no claims -> vacuously faithful
            return 1.0
        return max(0.0, min(1.0, data["supported"] / total))

    def answer_relevancy(self, question: str, answer: str) -> float | None:
        raw = self.llm.invoke(
            RELEVANCY_PROMPT.format(question=question, answer=answer)
        )
        data = _extract_json(raw)
        if not data or "score" not in data:
            return None
        try:
            return max(0.0, min(1.0, float(data["score"])))
        except (TypeError, ValueError):
            return None


if __name__ == "__main__":
    # Smoke test on one hand-made example — no pipeline needed.
    judge = LLMJudge()
    ctx = ["Article 17: Individuals have the right to erasure of their personal data "
           "without undue delay in certain situations."]
    ans = "People can request their data be erased without undue delay (Article 17)."
    print("faithfulness:", judge.faithfulness(ans, ctx))
    print("relevancy   :", judge.answer_relevancy("What is the right to erasure?", ans))
