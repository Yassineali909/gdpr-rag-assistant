# Compliance RAG Assistant

A **production-grade Retrieval-Augmented Generation (RAG)** assistant that answers
plain-language questions about dense regulatory texts (starting with the **GDPR**),
returns **source-cited** answers, and **refuses to answer** when the question falls
outside the documents it was given.

Built to run **100% locally and free** using [Ollama](https://ollama.com) — no API keys, no cloud cost.

---

## Why this project matters (the engineering, not the topic)

Anyone can build a "chat with a PDF" toy. This project is a portfolio piece because it
includes the **production layer** that separates a real AI engineer from a demo-builder:

| Feature | What it demonstrates |
|---|---|
| Structure-aware chunking | You understand that chunking quality drives RAG quality |
| Hybrid retrieval (BM25 + dense vectors) | You know pure semantic search misses exact terms like "Article 17" |
| Cross-encoder reranking | You can push retrieval quality beyond naive top-k |
| Source-grounded citations | You understand hallucination control |
| Out-of-scope refusal | You build for production safety, not just the happy path |
| RAGAS evaluation | You **measure** quality instead of claiming "it works well" |
| FastAPI + Docker + Streamlit | You can ship a system, not just a notebook |

**Keyword cluster earned:** RAG · vector databases · hybrid search · reranking · LLMs ·
prompt engineering · evaluation (RAGAS) · hallucination control · FastAPI · Docker.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
   User question  ───►  │  1. Hybrid Retriever                      │
                        │     • BM25 (keyword)                      │
                        │     • Dense vectors (ChromaDB)            │
                        │           │                               │
                        │           ▼                               │
                        │  2. Cross-encoder Reranker               │
                        │           │                               │
                        │           ▼                               │
                        │  3. LLM (Mistral via Ollama)             │
                        │     • grounded answer + citations        │
                        │     • out-of-scope refusal               │
                        └──────────────────────────────────────────┘
                                    │
                                    ▼
                     Answer + cited Articles (e.g. "Art. 17(1)")
```

---

## Project structure

```
compliance-rag/
├── README.md                 ← you are here
├── requirements.txt          ← Python dependencies
├── .gitignore
├── config.py                 ← all settings in one place
├── data/
│   └── sample_gdpr.txt       ← small simplified sample so you can test immediately
├── scripts/
│   └── download_gdpr.py       ← fetch the FULL official GDPR text
├── src/
│   ├── __init__.py
│   ├── ingest.py             ← Phase 1–2: load + structure-aware chunking
│   ├── embed_store.py        ← Phase 3: embeddings + ChromaDB
│   ├── retriever.py          ← Phase 5–6: hybrid retrieval + reranking
│   └── rag.py                ← Phase 4,7,8: RAG chain, citations, refusal
├── evaluation/
│   └── test_questions.json   ← eval question set
├── evaluate.py               ← Phase 9: RAGAS evaluation
├── api.py                    ← Phase 10: FastAPI backend
├── app.py                    ← Phase 10: Streamlit front-end
└── build_index.py            ← one command to ingest + embed everything
```

---

## Setup (WSL / Linux)

```bash
# 1. Make sure Ollama is running with a model pulled
ollama pull mistral

# 2. Create + activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Run it

```bash
# Step 1 — build the search index (ingest → chunk → embed → store)
python build_index.py

# Step 2a — ask a question from the command line
python -m src.rag "What are the rules for erasing personal data?"

# Step 2b — OR launch the web UI
streamlit run app.py

# Step 2c — OR run the API
uvicorn api:app --reload
```

---

## Build phases (roadmap)

- [x] **Phase 0** — Environment setup (WSL, Ollama, venv)
- [ ] **Phase 1–2** — Ingest + structure-aware chunking (`src/ingest.py`)
- [ ] **Phase 3** — Embed + store in ChromaDB (`src/embed_store.py`)
- [ ] **Phase 4** — Basic end-to-end RAG (`src/rag.py`)
- [ ] **Phase 5** — Hybrid retrieval (`src/retriever.py`)
- [ ] **Phase 6** — Reranking (`src/retriever.py`)
- [ ] **Phase 7** — Citations (`src/rag.py`)
- [ ] **Phase 8** — Out-of-scope refusal (`src/rag.py`)
- [ ] **Phase 9** — RAGAS evaluation (`evaluate.py`)
- [ ] **Phase 10** — Deploy: FastAPI + Streamlit + Docker

The starter code already wires up Phases 1–8. Your job is to run each phase, understand it,
tune it, and then add evaluation + deployment.

---

## Note on the data

`data/sample_gdpr.txt` contains a **small, simplified** set of articles written in plain
language so the pipeline runs out of the box. **Before showing this project off, replace it
with the full official text** using:

```bash
python scripts/download_gdpr.py
```

Official GDPR text is published by the EU on EUR-Lex and is free to reuse.
## Engineering decisions & known limitations (measured)

Every claim below is backed by a number produced by the eval harness (`evaluate.py`),
not by intuition. This section documents what was fixed, what was ruled out, and what
remains a known limitation — because a system whose failures you can name is worth more
than one that claims to have none.

### Citation-integrity bug: recital contamination (found & fixed)

The full GDPR text (297 chunks after chunking) interleaves 99 numbered Articles with
~170 unnumbered recitals. An early version of the ingest regex,
`^\s*Article\s+(\d+[a-z]?)\b.*$`, matched recital cross-references such as
"Article 16(2) TFEU mandates…" and, because no clean `Article N` header followed inside
the recital block, tagged **267 of 297 chunks as "Article 16."** The effect was severe:
retrieval surfaced mislabeled recital text at the top of the ranking and citations
pointed at non-existent article sub-numbers.

- **Fix 1 (regex):** require the header to stand alone on its line —
  `^\s*Article\s+(\d+[a-z]?)\s*$`. A trailing `\b.*$` allowed any line *beginning*
  with "Article N" to match; `\s*$` requires the line to be *only* the header.
- **Fix 2 (recital exclusion):** everything before the first real Article header is
  recitals/Treaty references, not articles — excluded from the searchable index to
  protect citation integrity.
- **Result:** 297 chunks, Article 16 reduced to exactly 1 (the real "right to
  rectification"), 99 distinct article tags matching the 99 real articles.
- **Process note:** the fix was in the code but the on-disk vector store was stale — it
  had been built by the pre-fix ingest. The lesson: `Chroma.from_documents` adds to an
  existing collection, so the store must be deleted (`rm -rf chroma_db/`) and rebuilt,
  not rebuilt on top of the old one.

### Retrieval recall on lay-language questions (fixed via query expansion)

Questions phrased in everyday language failed to retrieve the correct article, because
BM25 needs shared keywords and the small embedder (`bge-small`) maps street language far
from legalese. Two concrete failures:

- *"If a firm loses control of customer records to a hacker, how fast must it tell the
  regulator?"* — the correct Article 33 was retrieved but the cross-encoder scored it
  **0.003**, below the refusal gate, so a valid question was wrongly refused.
- *"How do I sue for damages in court?"* — Article 82 (compensation) **never entered the
  candidate pool at all.**

**Ruled out first (with evidence):** raising `RETRIEVE_K` from 8 → 20 was tested
directly; Article 82 *still* did not appear in the 38-candidate pool. This proved the
problem was vocabulary mismatch, not retrieval breadth — so widening was not the fix.

**Fix:** a deterministic static query-expansion layer (`src/query_expansion.py`) that
appends GDPR legal vocabulary triggered by lay terms (e.g. "sue/damages/court" →
"compensation, liability, judicial remedy"). Zero added latency, fully explainable.

- **Result:** retrieval hit rate rose from **62% → 89%** on the hardened eval set.
  Article 33's score went 0.003 → 0.995; Article 82 went from absent to retrieved.
- **Limitation (documented, not hidden):** the map was written with knowledge of the
  domain the eval probes, so passing the eval demonstrates the *mechanism* works — it
  does **not** prove generalization to unseen phrasings. A clean generalization test
  needs a held-out question set whose wording was not consulted while building the map.
  The production upgrade is an LLM-based rewrite (the retriever calls `expand_query()`
  and is agnostic to how expansion is produced).

### Refusal gate calibration (measured, left unchanged)

`MIN_RELEVANCE_SCORE = 0.15` was inherited as a guessed constant. Rather than tune it
blind, the top reranker score was measured for every eval question:

- Genuine in-scope questions scored **0.63–1.00**.
- Clean out-of-scope questions ("capital of France", "sourdough") scored **0.00–0.006**.
- The one borderline case (a vague, compound multi-article question) scored **0.135**.

The separation between the lowest in-scope score (0.63) and the highest score among
refused questions (0.135) is ~0.5 — a wide, comfortable margin with the 0.15 gate
sitting cleanly inside it. **Conclusion: leave 0.15 unchanged**, now confirmed by data
rather than assumed.

### Multi-article / abstract queries (known limitation)

*"What must a company prove about how it handles data, and what makes processing legal
in the first place?"* — a compound question spanning Articles 5 and 6, phrased
abstractly with no keywords the static expansion map catches. It scores 0.135 and is
refused (`got []`).

Two honest readings, both stated:
1. It is a **coverage gap** in the static expansion map — abstract multi-concept
   phrasing needs semantic (LLM) expansion or query decomposition, not keyword matching.
2. Arguably the **refusal is correct behavior**: a compliance assistant declining a
   vague, compound question rather than guessing is defensible. The eval counts it as a
   miss because it expected Article 6, but the design tradeoff is worth noting.

### Evaluation: local LLM-as-judge (built to replace RAGAS)

RAGAS was the intended evaluation tool but **does not import on this stack**: RAGAS 0.4.3
(the newest version resolvable on Python 3.14 / langchain 1.x) fails at import with
`from langchain_community.chat_models.vertexai import ChatVertexAI` — a module removed in
the langchain-community split. No newer RAGAS resolves.

Rather than downgrade the (now working) langchain pipeline to satisfy RAGAS, the two
metrics were implemented directly against the local Ollama model (`src/llm_judge.py`):

- **Faithfulness** — are the answer's claims supported by the retrieved context?
  (hallucination check)
- **Answer-relevancy** — does the answer address the question?

**Judge failure modes found and handled:**
- The relevancy judge initially scored a correct Article 33 answer **0.0**, inventing a
  distinction between "notification" and "reporting." Diagnosed as **under-prompting**,
  not model weakness: adding a one-shot paraphrase example to the prompt corrected all
  affected cases to 1.0 — tested *before* reaching for a larger judge.
- Faithfulness scores **1.00 across all questions.** This is treated with suspicion, not
  celebrated: a small (3B) judge returning "all claims supported" everywhere likely
  reflects judge leniency as much as pipeline grounding. Validating it requires a
  stronger judge or a deliberately-planted unfaithful answer as a control — documented
  as the next step.

The harness is judge-agnostic (`config.LLM_MODEL` swaps the judge). Scores are reported
as **directional**, with their limits stated, rather than as ground truth.

### Summary of current measured state

| Metric | Value | Notes |
|---|---|---|
| Retrieval hit rate | 8/9 (89%) | 1 documented miss (compound abstract query) |
| Refusal accuracy | 2/2 (100%) | Clean out-of-scope correctly refused |
| Refusal gate | 0.15 | Confirmed by ~0.5 score separation |
| Faithfulness (local judge) | 1.00 | Directional; possibly judge-lenient |
| Answer-relevancy (local judge) | 1.00 | After prompt hardening (was 0.63) |

## Deployment (Docker)

The FastAPI backend is containerized. **Ollama runs on the host**, not in the
container, so the container reaches it via a host-gateway mapping.

```bash
# 1. Ensure Ollama is running on the host with the model pulled
ollama serve &          # if not already running
ollama pull llama3.2:3b

# 2. Build the image (installs pinned deps, builds the search index)
docker build -t compliance-rag .

# 3. Run, mapping the container to the host's Ollama
docker run --rm -p 8000:8000 \
    --add-host=host.docker.internal:host-gateway \
    compliance-rag

# 4. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What is the right to be forgotten?"}'
```

**Notes:**
- Dependencies are pinned via `requirements-lock.txt` (frozen from the working
  dev environment) so the container reproduces the tested stack exactly.
- `OLLAMA_HOST` defaults to `host.docker.internal:11434` in the image; override
  with `-e OLLAMA_HOST=...` to point elsewhere.
- The search index is built into the image at build time. A production setup
  would mount it as a volume instead (documented future improvement).
- LLM and judge both run on `llama3.2:3b` via Ollama (see `config.LLM_MODEL`).
