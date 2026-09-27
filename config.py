"""
Central configuration for the Compliance RAG Assistant.
Keeping every tunable knob in one place makes the project easy to reason about
and easy to explain in an interview.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"

# The source document the assistant answers from.
# Start with the sample; swap to "full_gdpr.txt" after running scripts/download_gdpr.py
SOURCE_FILE = DATA_DIR / "full_gdpr.txt"

# --- Models (all local via Ollama / HuggingFace) ---
LLM_MODEL = "llama3.2:3b"                      # served by Ollama on localhost:11434
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # small, fast, good quality
RERANKER_MODEL = "BAAI/bge-reranker-base"  # cross-encoder for reranking

# --- Retrieval settings ---
# How many candidates each retriever pulls before reranking:
RETRIEVE_K = 12
# How many chunks survive reranking and go to the LLM:
TOP_K = 4
# Weights for the hybrid ensemble [BM25, dense]. Must sum to 1.0.
HYBRID_WEIGHTS = [0.4, 0.6]

# --- Safety ---
# If the best reranked score is below this, we treat the question as out-of-scope
# and refuse instead of hallucinating. Tune this with real questions.
MIN_RELEVANCE_SCORE = 0.15

# --- Chunking ---
# Fallback chunk size for text that isn't cleanly structured by Article.
FALLBACK_CHUNK_SIZE = 900
FALLBACK_CHUNK_OVERLAP = 150

# The name of the collection inside ChromaDB
COLLECTION_NAME = "compliance_docs"

# --- Ollama connection ---
# Defaults to localhost for normal (non-Docker) runs. In Docker, override with
# OLLAMA_HOST=http://host.docker.internal:11434 so the container reaches the
# Ollama server running on the host machine.
import os
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
