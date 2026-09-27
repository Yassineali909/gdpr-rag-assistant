# Phase 10: containerize the FastAPI backend.
#
# IMPORTANT — Ollama runs on the HOST, not in this container.
# The RAG pipeline reaches it via config.OLLAMA_HOST, which reads the OLLAMA_HOST
# env var (default set below). Because "host.docker.internal" does not resolve by
# default on Linux/WSL Docker, the run command must add a host-gateway mapping:
#
#   docker build -t compliance-rag .
#   docker run --rm -p 8000:8000 \
#       --add-host=host.docker.internal:host-gateway \
#       compliance-rag
#
# Then: curl http://localhost:8000/health   (should report ready)
#       curl -X POST http://localhost:8000/ask \
#            -H "Content-Type: application/json" \
#            -d '{"question":"What is the right to be forgotten?"}'
#
# (Ollama must be running on the host: `ollama serve` + the model pulled.)

FROM python:3.12-slim

WORKDIR /app

# System deps for some ML wheels (sentence-transformers, chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the EXACT working versions (frozen from the dev venv) so the container
# reproduces the tested stack instead of re-resolving loose ranges.
COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-lock.txt

COPY . .

# Build the search index at image-build time (downloads the embedding model,
# embeds the GDPR chunks, persists ChromaDB). Needs network during build.
RUN python build_index.py

# Point the pipeline at the host's Ollama. Overridable at run time.
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
