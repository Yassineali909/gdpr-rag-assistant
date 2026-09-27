"""
Phase 3: Turn chunks into vectors and store them in ChromaDB.

Embeddings are numeric fingerprints of meaning. Two texts about the same idea land
close together in vector space, which is how semantic search works. We store the
vectors (plus the original text and its article metadata) in a local ChromaDB.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once. Downloads on first run, then cached."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # good for cosine similarity
    )


def build_vector_store(documents: list[Document]) -> Chroma:
    """Embed all documents and persist them to disk in ChromaDB."""
    embeddings = get_embeddings()
    print(f"[embed] Embedding {len(documents)} chunks with {config.EMBEDDING_MODEL} ...")
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"[embed] Persisted vector store to {config.CHROMA_DIR}")
    return store


def load_vector_store() -> Chroma:
    """Reopen an already-built vector store (no re-embedding)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


if __name__ == "__main__":
    # python -m src.embed_store  -> rebuild the index from the source file
    from src.ingest import build_documents

    docs = build_documents()
    build_vector_store(docs)
