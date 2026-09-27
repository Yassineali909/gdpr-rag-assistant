"""
One command to build the whole search index from the source document.

    python build_index.py

Run this once after setup, and again whenever you change the source file
(config.SOURCE_FILE) or the chunking logic.
"""
from src.ingest import build_documents
from src.embed_store import build_vector_store


def main():
    print("=" * 60)
    print("Building the compliance search index")
    print("=" * 60)
    documents = build_documents()
    build_vector_store(documents)
    print("\nDone. You can now ask questions:")
    print('  python -m src.rag "What are the rules for erasing personal data?"')
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
