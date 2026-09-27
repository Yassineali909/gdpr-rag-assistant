"""
Phase 1-2: Load the regulation and split it into coherent chunks.

The key idea: regulations have STRUCTURE (Article 1, Article 2, ...). Instead of
blindly cutting the text every N characters, we split on Article boundaries so each
chunk is a self-contained legal unit, and we attach the Article number as metadata.
That metadata is what powers citations later — so we never throw it away.
"""
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


# Matches lines like "Article 17", "Article 6a", etc. at the start of a line.
ARTICLE_PATTERN = re.compile(r"(?im)^\s*Article\s+(\d+[a-z]?)\s*$")


def load_text(source_file: Path = config.SOURCE_FILE) -> str:
    """Read the raw regulation text from disk."""
    if not source_file.exists():
        raise FileNotFoundError(
            f"Source file not found: {source_file}\n"
            f"Run `python scripts/download_gdpr.py` or check config.SOURCE_FILE."
        )
    return source_file.read_text(encoding="utf-8")


def structure_aware_chunks(text: str) -> list[Document]:
    """
    Split the text on 'Article N' headers. Each Article becomes one (or more) chunks,
    tagged with its article number so answers can cite it.
    Falls back to plain recursive splitting for anything before the first Article.
    """
    matches = list(ARTICLE_PATTERN.finditer(text))

    # If the document has no recognizable Article structure, fall back entirely.
    if not matches:
        return _fallback_split(text, article="N/A")

    docs: list[Document] = []

    # Everything before the first real Article header is recitals / Treaty
    # references, NOT articles. Indexing them caused recital text to be
    # mislabeled (e.g. "Article 16(46)") and pollute citations, so we exclude
    # them from the searchable index entirely to protect citation integrity.
    # (If you want them searchable later, tag them article="Recital" instead.)

    # Walk each Article span: from this header to the next header.
    for i, match in enumerate(matches):
        article_no = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()

        # Long articles get sub-split, but every piece keeps the same article tag.
        if len(article_text) > config.FALLBACK_CHUNK_SIZE * 1.5:
            docs.extend(_fallback_split(article_text, article=article_no))
        else:
            docs.append(
                Document(
                    page_content=article_text,
                    metadata={"article": article_no, "source": "GDPR"},
                )
            )

    return docs


def _fallback_split(text: str, article: str) -> list[Document]:
    """Plain recursive character splitting, used for unstructured spans."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.FALLBACK_CHUNK_SIZE,
        chunk_overlap=config.FALLBACK_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        Document(page_content=p, metadata={"article": article, "source": "GDPR"})
        for p in pieces
    ]


def build_documents(source_file: Path = config.SOURCE_FILE) -> list[Document]:
    """Full ingest pipeline: load -> chunk. Returns LangChain Documents."""
    text = load_text(source_file)
    docs = structure_aware_chunks(text)
    print(f"[ingest] Loaded {source_file.name} -> {len(docs)} chunks")
    return docs


if __name__ == "__main__":
    # Quick manual test: python -m src.ingest
    documents = build_documents()
    for d in documents[:3]:
        print("-" * 60)
        print(f"Article {d.metadata['article']}:")
        print(d.page_content[:200], "...")
