"""
Phase 10: Streamlit web UI.

A simple chat-style interface where a user asks a compliance question and gets a
grounded answer WITH the source Articles shown alongside — the citation feature is
the star of the show, so we make it visible.

Run:  streamlit run app.py
"""
import streamlit as st

from src.rag import ComplianceRAG

st.set_page_config(page_title="Compliance RAG Assistant", page_icon="⚖️")


@st.cache_resource
def get_rag():
    """Load the RAG system once and cache it across reruns."""
    return ComplianceRAG()


st.title("⚖️ Compliance RAG Assistant")
st.caption(
    "Ask a question about the GDPR. Answers are grounded in the regulation and cite "
    "their sources. If the answer isn't in the text, the assistant will say so."
)

rag = get_rag()

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the rules for erasing personal data?",
)

if st.button("Ask") and question:
    with st.spinner("Searching the regulation..."):
        result = rag.answer(question)

    if result.refused:
        st.warning(result.answer)
    else:
        st.markdown("### Answer")
        st.write(result.answer)

        st.markdown("### Sources")
        for article, score, text in result.sources:
            with st.expander(f"Article {article}  ·  relevance {score:.2f}"):
                st.write(text)

with st.sidebar:
    st.header("About")
    st.write(
        "This is a production-style RAG system: structure-aware chunking, hybrid "
        "retrieval (BM25 + dense vectors), cross-encoder reranking, source citations, "
        "and out-of-scope refusal."
    )
    st.write("Runs fully locally with Ollama — no API keys, no cloud.")
