"""
app.py
------
Streamlit front-end for the University Knowledge Assistant.
Lets students upload university documents, build the knowledge base,
and ask questions conversationally with grounded, source-cited answers.
"""
import os
import time

import streamlit as st

import config
from ingest import build_knowledge_base

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #FAFAF9; }
    .stChatMessage { border-radius: 10px; }
    .source-pill {
        display: inline-block;
        background: #EEF2FF;
        color: #3730A3;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin: 2px 6px 2px 0;
        border: 1px solid #C7D2FE;
    }
    .relevance-tag {
        color: #6B7280;
        font-size: 0.75rem;
    }
    h1 { letter-spacing: -0.02em; }
    .kb-status-ok { color: #15803D; font-weight: 600; }
    .kb-status-missing { color: #B91C1C; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def kb_exists() -> bool:
    return os.path.isdir(config.CHROMA_DIR) and len(os.listdir(config.CHROMA_DIR)) > 0


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from rag_pipeline import RAGPipeline
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Sidebar: document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Knowledge Base")

    status = kb_exists()
    if status:
        st.markdown('<span class="kb-status-ok">● Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="kb-status-missing">● Not built yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Upload documents")
    uploaded_files = st.file_uploader(
        "Add university PDFs or .txt files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        os.makedirs(config.DOCS_DIR, exist_ok=True)
        for f in uploaded_files:
            save_path = os.path.join(config.DOCS_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to sample_docs/")

    existing_docs = []
    if os.path.isdir(config.DOCS_DIR):
        existing_docs = [f for f in os.listdir(config.DOCS_DIR) if f.endswith((".pdf", ".txt"))]

    if existing_docs:
        st.markdown(f"**{len(existing_docs)} document(s) staged:**")
        for d in existing_docs:
            st.caption(f"📄 {d}")

    st.markdown("---")
    if st.button("🔨 Build / Rebuild Knowledge Base", use_container_width=True, type="primary"):
        if not existing_docs:
            st.error("Add at least one document first.")
        else:
            with st.spinner("Extracting, chunking, embedding, and indexing documents..."):
                build_knowledge_base()
            st.cache_resource.clear()
            st.success("Knowledge base built successfully.")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    top_k = st.slider("Chunks retrieved per question", 2, 8, config.TOP_K)

    st.markdown("---")
    st.caption(
        "Built as a Generative AI capstone project: document ingestion → "
        "chunking → embeddings → vector search → LLM-grounded generation."
    )

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
st.title("University Knowledge Assistant")
st.caption("Ask questions about your university's handbooks, regulations, exam guidelines, or course material. Answers are grounded in your uploaded documents, with sources cited.")

if not kb_exists():
    st.info(
        "👈 Upload one or more university documents in the sidebar, then click "
        "**Build Knowledge Base** to get started."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources used"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<span class="source-pill">{s["source"]}</span>'
                        f'<span class="relevance-tag">relevance {s["relevance"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

question = st.chat_input("Ask about exam rules, attendance policy, course structure...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            pipeline = load_pipeline()
            result = pipeline.answer(question, top_k=top_k)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📚 Sources used"):
                for s in result["sources"]:
                    st.markdown(
                        f'<span class="source-pill">{s["source"]}</span>'
                        f'<span class="relevance-tag">relevance {s["relevance"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
