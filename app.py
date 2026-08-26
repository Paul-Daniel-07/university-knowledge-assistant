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
# Styling — academic design system: navy + parchment + brass, serif headings
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600&display=swap');

    :root {
        --navy: #14213D;
        --navy-light: #1E2F52;
        --parchment: #FAF7F2;
        --card: #FFFFFF;
        --brass: #B08A3E;
        --brass-dark: #8C6D2F;
        --ink: #1F2430;
        --ink-muted: #64605A;
        --hairline: #E4DFD3;
    }

    .stApp { background-color: var(--parchment); }
    .main { background-color: var(--parchment); }

    h1, h2, h3 { font-family: 'Lora', Georgia, serif !important; color: var(--navy); letter-spacing: -0.01em; }
    h1 { font-weight: 700 !important; border-bottom: 2px solid var(--brass); padding-bottom: 0.4rem; display: inline-block; }
    body, p, div, span, li { font-family: 'Inter', -apple-system, sans-serif; color: var(--ink); }
    .stCaption, [data-testid="stCaptionContainer"] { font-family: 'Inter', sans-serif !important; color: var(--ink-muted) !important; }

    /* Sidebar: deep navy, cream text */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
        border-right: 1px solid var(--navy-light);
    }
    section[data-testid="stSidebar"] * { color: #F2EFE8 !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: 'Lora', serif !important; color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] hr { border-color: var(--navy-light); }

    /* File uploader dropzone — match the navy sidebar instead of default white */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
    section[data-testid="stSidebar"] .stFileUploader section {
        background-color: var(--navy-light) !important;
        border: 1.5px dashed var(--brass) !important;
        border-radius: 6px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #F2EFE8 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--brass) !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* Chat bubbles */
    .stChatMessage {
        border-radius: 6px;
        border: 1px solid var(--hairline);
        background-color: var(--card);
    }

    /* Buttons: brass accent, not generic purple */
    .stButton > button, .stFormSubmitButton > button {
        background-color: var(--brass) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover { background-color: var(--brass-dark) !important; }

    /* Source citation pills — brass-on-parchment, not blue */
    .source-pill {
        display: inline-block;
        background: #F1E8D4;
        color: var(--brass-dark);
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin: 2px 6px 2px 0;
        border: 1px solid #DDCBA3;
    }
    .relevance-tag { color: var(--ink-muted); font-size: 0.75rem; font-family: 'Inter', sans-serif; }

    .kb-status-ok { color: #2F6B45; font-weight: 600; font-family: 'Inter', sans-serif; }
    .kb-status-missing { color: #A13B3B; font-weight: 600; font-family: 'Inter', sans-serif; }

    /* Chat input — distinctive parchment card with brass focus glow */
    [data-testid="stChatInput"] {
        border: 1.5px solid var(--hairline) !important;
        border-radius: 10px !important;
        background-color: #FFFDF8 !important;
        box-shadow: 0 1px 3px rgba(20, 33, 61, 0.06) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--brass) !important;
        box-shadow: 0 0 0 3px rgba(176, 138, 62, 0.18) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        color: var(--ink) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--ink-muted) !important;
        font-style: italic;
    }
    [data-testid="stChatInput"] button {
        background-color: var(--brass) !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] button:hover {
        background-color: var(--brass-dark) !important;
    }
    [data-testid="stChatInput"] button svg {
        fill: #FFFFFF !important;
    }
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