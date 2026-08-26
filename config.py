"""
Central configuration for the University Knowledge Assistant.
Edit these values to tune chunking, retrieval, and model behaviour.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "sample_docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "university_knowledge_base"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# --- Embeddings ----------------------------------------------------------
# Local, free, no API key required. Runs on CPU fine for this dataset size.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Retrieval -------------------------------------------------------------
TOP_K = 4                 # number of chunks retrieved per query

# --- LLM (generation) ------------------------------------------------------
# Uses the Google Gemini API — free tier, no credit card required.
# Get a key at https://aistudio.google.com/apikey
# Locally: set GOOGLE_API_KEY in a .env file (see .env.example).
# On Streamlit Community Cloud: set it under App settings -> Secrets instead
# (as GOOGLE_API_KEY = "AI...") — .env files aren't used there, so we fall
# back to st.secrets when the env var isn't set.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass  # not running under Streamlit, or no secrets configured — stays ""
LLM_MODEL = "gemini-2.5-flash"   # free-tier eligible
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.2      # low temperature: answers should stay grounded, not creative