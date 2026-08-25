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
# Uses the Anthropic API. Set ANTHROPIC_API_KEY in a .env file (see .env.example).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-sonnet-5"
LLM_MAX_TOKENS 
 # low temperature: answers should stay grounded, not creative
