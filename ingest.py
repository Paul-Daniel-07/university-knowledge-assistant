"""
ingest.py
---------
Step 1 of the RAG pipeline: turn raw university documents (PDF / TXT)
into a searchable vector knowledge base.

Pipeline:
  1. Collect documents from DOCS_DIR
  2. Extract raw text (pypdf for PDFs, plain read for .txt)
  3. Clean the text
  4. Split into overlapping chunks
  5. Embed each chunk with a local sentence-transformer model
  6. Store chunks + embeddings + metadata in a persistent ChromaDB collection

Run directly to (re)build the knowledge base:
    python ingest.py
"""
import os
import re
import glob
import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# 1-2. Load & extract text
# ---------------------------------------------------------------------------
def load_documents(docs_dir: str) -> list[dict]:
    """Return a list of {source, text} dicts for every PDF/TXT file found."""
    documents = []

    file_paths = glob.glob(os.path.join(docs_dir, "*.pdf")) + \
                 glob.glob(os.path.join(docs_dir, "*.txt"))

    if not file_paths:
        print(f"No documents found in {docs_dir}. Add PDFs or .txt files and re-run.")
        return documents

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            if path.lower().endswith(".pdf"):
                text = extract_text_from_pdf(path)
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            if text.strip():
                documents.append({"source": filename, "text": text})
                print(f"  Loaded: {filename} ({len(text)} chars)")
            else:
                print(f"  Skipped (empty): {filename}")
        except Exception as e:
            print(f"  Failed to read {filename}: {e}")

    return documents


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


# ---------------------------------------------------------------------------
# 3. Clean
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)          # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excess blank lines
    return text.strip()


# ---------------------------------------------------------------------------
# 4. Chunk
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character chunker that tries to break on
    sentence/paragraph boundaries where possible."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end the chunk at a sentence boundary near `end`
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap  # step forward, keeping overlap for context continuity

    return chunks


# ---------------------------------------------------------------------------
# 5-6. Embed & store
# ---------------------------------------------------------------------------
def build_knowledge_base():
    print("Step 1/4: Loading documents...")
    documents = load_documents(config.DOCS_DIR)
    if not documents:
        return

    print("\nStep 2/4: Cleaning and chunking...")
    all_chunks, all_metadatas, all_ids = [], [], []
    for doc in documents:
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})
            all_ids.append(str(uuid.uuid4()))
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    print("\nStep 3/4: Generating embeddings (local model, first run downloads weights)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    print("\nStep 4/4: Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # Fresh build each run so re-ingesting doesn't duplicate chunks
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(f"\nDone. Knowledge base built at: {config.CHROMA_DIR}")
    print(f"Collection '{config.COLLECTION_NAME}' contains {collection.count()} chunks.")


if __name__ == "__main__":
    build_knowledge_base()
