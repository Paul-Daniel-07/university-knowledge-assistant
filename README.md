# University Knowledge Assistant

A context-aware university knowledge assistant built using **Retrieval-Augmented
Generation (RAG)**. Upload university documents (handbooks, exam guidelines, course
material, policies), and ask questions in natural language — the system retrieves the
exact relevant passages and generates a grounded answer with source citations, instead
of relying on the LLM's general knowledge alone.

Built as a Generative AI capstone / internship project.

---

## Why RAG?

Large Language Models are powerful but don't know your university's specific rules —
and if asked directly, they will confidently *guess* wrong answers ("hallucinate").
RAG solves this by:

1. Storing your actual documents as searchable vector embeddings
2. Retrieving the most relevant passages for each question
3. Instructing the LLM to answer **only** from those retrieved passages

This is the same architecture pattern used in real enterprise products: internal
support bots, documentation assistants, and customer-facing help chatbots.

---

## Architecture

```
                    ┌─────────────────────┐
   Upload PDFs/TXT  │   1. INGESTION       │
   ───────────────► │   (ingest.py)        │
                    │                       │
                    │  Extract text (pypdf) │
                    │  Clean text           │
                    │  Chunk (800 chars,    │
                    │    150 char overlap)  │
                    │  Embed (MiniLM-L6-v2) │
                    │  Store in ChromaDB    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
   User question     │   2. RETRIEVAL +     │
   ───────────────► │      GENERATION       │
                    │   (rag_pipeline.py)  │
                    │                       │
                    │  Embed the question   │
                    │  Vector search top-K  │
                    │  chunks in ChromaDB   │
                    │  Send chunks + Q to   │
                    │  Claude (grounded     │
                    │  system prompt)       │
                    └──────────┬───────────┘
                               │
                               ▼
                    Answer + cited sources
                    (app.py — Streamlit UI)
```

---

## Tech Stack

| Component        | Choice                          | Why |
|-------------------|----------------------------------|-----|
| Text extraction   | `pypdf`                         | Reliable, lightweight PDF text extraction |
| Chunking          | Custom sliding-window splitter   | Sentence-boundary aware, configurable overlap |
| Embeddings        | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally, no API cost for indexing |
| Vector database   | `ChromaDB` (persistent, local)  | Zero-setup, file-based, production-capable |
| LLM (generation)  | Anthropic Claude (`claude-sonnet-5`) | Strong grounded-generation and instruction following |
| UI                | Streamlit                        | Fast to build, chat-native components |

---

## Project Structure

```
university-knowledge-assistant/
├── app.py              # Streamlit chat UI
├── ingest.py            # Document ingestion pipeline (Steps 1-6 of "How to Do")
├── rag_pipeline.py       # Retrieval + grounded generation
├── config.py             # All tunable settings in one place
├── requirements.txt
├── .env.example           # Copy to .env and add your API key
├── sample_docs/            # Put your PDFs/TXT here (a sample handbook is included)
└── chroma_db/               # Auto-created persistent vector store
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Anthropic API key
```bash
cp .env.example .env
# then edit .env and paste your key from https://console.anthropic.com/
```
> The app still runs without a key — it will show you the most relevant retrieved
> passage directly instead of an LLM-generated answer. This is useful for testing the
> retrieval half of the pipeline independently.

### 3. Add your documents
Drop PDF or `.txt` files into `sample_docs/` (a sample handbook is already included so
you can try the app immediately), or upload them directly from the sidebar once the app
is running.

### 4. Build the knowledge base
```bash
python ingest.py
```
Or click **"Build / Rebuild Knowledge Base"** in the app sidebar.

### 5. Run the app
```bash
streamlit run app.py
```

---

## Example Questions (using the included sample handbook)

- "What is the minimum attendance required to sit for exams?"
- "What happens if I'm caught using unfair means in an exam?"
- "How many books can I borrow from the library?"
- "What CGPA do I need to keep my scholarship?"
- "Is generative AI allowed for assignments?"

---

## How It Maps to the "How to Do" Steps

| Step in project brief | Where implemented |
|---|---|
| Collect university PDFs/documents | `sample_docs/` + Streamlit uploader |
| Extract text from documents | `ingest.extract_text_from_pdf` |
| Clean unnecessary characters/formatting | `ingest.clean_text` |
| Divide documents into smaller chunks | `ingest.chunk_text` |
| Generate embeddings | `ingest.build_knowledge_base` (SentenceTransformer) |
| Store embeddings in vector DB | `ingest.build_knowledge_base` (ChromaDB) |
| Question-processing pipeline | `rag_pipeline.RAGPipeline` |
| Convert question into embedding | `RAGPipeline.retrieve` |
| Retrieve most relevant chunks | `RAGPipeline.retrieve` |
| Pass context to LLM | `RAGPipeline.generate_answer` |
| Generate grounded response | `RAGPipeline.generate_answer` (system prompt enforces grounding) |
| Display answer with source references | `app.py` (expandable "Sources used" panel) |
| Build chatbot UI | `app.py` |
| Test with predefined questions | See "Example Questions" above |
| Measure retrieval/answer quality | `relevance` score shown per source; see Evaluation below |

---

## Evaluation Notes (for your project report)

- **Retrieval quality**: each retrieved chunk shows a relevance score (1 − cosine
  distance). Spot-check that top results are actually on-topic for a range of test
  questions.
- **Answer grounding**: the system prompt explicitly forbids using outside knowledge
  and requires the model to say "I couldn't find this in the available university
  documents" when the answer isn't in the retrieved context — verify this by asking an
  out-of-scope question.
- **Chunking trade-off**: `CHUNK_SIZE`/`CHUNK_OVERLAP` in `config.py` control the
  precision/recall trade-off — smaller chunks are more precise but may lose surrounding
  context; larger chunks give more context but dilute relevance scoring. Worth
  documenting a few experiments with different values in your report.

---

## Possible Extensions (good "future work" section)

- Hybrid search (keyword + semantic) for better recall on exact terms (dates, codes)
- Re-ranking retrieved chunks with a cross-encoder before generation
- Multi-turn conversational memory (follow-up questions referencing earlier answers)
- Support for `.docx` and scanned/OCR'd PDFs
- Admin analytics: most-asked questions, unanswered-question log
