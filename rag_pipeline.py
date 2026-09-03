"""
rag_pipeline.py
----------------
Step 2 of the RAG pipeline: given a user question, retrieve the most
relevant chunks from ChromaDB and ask the LLM to answer using ONLY that
retrieved context (grounded generation), returning the answer plus the
sources used.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config
from google.genai.errors import ServerError, ClientError


SYSTEM_PROMPT = """You are the University Knowledge Assistant. You answer student
questions using ONLY the context passages provided below, which come from official
university documents (handbooks, regulations, exam guidelines, course material, etc.).

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain the answer, say clearly: "I couldn't find this in
  the available university documents." Do not guess or fabricate policy details.
- Be concise and direct. Use bullet points for lists (e.g. steps, requirements).
- When helpful, mention which document the information came from.
"""


class RAGPipeline:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        try:
            self.collection = client.get_collection(config.COLLECTION_NAME)
        except Exception:
            raise RuntimeError(
                "Knowledge base not found. Run `python ingest.py` first to build it."
            )
        self.llm_client = genai.Client(api_key=config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else None

    # -- Retrieval -----------------------------------------------------
    def retrieve(self, question: str, top_k: int = None):
        top_k = top_k or config.TOP_K
        query_embedding = self.embedding_model.encode([question]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        retrieved = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "relevance": round(1 - dist, 3),  # cosine distance -> similarity-ish score
            })
        return retrieved

          def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "I couldn't find this in the available university documents."

        context_block = "\n\n".join(
            f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
            for c in retrieved_chunks
        )

        if self.llm_client is None:
            return (
                "(LLM not configured — set GOOGLE_API_KEY in .env to enable grounded "
                "answers. Showing the most relevant retrieved passage instead.)\n\n"
                + retrieved_chunks[0]["text"]
            )

        gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )
        prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        try:
            response = self.llm_client.models.generate_content(
                model=config.LLM_MODEL, contents=prompt, config=gen_config,
            )
            return response.text
        except ServerError:
            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-3.1-flash-lite", contents=prompt, config=gen_config,
                )
                return response.text
            except ServerError:
                return (
                    "⚠️ Gemini is temporarily overloaded on Google's side. "
                    "Please wait a few seconds and try again.\n\n"
                    f"Meanwhile, here's the most relevant passage found:\n\n{retrieved_chunks[0]['text']}"
                )
        except ClientError as e:
            return f"⚠️ There was a problem calling the Gemini API: {e}"

    # -- Full pipeline ---------------------------------------------------
    def answer(self, question: str, top_k: int = None) -> dict:
        chunks = self.retrieve(question, top_k)
        answer_text = self.generate_answer(question, chunks)
        return {
            "question": question,
            "answer": answer_text,
            "sources": chunks,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in ("quit", "exit"):
            break
        result = pipeline.answer(q)
        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources used:")
        for s in result["sources"]:
            print(f"  - {s['source']} (chunk {s['chunk_index']}, relevance {s['relevance']})")