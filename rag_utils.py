#These are helper functions for RAG

from sentence_transformers import SentenceTransformer
import faiss
import os

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_index_docs(file_path="data/gradio_docs.md"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks (e.g., by heading or newlines)
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    # Embed each chunk
    embeddings = embedding_model.encode(chunks)

    # Index with FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return chunks, index, embeddings

def get_top_k_context(query, chunks, index, embeddings, k=2):
    query_vec = embedding_model.encode([query])
    scores, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]