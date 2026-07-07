"""
Thin wrapper around ChromaDB. Embeddings are computed with a local
sentence-transformers model so semantic search works with zero API
cost and zero API key — only the *generation* step optionally needs
an LLM key.
"""
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedder


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name="documents")
        self.embedder = get_embedder()

    def add_chunks(self, doc_id: int, chunks: List[str]) -> int:
        ids = [f"doc{doc_id}_chunk{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )
        return len(chunks)

    def query(self, question: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or settings.TOP_K
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine distance -> similarity
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i],
                        "score": round(float(similarity), 4),
                    }
                )
        return chunks

    def count(self) -> int:
        return self.collection.count()


_vectorstore_singleton = None


def get_vectorstore() -> VectorStore:
    global _vectorstore_singleton
    if _vectorstore_singleton is None:
        _vectorstore_singleton = VectorStore()
    return _vectorstore_singleton
