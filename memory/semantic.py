"""
Semantic Memory
===============
Stores knowledge as dense vector embeddings using ChromaDB (local, no server needed).
Enables fuzzy / similarity-based recall: "what do I know about X?"

Characteristics:
  - Persistent on disk (./chroma_db/ by default)
  - Uses SentenceTransformers for local embeddings (no API key needed)
  - Returns the most semantically similar past knowledge for a query
  - Best for: long-form facts, Q&A pairs, documents, summaries

How it works:
    1. Text  → embedding model → 384-dimensional vector
    2. Vector stored in ChromaDB with metadata
    3. Query → embed → cosine similarity search → top-k results

Typical usage:
    mem = SemanticMemory()
    mem.store("The user prefers dark mode and uses VS Code.")
    results = mem.search("editor preferences")
    # → ["The user prefers dark mode and uses VS Code."]
"""

import os
import time
import uuid
from typing import Dict, List, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class SemanticMemory:
    """
    Vector-based semantic memory backed by ChromaDB.

    Falls back to a simple list-based in-memory store if ChromaDB
    or sentence-transformers are not installed, so the agent still
    works – just without real semantic search.
    """

    def __init__(
        self,
        collection_name: str = "agent_knowledge",
        persist_dir: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self._fallback_store: List[str] = []

        # Native Torch/SentenceTransformers stacks can be fragile across local
        # Python builds. Keep the agent usable by default and require an
        # explicit opt-in for semantic embeddings.
        if os.environ.get("ENABLE_SEMANTIC_MEMORY") != "1":
            print("⚠️  Semantic embeddings disabled. Using fallback store.")
            print("   Set ENABLE_SEMANTIC_MEMORY=1 to try the ChromaDB backend.\n")
            self._use_chroma = False
            return

        if not CHROMA_AVAILABLE:
            print("⚠️  ChromaDB not found. Semantic memory using simple fallback store.")
            print("   Install with: pip install chromadb sentence-transformers\n")
            self._use_chroma = False
            return

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            print(f"✅ ChromaDB semantic memory ready  |  Collection: {collection_name}")
        except Exception as e:
            print(f"⚠️  ChromaDB init failed ({e}). Using fallback store.")
            self._use_chroma = False

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Embed and store a piece of knowledge.

        Args:
            text:     The text to remember.
            metadata: Optional dict with extra context (e.g. session, timestamp).

        Returns:
            Confirmation string.
        """
        if not self._use_chroma:
            self._fallback_store.append(text)
            return f"Stored (fallback): {text[:60]}..."

        doc_id = str(uuid.uuid4())
        meta = {"timestamp": time.time(), **(metadata or {})}
        self._collection.add(documents=[text], metadatas=[meta], ids=[doc_id])
        return f"Stored in semantic memory: {text[:60]}..."

    def search(self, query: str, n_results: int = 3) -> List[str]:
        """
        Return the top-k most semantically similar stored texts.

        Args:
            query:     Natural language query.
            n_results: Number of results to return.

        Returns:
            List of matching text strings (most relevant first).
        """
        if not self._use_chroma:
            # Naïve keyword fallback
            query_words = set(query.lower().split())
            scored = [
                (len(query_words & set(doc.lower().split())), doc)
                for doc in self._fallback_store
            ]
            scored.sort(reverse=True)
            return [doc for _, doc in scored[:n_results] if _ > 0]

        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        """Return total number of stored entries."""
        if not self._use_chroma:
            return len(self._fallback_store)
        return self._collection.count()

    def clear(self):
        """Delete all entries from the collection."""
        if self._use_chroma:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._ef,
            )
        else:
            self._fallback_store.clear()

    def __repr__(self) -> str:
        backend = "ChromaDB" if self._use_chroma else "Fallback"
        return f"SemanticMemory(backend={backend}, entries={self.count()})"
