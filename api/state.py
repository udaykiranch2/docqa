"""Singleton state holder for the RAG chain and related resources.

Manages the lifecycle of the QA chain, vector store, and embedding model
so they can be shared across FastAPI request handlers.
"""

import threading
from typing import Optional, Tuple

from langchain_core.runnables import Runnable
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from embedding_store import get_embedding_model, get_vector_store, rebuild_bm25_from_store
from rag_pipeline import build_qa_chain


class RAGState:
    """Holds initialized RAG components behind a lock for thread safety."""

    def __init__(self):
        self._lock = threading.Lock()
        self.chain: Optional[Runnable] = None
        self.retriever = None
        self.vector_store: Optional[Chroma] = None
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None

    def initialize(self) -> None:
        """Bootstrap embedding model, vector store, and QA chain."""
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store(self.embedding_model)
        self._rebuild_chain()

    def _rebuild_chain(self) -> None:
        """Build (or rebuild) the QA chain. Must be called with _lock held or during init."""
        self.chain, self.retriever = build_qa_chain(
            vector_store=self.vector_store
        )

    def rebuild_after_ingest(self) -> None:
        """Rebuild the chain after documents have been re-ingested."""
        with self._lock:
            self._rebuild_chain()

    def get_chain_and_retriever(self) -> Tuple:
        """Return the current (chain, retriever) tuple."""
        with self._lock:
            return self.chain, self.retriever


# Module-level singleton — created once at FastAPI startup.
rag_state: Optional[RAGState] = None


def get_rag_state() -> RAGState:
    """Return the global RAGState instance (raises if not initialized)."""
    if rag_state is None:
        raise RuntimeError("RAGState not initialized. Is the FastAPI app running?")
    return rag_state
