"""Embedding generation and ChromaDB vector store management."""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from typing import List, Optional

import config

# Module-level BM25 retriever instance, rebuilt whenever documents are indexed.
_bm25_retriever: Optional[BM25Retriever] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize the Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embedding_model: Optional[HuggingFaceEmbeddings] = None) -> Chroma:
    """Create or load the ChromaDB vector store."""
    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def clear_collection(vector_store: Optional[Chroma] = None) -> Chroma:
    """Delete all documents from the collection and reset BM25."""
    global _bm25_retriever

    if vector_store is None:
        vector_store = get_vector_store()

    vector_store.delete_collection()
    _bm25_retriever = None
    print("Cleared existing collection.")

    # Return a fresh vector_store (the old one's collection was deleted).
    return get_vector_store()


def index_documents(chunks: List, vector_store: Optional[Chroma] = None, clear_existing: bool = True) -> Chroma:
    """Add document chunks to the vector store and build/update the BM25 retriever."""
    global _bm25_retriever

    if vector_store is None:
        vector_store = get_vector_store()

    if clear_existing:
        vector_store = clear_collection(vector_store)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"Indexed {len(texts)} chunks into ChromaDB")

    # Build (or rebuild) the BM25 retriever from the current chunks.
    _bm25_retriever = BM25Retriever.from_documents(chunks, k=config.TOP_K_RESULTS)
    print(f"BM25 retriever built with {len(chunks)} chunks")

    return vector_store


def rebuild_bm25_from_store(vector_store: Optional[Chroma] = None) -> None:
    """Rebuild the BM25 retriever from documents already in the vector store."""
    global _bm25_retriever

    if vector_store is None:
        vector_store = get_vector_store()

    existing = vector_store.get(include=["documents", "metadatas"])
    if not existing or not existing["ids"]:
        print("No existing documents found in ChromaDB; BM25 retriever not built.")
        return

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(existing["documents"], existing["metadatas"])
    ]
    _bm25_retriever = BM25Retriever.from_documents(docs, k=config.TOP_K_RESULTS)
    print(f"BM25 retriever rebuilt from {len(docs)} existing ChromaDB chunks")


def get_bm25_retriever() -> Optional[BM25Retriever]:
    """Return the current BM25 retriever, or None if no documents have been indexed."""
    return _bm25_retriever


def get_collection_stats(vector_store: Optional[Chroma] = None) -> dict:
    """Return stats about the current collection (chunk count, source files)."""
    if vector_store is None:
        vector_store = get_vector_store()

    existing = vector_store.get(include=["metadatas"])
    ids = existing["ids"] if existing else []
    metadatas = existing["metadatas"] if existing else []

    source_files = set()
    for meta in metadatas:
        src = meta.get("source_file")
        if src:
            source_files.add(src)

    return {
        "chunk_count": len(ids),
        "source_files": sorted(source_files),
    }


def retrieve_documents(query: str, vector_store: Chroma, top_k: Optional[int] = None) -> List:
    """Retrieve relevant document chunks for a query."""
    if top_k is None:
        top_k = config.TOP_K_RESULTS

    results = vector_store.similarity_search_with_relevance_scores(
        query, k=top_k
    )

    # Filter by minimum similarity score
    filtered = [
        (doc, score)
        for doc, score in results
        if score >= config.MIN_SIMILARITY_SCORE
    ]

    if not filtered:
        print("No sufficiently relevant documents found.")
        return []

    return filtered
