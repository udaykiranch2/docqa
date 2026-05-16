"""Embedding generation and ChromaDB vector store management."""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional

import config


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


def index_documents(chunks: List, vector_store: Optional[Chroma] = None) -> Chroma:
    """Add document chunks to the vector store."""
    if vector_store is None:
        vector_store = get_vector_store()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"Indexed {len(texts)} chunks into ChromaDB")

    return vector_store


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
