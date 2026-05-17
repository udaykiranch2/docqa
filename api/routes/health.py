"""Health and status endpoints."""

from fastapi import APIRouter

import config
from api.models import HealthResponse, StatsResponse
from api.state import get_rag_state
from embedding_store import get_collection_stats

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    state = get_rag_state()
    stats = get_collection_stats(state.vector_store)
    return HealthResponse(
        status="ok",
        index_ready=stats["chunk_count"] > 0,
        hf_token_configured=bool(config.HF_TOKEN),
    )


@router.get("/documents/stats", response_model=StatsResponse)
def document_stats():
    state = get_rag_state()
    stats = get_collection_stats(state.vector_store)
    return StatsResponse(**stats)
