"""RAGAS-based RAG health check endpoint."""

import logging
import traceback

from fastapi import APIRouter, HTTPException

from api.models import RagasHealthResponse
from api.state import get_rag_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/health", tags=["ragas"])


@router.post("/ragas", response_model=RagasHealthResponse)
def ragas_health_check():
    """Run a RAGAS evaluation health check against the current RAG pipeline.

    This endpoint runs a set of test questions through the RAG pipeline and
    evaluates the quality using RAGAS metrics (Faithfulness, Response
    Relevancy, Context Recall, Context Precision). Returns per-metric scores
    and an overall healthy/unhealthy verdict.

    **Note**: This is a long-running endpoint — it makes multiple LLM calls
    for both answer generation and metric computation.
    """
    state = get_rag_state()
    chain_and_retriever = state.get_chain_and_retriever()

    if chain_and_retriever[0] is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain is not initialised. Upload documents first.",
        )

    try:
        from ragas_evaluation import run_health_check

        result = run_health_check(chain_and_retriever)
    except Exception as exc:
        logger.exception("RAGAS health check failed")
        raise HTTPException(
            status_code=500,
            detail=f"RAGAS evaluation failed: {exc}",
        ) from exc

    return RagasHealthResponse(**result)
