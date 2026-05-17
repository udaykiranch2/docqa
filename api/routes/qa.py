"""Question & Answer endpoint."""

from fastapi import APIRouter

from api.models import QuestionRequest, AskResponse, SourceDocument
from api.state import get_rag_state
from rag_pipeline import ask_question

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])


@router.post("/ask", response_model=AskResponse)
def ask(request: QuestionRequest):
    """Ask a question and get an answer with source documents."""
    state = get_rag_state()
    chain_and_retriever = state.get_chain_and_retriever()
    result = ask_question(request.question, chain_and_retriever)

    sources = []
    seen = set()
    for doc in result["source_documents"]:
        src = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "")
        label = f"{src}" + (f" (page {page})" if page != "" else "")
        if label not in seen:
            seen.add(label)
            sources.append(
                SourceDocument(
                    source_file=src,
                    page=page if page else None,
                    content_preview=doc.page_content[:200],
                )
            )

    return AskResponse(answer=result["answer"], sources=sources)
