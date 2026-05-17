"""Document upload and management endpoints."""

import os
import shutil
import tempfile

from fastapi import APIRouter, UploadFile, File

from api.models import IngestResponse, ClearResponse
from api.state import get_rag_state
from document_processor import load_from_path, chunk_documents
from embedding_store import index_documents, clear_collection, get_collection_stats

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".py"}


@router.post("/upload", response_model=IngestResponse)
def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more documents, ingest them into ChromaDB."""
    state = get_rag_state()

    # Save uploaded files to a temporary directory.
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    saved_paths = []

    try:
        for upload in files:
            ext = os.path.splitext(upload.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            safe_name = os.path.basename(upload.filename or "unnamed")
            dest = os.path.join(tmp_dir, safe_name)
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(dest)

        if not saved_paths:
            return IngestResponse(
                success=False,
                message="No supported files found in upload.",
            )

        # Load, chunk, and index.
        documents = load_from_path(tmp_dir)
        if not documents:
            return IngestResponse(
                success=False,
                message="No text could be extracted from the uploaded files.",
            )

        chunks = chunk_documents(documents)
        index_documents(chunks, state.vector_store, clear_existing=False)

        # Rebuild the QA chain (BM25 state changed).
        state.rebuild_after_ingest()

        stats = get_collection_stats(state.vector_store)
        return IngestResponse(
            success=True,
            message=f"Ingested {len(chunks)} chunks from {len(saved_paths)} file(s).",
            chunk_count=stats["chunk_count"],
            source_files=stats["source_files"],
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.delete("/clear", response_model=ClearResponse)
def clear_index():
    """Clear all documents from the index."""
    state = get_rag_state()
    state.vector_store = clear_collection(state.vector_store)
    state.rebuild_after_ingest()
    return ClearResponse(success=True, message="Index cleared.")
