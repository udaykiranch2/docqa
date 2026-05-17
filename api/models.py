"""Pydantic request/response models for the RAG API."""

from typing import List, Optional

from pydantic import BaseModel


# --- Request Models ---

class QuestionRequest(BaseModel):
    question: str


# --- Response Models ---

class SourceDocument(BaseModel):
    source_file: str
    page: Optional[int] = None
    content_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunk_count: int = 0
    source_files: List[str] = []


class ClearResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    hf_token_configured: bool


class StatsResponse(BaseModel):
    chunk_count: int
    source_files: List[str]
