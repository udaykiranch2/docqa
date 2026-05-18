"""FastAPI application for the RAG Document Q&A system."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.state as state_module
from api.routes import health, documents, qa, ragas_health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG resources on startup, clean up on shutdown."""
    rag = state_module.RAGState()
    rag.initialize()
    state_module.rag_state = rag
    yield
    state_module.rag_state = None


app = FastAPI(
    title="RAG Document Q&A API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)
app.include_router(ragas_health.router)
