# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                            │
│  Entry point -- parses args, prompts for path, runs ingestion    │
│  or launches interactive Q&A                                     │
└──────────┬────────────────────────────┬──────────────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────┐   ┌───────────────────────────────────────┐
│ document_processor.py│   │         rag_pipeline.py               │
│                      │   │                                       │
│ load_from_path()     │   │  get_llm()      -- Ollama / HF setup  │  (Qwen 3 8B / HF)  │
│   └─ _load_file()    │   │  build_qa_chain()-- LCEL RAG chain    │
│   └─ _load_directory │   │  ask_question() -- query + sources    │
│ chunk_documents()    │   │                                       │
│                      │   │  Uses: LangChain, ChatOllama          │
│ Loaders: PyMuPDF,    │   └───────────────┬───────────────────────┘
│   CSV, Docx, Python  │                   │
└──────────┬───────────┘                   │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      embedding_store.py                          │
│                                                                  │
│  get_embedding_model()  -- HuggingFaceEmbeddings (CPU)           │
│  get_vector_store()     -- ChromaDB instance (persistent)        │
│  index_documents()      -- add chunks to vector store + BM25     │
│  clear_collection()     -- wipe ChromaDB index + BM25            │
│  retrieve_documents()   -- similarity search with score filter   │
│  rebuild_bm25_from_store() -- rebuild BM25 from existing data    │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│      config.py       │
│  All settings from   │
│  .env / defaults     │
└──────────────────────┘
```

## Data Flow

```
1. Ingestion
   File/Dir path ──► load_from_path() ──► LangChain Loaders
        ──► Raw Documents ──► RecursiveCharacterTextSplitter
        ──► Chunks ──► HuggingFaceEmbeddings (encode)
        ──► ChromaDB (store vectors + metadata) + BM25 index

2. Query
   User question ──► EnsembleRetriever (vector + BM25)
        ──► FlashRank reranking ──► Top-N chunks
        ──► Prompt (system context + question)
        ──► Ollama (local LLM) / HuggingFace Inference API
        ──► Answer + source attribution
```

## Project Structure

```
summarize_docs/
├── main.py                 # CLI entry point -- interactive & ingest modes
├── config.py               # Configuration loaded from .env
├── document_processor.py   # Document loading, path resolution, chunking
├── embedding_store.py      # Embedding model + ChromaDB vector store
├── rag_pipeline.py         # LLM setup, RAG chain, Q&A logic
├── ragas_evaluation.py     # RAGAS metric evaluation logic
├── run_ragas.py            # Standalone RAGAS runner
├── run_api.py              # FastAPI server launcher
├── run_ui.py               # Streamlit UI launcher
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .env                    # Your secrets (not committed)
├── SETUP.md                # Step-by-step setup instructions
├── api/
│   ├── app.py              # FastAPI application factory
│   ├── models.py           # Pydantic request/response models
│   ├── state.py            # Shared app state (QA chain, vector store)
│   └── routes/
│       ├── health.py       # Health check endpoint
│       ├── qa.py           # Q&A endpoint
│       ├── documents.py    # Document ingestion endpoint
│       └── ragas_health.py # RAGAS evaluation endpoint
├── ui/
│   ├── app.py              # Streamlit frontend application
│   └── api_client.py       # HTTP client for the FastAPI backend
├── documents/              # Default folder for documents (auto-created)
└── chroma_db/              # Persistent ChromaDB data (auto-created)
```
