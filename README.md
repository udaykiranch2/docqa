# RAG Document Q&A System

A local Retrieval-Augmented Generation (RAG) system that lets you ask questions about your documents using open-source LLMs via the Hugging Face Inference API, with ChromaDB as the vector store.

## Features

- **Ask questions over your own documents** -- PDFs, text files, CSVs, Word docs, and Python source files
- **Flexible path input** -- ingest a single file, an entire directory, or use the default `./documents` folder
- **Interactive Q&A** -- conversational interface with source attribution
- **Fully local embeddings** -- runs `sentence-transformers` on CPU, no GPU required
- **Persistent vector store** -- ChromaDB saves indexed data so you only ingest once
- **Configurable** -- chunk size, retrieval count, similarity threshold, and model selection via `.env`

## Architecture

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
│ load_from_path()     │   │  get_llm()      -- HF endpoint setup  │
│   └─ _load_file()    │   │  build_qa_chain()-- LCEL RAG chain    │
│   └─ _load_directory │   │  ask_question() -- query + sources    │
│ chunk_documents()    │   │                                       │
│                      │   │  Uses: LangChain, HuggingFaceEndpoint │
│ Loaders: PyPDF, TXT, │   └───────────────┬───────────────────────┘
│   CSV, Docx, Python  │                   │
└──────────┬───────────┘                   │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      embedding_store.py                          │
│                                                                  │
│  get_embedding_model()  -- HuggingFaceEmbeddings (CPU)           │
│  get_vector_store()     -- ChromaDB instance (persistent)        │
│  index_documents()      -- add chunks to vector store            │
│  retrieve_documents()   -- similarity search with score filter   │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│      config.py       │
│  All settings from   │
│  .env / defaults     │
└──────────────────────┘
```

### Data Flow

```
1. Ingestion
   File/Dir path ──► load_from_path() ──► LangChain Loaders
        ──► Raw Documents ──► RecursiveCharacterTextSplitter
        ──► Chunks ──► HuggingFaceEmbeddings (encode)
        ──► ChromaDB (store vectors + metadata)

2. Query
   User question ──► ChromaDB similarity search ──► Top-K chunks
        ──► Prompt (system context + question)
        ──► HuggingFace Inference API (LLM)
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
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── .env                    # Your secrets (not committed)
├── SETUP.md                # Step-by-step setup instructions
├── documents/              # Default folder for documents (auto-created)
└── chroma_db/              # Persistent ChromaDB data (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Hugging Face token
cp .env.example .env
# Edit .env and add: HF_TOKEN=hf_xxxxx

# 3. Run
python main.py
# Paste a file/dir path at the prompt, or press Enter for ./documents
```

See [SETUP.md](SETUP.md) for the full setup guide.

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face API token |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `HF_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | LLM for answer generation |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage location |
| `CHROMA_COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap characters between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MIN_SIMILARITY_SCORE` | `0.3` | Minimum relevance score (0-1) |
| `DOCUMENTS_DIR` | `./documents` | Default document folder |

## Usage Examples

### Ingest a single file and ask questions

```bash
python main.py --ingest C:\docs\annual-report.pdf
# Then run Q&A:
python main.py
```

### Ingest an entire directory

```bash
python main.py --ingest "C:\My Documents\project-files"
```

### Interactive path prompt

```bash
python main.py
# At the Path: prompt, paste any file or directory
```

### Re-ingest after adding new files

```bash
python main.py --ingest
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RAG orchestration | LangChain |
| Embeddings | sentence-transformers (local, CPU) |
| Vector store | ChromaDB (local, persistent) |
| LLM | Hugging Face Inference API |
| Document loaders | PyPDF, python-docx, langchain-community |
| CLI | Rich (terminal formatting) |
| Language | Python 3.10+ |

## License

This project is provided as-is for educational and internal use.
