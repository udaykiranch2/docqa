# RAG Document Q&A System

A local Retrieval-Augmented Generation (RAG) system that lets you ask questions about your documents using open-source LLMs. Runs Qwen 3 (8B) locally via Ollama by default (no API costs), with ChromaDB as the vector store.

Includes a CLI, a FastAPI backend, and a Streamlit frontend.

## Features

- **Ask questions over your own documents** -- PDFs, text files, CSVs, Word docs, and Python source files
- **Local LLM by default** -- uses Ollama with Qwen 3 8B (no cloud API or token required)
- **Hybrid retrieval** -- combines vector similarity search with BM25 keyword matching
- **Reranking** -- FlashRank reranks retrieved chunks so the most relevant context reaches the LLM
- **Fully local embeddings** -- runs `sentence-transformers` on CPU, no GPU required
- **Persistent vector store** -- ChromaDB saves indexed data so you only ingest once
- **Multiple interfaces** -- interactive CLI, REST API (FastAPI), and web UI (Streamlit)
- **RAG evaluation** -- built-in RAGAS health checks to measure pipeline quality
- **Configurable** -- chunk size, retrieval count, similarity threshold, and model selection via `.env`

## Quick Start

```bash
# 1. Install Ollama and pull a model
ollama pull qwen3:8b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

See [SETUP.md](SETUP.md) for the full setup guide, including HuggingFace API mode and optional configuration.

## Usage

### CLI (interactive)

```bash
# Interactive Q&A (prompts for document path)
python main.py

# Ingest a specific file or directory
python main.py --ingest path/to/documents

# Clear the vector store index
python main.py --clear

# Run RAGAS quality evaluation
python main.py --ragas
```

### API server

```bash
python run_api.py
# Starts FastAPI at http://localhost:8000
```

### Web UI

```bash
python run_ui.py
# Starts Streamlit frontend
```

### Supported file formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Plain text | `.txt` |
| CSV | `.csv` |
| Word | `.docx` |
| Python source | `.py` |

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_LLM_MODE` | `ollama` | `ollama` for local, `huggingface` for HF Inference API |
| `HF_LLM_MODEL` | `qwen3:8b` | Model name (Ollama model or HF model ID) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `HF_TOKEN` | *(empty)* | Hugging Face API token (only needed for HF mode) |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap characters between chunks |
| `TOP_K_RESULTS` | `10` | Number of chunks retrieved per query |
| `MIN_SIMILARITY_SCORE` | `0.2` | Minimum relevance score (0-1) |
| `DOCUMENTS_DIR` | `./documents` | Default document folder |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RAG orchestration | LangChain |
| LLM | Ollama (Qwen 3 8B) / Hugging Face Inference API |
| Embeddings | sentence-transformers (local, CPU) |
| Vector store | ChromaDB (local, persistent) |
| Keyword retrieval | BM25 (via rank_bm25) |
| Reranking | FlashRank |
| REST API | FastAPI |
| Web UI | Streamlit |
| Evaluation | RAGAS |
| CLI | Rich (terminal formatting) |

## RAGAS Evaluation Results

Pipeline evaluated with RAGAS metrics using Llama 3 (evaluator) and Qwen 3 8B (RAG LLM) locally. All metrics pass.

| Metric | Score | Threshold | Pass |
|--------|------:|----------:|------|
| Faithfulness | 0.8654 | 0.85 | PASS |
| Response Relevancy | 0.8869 | 0.80 | PASS |
| Context Recall | 0.9167 | 0.75 | PASS |
| Context Precision | 0.8587 | 0.80 | PASS |

**Overall Status: HEALTHY**

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for system diagrams and data flow.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test your changes -- run `python main.py --ragas` to verify pipeline quality
5. Submit a pull request with a clear description of the change

### Guidelines

- Keep changes focused -- one feature or fix per PR
- Follow the existing code style
- Update `.env.example` and `SETUP.md` if you add new configuration options
- Do not commit `.env`, `chroma_db/`, or `__pycache__/` directories

## License

This project is provided as-is for educational and internal use.
