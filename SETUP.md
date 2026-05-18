# Setup Guide: RAG Document Q&A System

## Prerequisites

- Python 3.10+
- pip
- [Ollama](https://ollama.com) (for local LLM -- recommended)
- A Hugging Face account with an API token (only needed for HF Inference API mode)

## Step 1: Install Ollama (Recommended)

1. Download and install Ollama from [https://ollama.com](https://ollama.com)
2. Pull the default model:

```bash
ollama pull qwen3:8b
```

This downloads Qwen 3 8B (~5 GB). Other options: `llama3`, `mistral`, `qwen2`, `gemma3`.

Ollama runs as a background service on `http://localhost:11434` by default.

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `langchain` | RAG orchestration |
| `langchain-classic` | Ensemble retriever and contextual compression |
| `langchain-community` | Community loaders (BM25 retriever, FlashRank) |
| `langchain-huggingface` | Hugging Face integrations |
| `langchain-ollama` | Ollama integration |
| `sentence-transformers` | Local embeddings (downloads model on first run, ~90MB) |
| `chromadb` | Local vector database |
| `pymupdf` | PDF document loader |
| `python-docx` | Word document loader |
| `flashrank` | Reranking of retrieved results |
| `rank_bm25` | BM25 keyword retrieval (hybrid search) |
| `fastapi` + `uvicorn` | REST API server |
| `streamlit` | Web frontend |
| `ragas` | RAG pipeline evaluation |
| `rich` | Terminal formatting |

## Step 3: Configure Environment

Copy the example env file:

```bash
cp .env.example .env
```

For **Ollama mode** (default), no additional configuration is needed. The defaults work out of the box:

```
RAG_LLM_MODE=ollama
HF_LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

For **Hugging Face API mode**, edit `.env` and add your token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RAG_LLM_MODE=huggingface
HF_LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
```

### Get a Hugging Face Token (HF mode only)

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign up or log in
3. Click **New token** with **Read** access
4. Copy the token into `.env`

### All configuration options

```
# LLM mode: "ollama" (local, free) or "huggingface" (cloud API)
RAG_LLM_MODE=ollama
HF_LLM_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

# Hugging Face (only needed for huggingface mode)
HF_TOKEN=

# Embedding model (runs locally)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ChromaDB storage
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=documents

# Chunking
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# Retrieval
TOP_K_RESULTS=10
MIN_SIMILARITY_SCORE=0.2

# Default documents folder
DOCUMENTS_DIR=./documents

# RAGAS evaluation
RAGAS_EVAL_LLM_MODE=ollama
RAGAS_EVAL_LLM_MODEL=qwen3:8b
```

## Step 4: Run the Application

### Interactive CLI (recommended)

```bash
python main.py
```

On launch you will see a path prompt:

```
Enter a file or directory path to ingest, or press Enter to use default './documents'.
Path:
```

- **Paste a file path** to ingest a single file (e.g., `C:\docs\report.pdf`)
- **Paste a directory path** to ingest all supported files in that folder (subdirectories are included)
- **Press Enter** to use the existing index (if available) or the default `./documents` folder

### Ingest-only mode (non-interactive)

```bash
python main.py --ingest C:\docs\report.pdf
python main.py --ingest C:\docs\my_folder
python main.py --ingest
```

### Clear the index

```bash
python main.py --clear
```

### API server

```bash
python run_api.py
# FastAPI at http://localhost:8000
```

### Web UI

```bash
python run_ui.py
# Streamlit frontend
```

### RAGAS evaluation

```bash
python main.py --ragas
```

### Supported file formats

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Plain text | `.txt` |
| CSV | `.csv` |
| Word | `.docx` |
| Python source | `.py` |

## Step 5: Ask Questions

Once the Q&A prompt appears, type your question and press Enter:

```
Your question: What does the document say about project deadlines?
```

Type `quit`, `exit`, or `q` to stop. Press `Ctrl+C` during a query to interrupt it and return to the prompt.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Ollama connection refused` | Start Ollama (`ollama serve`) and verify it's running on port 11434 |
| `model "qwen3:8b" not found` | Run `ollama pull qwen3:8b` to download the model |
| `HF_TOKEN not set` | Only needed in `huggingface` mode. Set `RAG_LLM_MODE=ollama` to use local models instead |
| `Model is loading` | Wait 20-30 seconds and retry -- Hugging Face cold-start |
| `401 Unauthorized` | Check your HF token is correct and active |
| `No documents found` | Provide a valid file or directory path, or add files to `./documents` |
| `Unsupported file type` | Only `.pdf`, `.txt`, `.csv`, `.docx`, `.py` are supported |
| `Path does not exist` | Check the path for typos; use absolute paths if unsure |
| Slow first run | Embedding model downloads on first use (~90MB) |
