# Setup Guide: RAG Document Q&A System

## Prerequisites

- Python 3.10+
- pip
- A Hugging Face account with an API token

## Step 1: Get Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Sign up or log in
3. Click **New token**
4. Give it a name (e.g., `rag-project`)
5. Select **Read** access (for inference) or **Write** if you plan to push models
6. Copy the token

## Step 2: Configure Environment

Copy the example env file and add your token:

```bash
cp .env.example .env
```

Edit `.env` and paste your token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Optional overrides (defaults are already set):

```
# Models
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# ChromaDB storage
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=documents

# Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval
TOP_K_RESULTS=5
MIN_SIMILARITY_SCORE=0.3

# Default documents folder (used when no path is specified)
DOCUMENTS_DIR=./documents
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

| Package | Purpose |
|---------|---------|
| `langchain` | RAG orchestration |
| `sentence-transformers` | Local embeddings (downloads model on first run, ~90MB) |
| `chromadb` | Local vector database |
| `pypdf` | PDF document loader |
| `python-docx` | Word document loader |
| `huggingface-hub` | Hugging Face API access |
| `rich` | Terminal formatting |

## Step 4: Run the Application

### Interactive mode (recommended)

```bash
python main.py
```

On launch you will see a path prompt:

```
Enter a file or directory path to ingest, or press Enter to use default './documents'.
Path:
```

- **Paste a file path** to ingest a single file (e.g., `C:\docs\report.pdf`)
- **Paste a directory path** to ingest all supported files in that folder
- **Press Enter** to use the existing index (if available) or the default `./documents` folder

### Ingest-only mode (non-interactive)

Ingest from a specific path without starting Q&A:

```bash
python main.py --ingest C:\docs\report.pdf
python main.py --ingest C:\docs\my_folder
```

Ingest from the default `./documents` folder:

```bash
python main.py --ingest
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

Type `quit`, `exit`, or `q` to stop.

## Choosing a Different LLM

If the default model is unavailable or too slow, edit `.env`:

```
HF_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

Browse available text-generation models at:
[https://huggingface.co/models?pipeline_tag=text-generation&sort=trending](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)

Make sure the model has **Inference API** enabled (check the model page).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `HF_TOKEN not set` | Create `.env` with your token (Step 2) |
| `Model is loading` | Wait 20-30 seconds and retry - Hugging Face cold-start |
| `401 Unauthorized` | Check your token is correct and active |
| `No documents found` | Provide a valid file or directory path, or add files to `./documents` |
| `Unsupported file type` | Only `.pdf`, `.txt`, `.csv`, `.docx`, `.py` are supported |
| `Path does not exist` | Check the path for typos; use absolute paths if unsure |
| Slow first run | Embedding model downloads on first use (~90MB) |
