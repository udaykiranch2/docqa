import os
from dotenv import load_dotenv

load_dotenv()

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "qwen3:8b")
# "ollama" = local Ollama (free), "huggingface" = HF Inference API (may incur costs)
RAG_LLM_MODE = os.getenv("RAG_LLM_MODE", "ollama")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# RAG Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.2"))

# Documents folder
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")

# RAGAS Evaluation Configuration
# "ollama" = local Ollama (free, no API needed)
# "api" = HF serverless inference (requires HF_TOKEN, may incur costs)
# "local" = run a model locally via transformers (requires GPU / slow on CPU)
RAGAS_EVAL_LLM_MODE = os.getenv("RAGAS_EVAL_LLM_MODE", "ollama")
RAGAS_EVAL_LLM_MODEL = os.getenv(
    "RAGAS_EVAL_LLM_MODEL", "qwen3:8b"
)
RAGAS_EVAL_LLM_PROVIDER = os.getenv(
    "RAGAS_EVAL_LLM_PROVIDER", "together-ai"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RAGAS_TEST_DATASET_PATH = os.getenv(
    "RAGAS_TEST_DATASET_PATH", "./ragas_test_dataset.json"
)
# RAGAS metric thresholds (0.00 - 1.00) — scores below these trigger FAIL
RAGAS_THRESHOLDS = {
    "faithfulness": float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.85")),
    "response_relevancy": float(os.getenv("RAGAS_RELEVANCY_THRESHOLD", "0.80")),
    "context_recall": float(os.getenv("RAGAS_CONTEXT_RECALL_THRESHOLD", "0.75")),
    "context_precision": float(os.getenv("RAGAS_CONTEXT_PRECISION_THRESHOLD", "0.80")),
}
