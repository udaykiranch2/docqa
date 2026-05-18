"""RAGAS-based RAG health check evaluation.

Runs a small set of synthetic test questions through the RAG pipeline and
evaluates the quality of retrieved contexts and generated answers using
RAGAS metrics: Faithfulness, ResponseRelevancy, ContextPrecision, and
LLMContextRecall.

The health check produces per-metric scores and an overall pass/fail verdict
based on configurable thresholds.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import config
from rag_pipeline import ask_question

# ---------------------------------------------------------------------------
# Default test dataset
# ---------------------------------------------------------------------------

_DEFAULT_TEST_QUESTIONS: List[Dict[str, str]] = [
    {
        "user_input": "What is the main purpose of this document collection?",
        "reference": (
            "The documents provide information that can be queried "
            "through a question-answering system."
        ),
    },
    {
        "user_input": "Summarize the key topics covered in the documents.",
        "reference": (
            "The documents cover various topics that are indexed and "
            "searchable through the RAG pipeline."
        ),
    },
    {
        "user_input": "What are the main conclusions from the documents?",
        "reference": (
            "The conclusions depend on the specific documents that "
            "have been indexed in the system."
        ),
    },
]


def _load_test_dataset() -> List[Dict[str, str]]:
    """Load the test dataset from JSON file, falling back to defaults."""
    path = config.RAGAS_TEST_DATASET_PATH
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data
    return _DEFAULT_TEST_QUESTIONS


# ---------------------------------------------------------------------------
# LLM wrapper for RAGAS
# ---------------------------------------------------------------------------

def _get_ragas_llm():
    """Build a LLM for RAGAS evaluation.

    Uses ``llm_factory`` with an OpenAI-compatible client so that the
    ``instructor`` library enforces structured JSON output.  This prevents
    parse failures in RAGAS metrics like ResponseRelevancy and
    ContextPrecision that expect strict JSON responses.

    Modes controlled by ``RAGAS_EVAL_LLM_MODE``:

    * ``"ollama"`` (default) — uses a local Ollama model via its
      OpenAI-compatible ``/v1`` endpoint.  Free, no API token needed.
      Requires Ollama running locally (e.g. ``ollama run llama3``).
    * ``"api"`` — uses the HuggingFace Inference API.  Requires HF_TOKEN
      and may incur costs.
    * ``"local"`` — loads a model via the ``transformers`` pipeline.
      Free but slow on CPU, requires GPU for reasonable speed.
    """
    from ragas.llms import llm_factory

    if config.RAGAS_EVAL_LLM_MODE == "ollama":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )
        return llm_factory(
            model=config.RAGAS_EVAL_LLM_MODEL,
            client=client,
        )

    if config.RAGAS_EVAL_LLM_MODE == "api":
        if not config.HF_TOKEN:
            raise ValueError("HF_TOKEN not set — required for API mode.")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://api-inference.huggingface.co/v1",
            api_key=config.HF_TOKEN,
        )
        return llm_factory(
            model=config.RAGAS_EVAL_LLM_MODEL,
            client=client,
        )

    # --- Local mode (default, free) ---
    from ragas.llms import LangchainLLMWrapper
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    model_name = config.RAGAS_EVAL_LLM_MODEL
    print(f"Loading local evaluation model: {model_name} ...")

    import torch
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}" + (f" ({torch.cuda.get_device_name(0)})" if has_cuda else ""))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    """Build an embedding adapter for RAGAS from the existing HF embeddings."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from embedding_store import get_embedding_model

    model = get_embedding_model()
    return LangchainEmbeddingsWrapper(model)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def run_health_check(
    chain_and_retriever,
    test_dataset: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run a RAGAS health check against the current RAG pipeline.

    Parameters
    ----------
    chain_and_retriever : tuple
        The (chain, retriever) tuple returned by ``build_qa_chain``.
    test_dataset : list, optional
        Override the test questions. Each item must have ``user_input`` and
        ``reference`` keys. If *None*, the configured/default dataset is used.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``status``: ``"healthy"`` or ``"unhealthy"``
        - ``metrics``: per-metric scores (float 0–1, or ``None`` on failure)
        - ``details``: list of per-question results
        - ``message``: human-readable summary
    """
    if test_dataset is None:
        test_dataset = _load_test_dataset()

    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    # Import RAGAS metrics (stable API)
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextRecall,
        ContextPrecision,
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=2),
        LLMContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    # Run the RAG pipeline on each test question to gather responses + contexts.
    eval_rows: List[Dict[str, Any]] = []
    for item in test_dataset:
        question = item["user_input"]
        reference = item.get("reference", "")

        try:
            result = ask_question(question, chain_and_retriever)
            answer = result["answer"]
            contexts = [
                doc.page_content
                for doc in result.get("source_documents", [])
            ]
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            contexts = []

        eval_rows.append({
            "user_input": question,
            "response": answer,
            "reference": reference,
            "retrieved_contexts": contexts,
        })

    # Build the EvaluationDataset.
    from ragas import EvaluationDataset

    evaluation_dataset = EvaluationDataset.from_list(eval_rows)

    # Run evaluation.
    from ragas import evaluate, RunConfig
    from ragas.dataset_schema import EvaluationResult

    # Local models are slow — give each row plenty of time.
    # ContextPrecision iterates over all retrieved contexts sequentially
    # inside a single per-row timeout (5 contexts × ~40s = 200s+).
    run_config = RunConfig(timeout=900, max_retries=3, max_wait=45)

    raw_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        run_config=run_config,
        show_progress=False,
        raise_exceptions=False,
    )

    if not isinstance(raw_result, EvaluationResult):
        raise RuntimeError(
            f"Unexpected result type from ragas.evaluate(): {type(raw_result).__name__}"
        )

    eval_result: EvaluationResult = raw_result

    # Guard: if all per-row score dicts are empty the evaluation produced
    # no usable data (e.g. all LLM calls hit rate-limits / 402 errors).
    if not eval_result.scores or not any(eval_result.scores):
        raise RuntimeError(
            "RAGAS evaluation returned no scores. This usually means the "
            "LLM API calls failed (check HF token credits / rate limits)."
        )

    # Extract scores from the EvaluationResult.
    # eval_result.scores is a List[Dict[str, Any]] — one dict per row.
    # Average each metric column across all rows.
    # Map from the keys used in config thresholds to the actual metric
    # names used by RAGAS (ResponseRelevancy.name == "answer_relevancy").
    metric_key_map = {
        "faithfulness": "faithfulness",
        "response_relevancy": "answer_relevancy",
        "context_recall": "context_recall",
        "context_precision": "context_precision",
    }

    metric_scores: Dict[str, Optional[float]] = {}
    for config_key, ragas_key in metric_key_map.items():
        values = [row.get(ragas_key) for row in eval_result.scores if ragas_key in row]
        numeric = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        if numeric:
            metric_scores[config_key] = sum(numeric) / len(numeric)
        else:
            metric_scores[config_key] = None

    # Determine overall health using all four metric thresholds.
    # A None score (metric could not be computed) is treated as FAIL.
    status = "healthy"
    for name, threshold in config.RAGAS_THRESHOLDS.items():
        score = metric_scores.get(name)
        if score is None or score < threshold:
            status = "unhealthy"
            break

    # Build per-question details.
    details = []
    for row in eval_rows:
        details.append({
            "question": row["user_input"],
            "answer": row["response"][:200],
            "num_contexts": len(row["retrieved_contexts"]),
        })

    message = (
        f"RAGAS health check: status={status}, "
        f"faithfulness={metric_scores.get('faithfulness')}, "
        f"response_relevancy={metric_scores.get('response_relevancy')}, "
        f"context_recall={metric_scores.get('context_recall')}, "
        f"context_precision={metric_scores.get('context_precision')}"
    )

    return {
        "status": status,
        "metrics": metric_scores,
        "details": details,
        "message": message,
    }
