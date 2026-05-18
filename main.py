"""Document Q&A using RAG with Hugging Face models."""

import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import config
from document_processor import chunk_documents, load_documents, load_from_path
from embedding_store import (
    clear_collection,
    get_embedding_model,
    get_vector_store,
    index_documents,
)
from rag_pipeline import ask_question, build_qa_chain

console = Console()


def ingest_documents(path: Optional[str] = None):
    """Load, chunk, and index documents from a path or the default directory."""
    console.print("\n[bold blue]--- Document Ingestion ---[/bold blue]")
    try:
        if path:
            documents = load_from_path(path)
        else:
            documents = load_documents()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return False

    if not documents:
        console.print(
            "[yellow]No documents found.[/yellow]\n"
            "Provide a path to a file or directory with supported files "
            "(PDF, TXT, CSV, DOCX, PY)."
        )
        return False

    chunks = chunk_documents(documents)
    if not chunks:
        console.print("[red]No chunks created from documents.[/red]")
        return False

    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model)
    index_documents(chunks, vector_store)
    console.print("[green]Documents indexed successfully![/green]")
    return True


def interactive_qa():
    """Run interactive Q&A session."""
    console.print("\n[bold blue]--- Initializing Q&A Engine ---[/bold blue]")

    if config.RAG_LLM_MODE != "ollama" and not config.HF_TOKEN:
        console.print(
            "[red]Error: HF_TOKEN not set in .env file.[/red]\n"
            "See SETUP.md for instructions on getting your Hugging Face token."
        )
        sys.exit(1)

    try:
        qa_chain = build_qa_chain()  # returns (chain, retriever) tuple
    except Exception as e:
        console.print(f"[red]Failed to initialize Q&A chain: {e}[/red]")
        sys.exit(1)

    console.print(
        Panel(
            "Document Q&A is ready!\n"
            "Type your question and press Enter.\n"
            "Type [bold]'quit'[/bold] or [bold]'exit'[/bold] to stop.",
            title="RAG Document Assistant",
            border_style="green",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]Your question:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        try:
            result = ask_question(question, qa_chain)
        except KeyboardInterrupt:
            console.print("\n[yellow]Query interrupted. Press Ctrl+C again to exit.[/yellow]")
            continue

        try:
            console.print(Panel(result["answer"], title="Answer", border_style="green"))

            if result["source_documents"]:
                sources = set()
                for doc in result["source_documents"]:
                    src = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "")
                    label = f"{src}" + (f" (page {page})" if page != "" else "")
                    sources.add(label)
                console.print(f"[dim]Sources: {', '.join(sorted(sources))}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_ragas_check():
    """Run a RAGAS health check from the CLI."""
    console.print(
        Panel(
            "[bold]RAGAS Health Check[/bold]\n"
            "Evaluating RAG pipeline quality with RAGAS metrics",
            border_style="blue",
        )
    )

    if config.RAGAS_EVAL_LLM_MODE != "ollama" and not config.HF_TOKEN:
        console.print(
            "[red]Error: HF_TOKEN not set in .env file.[/red]\n"
            "Required for RAGAS evaluation LLM in API mode."
        )
        sys.exit(1)

    # Ensure there is an index to evaluate against.
    import os

    has_index = os.path.exists(config.CHROMA_PERSIST_DIR) and os.listdir(
        config.CHROMA_PERSIST_DIR
    )
    if not has_index:
        console.print(
            "[red]No indexed documents found. Run `python main.py --ingest` first.[/red]"
        )
        sys.exit(1)

    console.print("[dim]Building QA chain...[/dim]")
    try:
        chain_and_retriever = build_qa_chain()
    except Exception as e:
        console.print(f"[red]Failed to build QA chain: {e}[/red]")
        sys.exit(1)

    console.print("[dim]Running RAGAS evaluation (this may take a while)...[/dim]")
    console.print("[dim]Press Ctrl+C to cancel.[/dim]\n")

    try:
        from ragas_evaluation import run_health_check

        result = run_health_check(chain_and_retriever)
    except KeyboardInterrupt:
        console.print("\n[yellow]RAGAS evaluation cancelled by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]RAGAS evaluation failed: {e}[/red]")
        sys.exit(1)

    # --- Display results ---
    status = result["status"]
    metrics = result["metrics"]
    details = result["details"]

    status_style = "bold green" if status == "healthy" else "bold red"
    console.print(f"\nOverall Status: [{status_style}]{status.upper()}[/{status_style}]\n")

    # Metrics table
    table = Table(title="RAGAS Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Pass", justify="center")

    def _fmt_score(val):
        return f"{val:.4f}" if val is not None else "N/A"

    def _pass_fail(val, threshold):
        if val is None:
            return "[red]FAIL[/red]"
        return "[green]PASS[/green]" if val >= threshold else "[red]FAIL[/red]"

    thresholds = config.RAGAS_THRESHOLDS
    display_names = {
        "faithfulness": "Faithfulness",
        "response_relevancy": "Response Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    for key in ["faithfulness", "response_relevancy", "context_recall", "context_precision"]:
        score = metrics.get(key)
        threshold = thresholds[key]
        table.add_row(
            display_names[key],
            _fmt_score(score),
            f"{threshold:.2f}",
            _pass_fail(score, threshold),
        )

    console.print(table)

    # Per-question details
    console.print("\n[bold]Per-Question Details:[/bold]")
    for i, d in enumerate(details, 1):
        console.print(
            f"  {i}. [cyan]{d['question']}[/cyan]\n"
            f"     Contexts retrieved: {d['num_contexts']}\n"
            f"     Answer: {d['answer']}{'...' if len(d['answer']) >= 200 else ''}"
        )

    console.print(f"\n{result['message']}")
    return status == "healthy"


def main():
    console.print(
        Panel(
            "[bold]RAG Document Q&A System[/bold]\nPowered by Hugging Face + ChromaDB",
            border_style="blue",
        )
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        ingest_path = sys.argv[2] if len(sys.argv) > 2 else None
        ingest_documents(ingest_path)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        console.print("[yellow]Clearing ChromaDB index...[/yellow]")
        clear_collection()
        console.print("[green]Index cleared.[/green]")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--ragas":
        try:
            healthy = run_ragas_check()
        except KeyboardInterrupt:
            console.print("\n[yellow]RAGAS evaluation cancelled by user.[/yellow]")
            sys.exit(130)
        sys.exit(0 if healthy else 1)

    import os

    has_index = os.path.exists(config.CHROMA_PERSIST_DIR) and os.listdir(
        config.CHROMA_PERSIST_DIR
    )

    if not has_index:
        console.print("[yellow]No indexed documents found.[/yellow]")

    console.print(
        "\nEnter a file or directory path to ingest, or press Enter to "
        + (
            "use existing index"
            if has_index
            else f"use default '{config.DOCUMENTS_DIR}'"
        )
        + "."
    )
    path_input = (
        console.input("[bold cyan]Path:[/bold cyan] ").strip().strip('"').strip("'")
    )

    if path_input:
        if not ingest_documents(path_input):
            sys.exit(1)
    elif not has_index:
        if not ingest_documents():
            sys.exit(1)

    interactive_qa()


if __name__ == "__main__":
    main()
