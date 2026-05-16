"""Document Q&A using RAG with Hugging Face models."""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import config
from document_processor import load_documents, load_from_path, chunk_documents
from embedding_store import get_embedding_model, get_vector_store, index_documents
from rag_pipeline import build_qa_chain, ask_question

console = Console()


def ingest_documents(path: str = None):
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
            f"[yellow]No documents found.[/yellow]\n"
            f"Provide a path to a file or directory with supported files "
            f"(PDF, TXT, CSV, DOCX, PY)."
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

    if not config.HF_TOKEN:
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

            console.print(Panel(result["answer"], title="Answer", border_style="green"))

            if result["source_documents"]:
                sources = set()
                for doc in result["source_documents"]:
                    src = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "")
                    label = f"{src}" + (f" (page {page})" if page != "" else "")
                    sources.add(label)
                console.print(
                    f"[dim]Sources: {', '.join(sorted(sources))}[/dim]"
                )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    console.print(
        Panel(
            "[bold]RAG Document Q&A System[/bold]\n"
            "Powered by Hugging Face + ChromaDB",
            border_style="blue",
        )
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        ingest_path = sys.argv[2] if len(sys.argv) > 2 else None
        ingest_documents(ingest_path)
        return

    import os
    has_index = os.path.exists(config.CHROMA_PERSIST_DIR) and os.listdir(config.CHROMA_PERSIST_DIR)

    if not has_index:
        console.print("[yellow]No indexed documents found.[/yellow]")

    console.print(
        "\nEnter a file or directory path to ingest, or press Enter to "
        + ("use existing index" if has_index else f"use default '{config.DOCUMENTS_DIR}'")
        + "."
    )
    path_input = console.input("[bold cyan]Path:[/bold cyan] ").strip().strip('"').strip("'")

    if path_input:
        if not ingest_documents(path_input):
            sys.exit(1)
    elif not has_index:
        if not ingest_documents():
            sys.exit(1)

    interactive_qa()


if __name__ == "__main__":
    main()
