"""Launch the RAGAS health check from the command line.

Usage:
    python run_ragas.py

Exits with code 0 if the RAG pipeline is healthy, 1 otherwise, 130 if cancelled.
"""

from rich.console import Console

from main import run_ragas_check
import sys

console = Console()

if __name__ == "__main__":
    try:
        healthy = run_ragas_check()
    except KeyboardInterrupt:
        console.print("\n[yellow]RAGAS evaluation cancelled by user.[/yellow]")
        sys.exit(130)
    sys.exit(0 if healthy else 1)
