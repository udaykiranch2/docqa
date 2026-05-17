"""HTTP client for calling the FastAPI backend from Streamlit."""

from typing import Optional

import httpx

BASE_URL = "http://localhost:8000/api/v1"


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0)


def health_check() -> dict:
    with _client() as c:
        r = c.get(f"{BASE_URL}/health")
        r.raise_for_status()
        return r.json()


def get_stats() -> dict:
    with _client() as c:
        r = c.get(f"{BASE_URL}/documents/stats")
        r.raise_for_status()
        return r.json()


def upload_documents(files: list) -> dict:
    """Upload files (list of (filename, file-like objects)) to the backend."""
    with _client() as c:
        r = c.post(
            f"{BASE_URL}/documents/upload",
            files=[("files", (f[0], f[1])) for f in files],
        )
        r.raise_for_status()
        return r.json()


def clear_index() -> dict:
    with _client() as c:
        r = c.delete(f"{BASE_URL}/documents/clear")
        r.raise_for_status()
        return r.json()


def ask_question(question: str) -> dict:
    with _client() as c:
        r = c.post(f"{BASE_URL}/qa/ask", json={"question": question})
        r.raise_for_status()
        return r.json()
