"""Load and chunk documents from a file path or directory."""

import os
from typing import List

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    PythonLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED_EXT = {".pdf", ".txt", ".csv", ".docx", ".py"}


def get_loader(file_path: str):
    """Return the appropriate loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,
        ".py": PythonLoader,
    }
    loader_cls = loaders.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path)


def load_from_path(path: str) -> List:
    """Load supported documents from a file or directory path.

    Args:
        path: Absolute or relative path to a file or directory.

    Returns:
        List of loaded LangChain Document objects.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if os.path.isfile(path):
        return _load_file(path)

    if os.path.isdir(path):
        return _load_directory(path)

    raise ValueError(f"Path is neither a file nor a directory: {path}")


def _load_file(file_path: str) -> List:
    """Load a single file."""
    filename = os.path.basename(file_path)
    loader = get_loader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = filename

    # Filter out documents with empty content (e.g. image-only / vector-glyph PDFs).
    non_empty = [doc for doc in docs if doc.page_content.strip()]
    if len(non_empty) < len(docs):
        skipped = len(docs) - len(non_empty)
        print(
            f"Warning: {skipped} section(s) in {filename} had no extractable text. "
            f"The file may be image-based or use non-standard text rendering. "
            f"Try re-exporting the PDF with a different tool, or use OCR."
        )

    print(f"Loaded: {filename} ({len(non_empty)}/{len(docs)} sections with text)")
    return non_empty


def _load_directory(directory: str) -> List:
    """Recursively load all supported documents from a directory."""
    documents = []
    for root, _dirs, files in os.walk(directory):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXT:
                continue
            file_path = os.path.join(root, filename)
            try:
                documents.extend(_load_file(file_path))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return documents


def load_documents(directory: str = config.DOCUMENTS_DIR) -> List:
    """Load all supported documents from a directory (backward-compatible)."""
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created documents directory: {directory}")
        print("Add your PDF, TXT, CSV, DOCX, or PY files there and re-run.")
        return []
    return _load_directory(directory)


def chunk_documents(documents: List) -> List:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document sections into {len(chunks)} chunks")
    return chunks
