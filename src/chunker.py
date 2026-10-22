"""
Document Loading & Chunking Module
==================================
This module is responsible for:
1. Reading documents from the local filesystem (.txt, .md).
2. Splitting documents into smaller, overlapping chunks suitable for embedding models.

Why Chunking is Critical in RAG:
--------------------------------
1. **Embedding Granularity**: If you embed an entire 50-page document into a single vector,
   the embedding becomes a vague average of all topics. When a user asks a specific question,
   retrieval accuracy drops. Smaller chunks produce highly specific, targeted embeddings.
2. **Context Window Efficiency**: LLMs have token limits and higher costs/latencies with huge prompts.
   Sending only the 3-5 most relevant chunks gives the model the exact facts it needs.
3. **The Role of Overlap**: Without overlap, a key sentence split across chunk boundaries
   (e.g., "The critical password is" in chunk 1, "Alpha-9" in chunk 2) would lose its semantic connection.
   Overlap ensures boundary sentences appear fully in at least one chunk.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

console = Console()


@dataclass
class DocumentChunk:
    """Represents an extracted chunk of text with its corresponding metadata."""
    content: str
    metadata: Dict[str, Any]


def load_file(file_path: Path | str) -> str:
    """Reads a text or markdown file from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    source_name: str = "unknown",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Splits text into chunks of `chunk_size` characters with `chunk_overlap`.

    Sliding Window Strategy:
    - Start at position 0.
    - Advance by (chunk_size - chunk_overlap) at each step.
    - Strip whitespace to avoid empty trailing spaces.

    Args:
        text: The raw document text to split.
        source_name: Identifier for the source file or origin.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of DocumentChunk objects containing text content and metadata.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    chunks: List[DocumentChunk] = []
    text_length = len(text)
    step = chunk_size - chunk_overlap

    chunk_index = 0
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_str = text[start:end].strip()

        if chunk_str:  # Avoid inserting blank chunks
            metadata = {
                "source": source_name,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "char_length": len(chunk_str),
            }
            chunks.append(DocumentChunk(content=chunk_str, metadata=metadata))
            chunk_index += 1

        if end == text_length:
            break
        start += step

    return chunks


def load_and_chunk_file(
    file_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Loads a single file and splits it into chunks."""
    path = Path(file_path)
    raw_text = load_file(path)
    return chunk_text(
        text=raw_text,
        source_name=path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_and_chunk_directory(
    dir_path: Path | str,
    extensions: tuple = (".txt", ".md"),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Recursively loads all matching files in a directory and chunks them."""
    path = Path(dir_path)
    all_chunks: List[DocumentChunk] = []

    for file in path.rglob("*"):
        if file.is_file() and file.suffix.lower() in extensions:
            chunks = load_and_chunk_file(
                file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    # Test script demonstrating chunking on sample data
    from src.config import BASE_DIR

    sample_file = BASE_DIR / "sample_data" / "mars_rover_chronicles.txt"
    if sample_file.exists():
        console.print(f"[bold cyan]Testing chunker on: {sample_file.name}[/bold cyan]")
        test_chunks = load_and_chunk_file(sample_file, chunk_size=300, chunk_overlap=50)

        console.print(f"[green]Total chunks generated: {len(test_chunks)}[/green]\n")
        for i, chunk in enumerate(test_chunks):
            console.print(
                Panel(
                    chunk.content,
                    title=f"Chunk #{chunk.metadata['chunk_index']} (Source: {chunk.metadata['source']})",
                    subtitle=f"Chars: {chunk.metadata['char_length']} [{chunk.metadata['start_char']}:{chunk.metadata['end_char']}]",
                    border_style="blue",
                )
            )
    else:
        console.print(f"[red]Sample file not found at {sample_file}[/red]")
