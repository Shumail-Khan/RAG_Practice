"""
Document Ingestion Pipeline Module
==================================
This module connects the components together to build the ingestion pipeline:
1. Load documents from disk.
2. Split documents into chunks with metadata using `chunker.py`.
3. Compute dense vector embeddings using `embedder.py`.
4. Store chunks, metadata (JSONB), and vector embeddings (pgvector) in PostgreSQL.

Concept: The Ingestion Workflow
-------------------------------
Raw Documents -> Text Chunker -> Hugging Face Embedder -> PostgreSQL (pgvector)
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table

from src.config import BASE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.chunker import load_and_chunk_file, load_and_chunk_directory, DocumentChunk
from src.embedder import get_embeddings_batch
from src.db import get_connection, init_db, get_document_count

console = Console()


def store_chunks_in_db(chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
    """
    Inserts chunk texts, metadata, and corresponding vector embeddings into PostgreSQL.

    Args:
        chunks: List of DocumentChunk instances.
        embeddings: List of embedding vectors matching each chunk.

    Returns:
        Number of rows inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Mismatch between number of chunks and number of embeddings")

    console.print(f"[bold blue]Inserting {len(chunks)} chunks into pgvector database...[/bold blue]")

    # Prepare parameter tuples for fast batch insertion
    # (content, metadata_json, embedding_vector)
    rows_to_insert = [
        (
            chunk.content,
            json.dumps(chunk.metadata),
            embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    query = """
        INSERT INTO documents (content, metadata, embedding)
        VALUES (%s, %s::jsonb, %s::vector);
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            # executemany executes the batch efficiently
            cur.executemany(query, rows_to_insert)

    console.print(f"[bold green][OK] Successfully inserted {len(rows_to_insert)} records into PostgreSQL![/bold green]")
    return len(rows_to_insert)


def ingest_path(
    target_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset_first: bool = False,
) -> int:
    """
    End-to-end ingestion runner for a file or directory.

    Args:
        target_path: Path to a file or directory.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between chunks.
        reset_first: If True, re-initializes and empties the database first.

    Returns:
        Total number of chunks ingested.
    """
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    start_time = time.time()

    if reset_first:
        init_db(drop_existing=True)

    # Step 1: Chunking
    console.print(f"[bold cyan]Step 1/3: Loading and chunking documents from '{path}'...[/bold cyan]")
    if path.is_file():
        chunks = load_and_chunk_file(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        chunks = load_and_chunk_directory(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        console.print("[yellow]No documents or chunks found to ingest.[/yellow]")
        return 0

    console.print(f"-> Generated [bold green]{len(chunks)}[/bold green] chunks.")

    # Step 2: Embedding
    console.print("[bold cyan]Step 2/3: Generating embeddings with Hugging Face model...[/bold cyan]")
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = get_embeddings_batch(chunk_texts)
    console.print(f"-> Generated [bold green]{len(embeddings)}[/bold green] embeddings (384 dimensions each).")

    # Step 3: Database Storage
    console.print("[bold cyan]Step 3/3: Storing in PostgreSQL pgvector...[/bold cyan]")
    inserted_count = store_chunks_in_db(chunks, embeddings)

    elapsed = time.time() - start_time
    total_in_db = get_document_count()

    # Display summary table
    table = Table(title="Ingestion Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Source Path", str(path))
    table.add_row("Chunks Ingested This Run", str(inserted_count))
    table.add_row("Total Chunks in DB", str(total_in_db))
    table.add_row("Time Taken", f"{elapsed:.2f} seconds")

    console.print(table)
    return inserted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector database.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(BASE_DIR / "sample_data"),
        help="Path to file or folder to ingest (defaults to sample_data/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database table before ingestion",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Character size of each chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap characters between chunks",
    )

    args = parser.parse_args()

    try:
        ingest_path(
            target_path=args.path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            reset_first=args.reset,
        )
    except Exception as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
