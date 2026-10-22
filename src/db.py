"""
Database Connection & Vector Table Setup
========================================
This module handles:
1. Connecting to PostgreSQL using psycopg 3.
2. Enabling the `pgvector` extension inside Postgres.
3. Creating the `documents` table to store:
   - `content`: Original text chunk
   - `metadata`: JSON object containing source file, chunk number, etc.
   - `embedding`: Vector representation (384-dimensional float array)
4. Creating an HNSW (Hierarchical Navigable Small World) index for lightning-fast
   approximate nearest neighbor (ANN) vector search.

Key Concept: What is pgvector?
------------------------------
pgvector is an open-source extension for PostgreSQL that allows you to store
vector embeddings directly in PostgreSQL columns and perform similarity searches
using distance metrics such as:
- Cosine Distance (`<=>`) -> 1 - Cosine Similarity
- L2 Distance / Euclidean (`<->`)
- Inner Product (`<#>`)
"""

import sys
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from rich.console import Console

from src.config import (
    DATABASE_URI,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    EMBEDDING_DIMENSION,
)

console = Console()


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Context manager for database connections.
    Automatically registers pgvector type handlers with psycopg so that Python
    lists or numpy arrays can be directly mapped to/from PostgreSQL `vector` columns.
    """
    conn = psycopg.connect(DATABASE_URI, autocommit=True, row_factory=dict_row)
    try:
        # Register pgvector extension types with psycopg3 if the extension already exists
        try:
            register_vector(conn)
        except Exception:
            # During first-time init, the vector extension has not been created in Postgres yet
            pass
        yield conn
    finally:
        conn.close()


def init_db(drop_existing: bool = False) -> None:
    """
    Initializes PostgreSQL with pgvector extension and creates the documents table.

    Args:
        drop_existing: If True, drops the existing table first (useful for clean resets).
    """
    console.print("[bold blue]Connecting to PostgreSQL...[/bold blue]")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Step 1: Enable pgvector extension
            console.print("[yellow]-> Enabling 'vector' extension in PostgreSQL...[/yellow]")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Now that the extension is enabled, register the vector type on this connection
            register_vector(conn)

            if drop_existing:
                console.print("[red]-> Dropping existing 'documents' table...[/red]")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE;")

            # Step 2: Create documents table
            # Notice the column `embedding vector(384)` matching the Hugging Face model dimension.
            console.print(
                f"[yellow]-> Creating 'documents' table with vector({EMBEDDING_DIMENSION})...[/yellow]"
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({EMBEDDING_DIMENSION}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Step 3: Create HNSW vector index
            # Why HNSW?
            # Hierarchical Navigable Small World (HNSW) creates a multi-layer graph of vectors.
            # It provides faster query latency and higher recall than IVFFlat without requiring
            # a training step or a minimum number of existing rows.
            # `vector_cosine_ops` optimizes for cosine distance (<=>).
            console.print("[yellow]-> Creating HNSW cosine index for fast vector search...[/yellow]")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
                ON documents USING hnsw (embedding vector_cosine_ops);
                """
            )

    console.print(
        f"[bold green][OK] Database initialized successfully on {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}![/bold green]"
    )


def get_document_count() -> int:
    """Returns the total number of chunks currently stored in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM documents;")
            row = cur.fetchone()
            return row["total"] if row else 0


if __name__ == "__main__":
    import sys

    # If --reset flag is passed, drop and recreate
    reset = "--reset" in sys.argv
    try:
        init_db(drop_existing=reset)
        count = get_document_count()
        console.print(f"[bold cyan]Current stored document chunks: {count}[/bold cyan]")
    except Exception as e:
        console.print(f"[bold red]Database connection failed:[/bold red] {e}")
        console.print(
            "[yellow]Make sure the Docker container is running: [bold]docker compose up -d[/bold][/yellow]"
        )
