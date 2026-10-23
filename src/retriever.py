"""
Vector Similarity Retriever Module
==================================
This module is the "Retrieval" engine of RAG.
It takes a natural language query, generates its embedding, and queries PostgreSQL
using pgvector's vector similarity operators to find the most relevant chunks.

Key Concept: How pgvector Similarity Search Works
-------------------------------------------------
1. **Query Embedding**: The user's query is converted into a vector with the *exact same*
   embedding model (`sentence-transformers/all-MiniLM-L6-v2`) used during ingestion.
2. **The `<=>` Operator**:
   In pgvector, `<=>` computes the **Cosine Distance**:
       Distance = 1 - Cosine Similarity
   - Distance = 0.0 means identical angle/direction (maximum similarity).
   - Distance = 1.0 means orthogonal (no relationship).
   - Distance = 2.0 means opposite.
3. **Similarity Score Calculation**:
   We compute `similarity_score = 1 - (embedding <=> query_vector::vector)`.
   A score closer to 1.0 indicates higher relevance.
4. **Ordering & Top-K**:
   `ORDER BY embedding <=> query_vector::vector ASC LIMIT K` retrieves the K closest chunks.
   PostgreSQL uses the HNSW index to locate these candidates in logarithmic time!
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel

from src.embedder import get_embedding
from src.db import get_connection

console = Console()


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Finds the most semantically relevant document chunks for a given query.

    Args:
        query: The natural language question or search phrase.
        top_k: Number of highest-ranking chunks to retrieve.
        min_similarity: Minimum cosine similarity score threshold (0.0 to 1.0).

    Returns:
        List of dictionaries containing:
        - id: database primary key
        - content: chunk text
        - metadata: metadata dict (source file, chunk index, etc.)
        - similarity_score: float value between 0.0 and 1.0
    """
    # Step 1: Embed the query into vector space
    query_vector = get_embedding(query)

    # Step 2: Execute similarity search in PostgreSQL with pgvector
    sql = """
        SELECT
            id,
            content,
            metadata,
            (1 - (embedding <=> %s::vector)) AS similarity_score
        FROM documents
        WHERE (1 - (embedding <=> %s::vector)) >= %s
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (query_vector, query_vector, min_similarity, query_vector, top_k),
            )
            results = cur.fetchall()

    return results


def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a clean, numbered context block
    ready to be injected into an LLM prompt.
    """
    if not chunks:
        return "No relevant context found in the database."

    formatted_sections = []
    for idx, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "Unknown Source")
        chunk_idx = chunk["metadata"].get("chunk_index", "?")
        score = chunk.get("similarity_score", 0.0)

        header = f"--- [Document {idx}: {source} (Chunk #{chunk_idx}, Similarity: {score:.3f})] ---"
        body = chunk["content"]
        formatted_sections.append(f"{header}\n{body}")

    return "\n\n".join(formatted_sections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pgvector semantic retrieval.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What subsurface discoveries were made on Mars?",
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest chunks to retrieve",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score threshold (0.0 to 1.0)",
    )

    args = parser.parse_args()

    console.print(f"[bold cyan]Searching for:[/bold cyan] \"{args.query}\" (top_k={args.top_k})")
    retrieved = retrieve_relevant_chunks(
        query=args.query, top_k=args.top_k, min_similarity=args.min_score
    )

    console.print(f"[bold green]Retrieved {len(retrieved)} matching chunks:[/bold green]\n")
    for i, item in enumerate(retrieved, 1):
        source = item["metadata"].get("source", "Unknown")
        chunk_idx = item["metadata"].get("chunk_index", "0")
        score = item["similarity_score"]

        console.print(
            Panel(
                item["content"],
                title=f"Result #{i} | Score: {score:.4f} | Source: {source} (Chunk #{chunk_idx})",
                border_style="green" if score > 0.5 else "yellow",
            )
        )
