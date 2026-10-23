"""
Hugging Face Embedding Generator Module
=======================================
This module handles:
1. Loading an open-source Transformer embedding model from Hugging Face.
2. Converting raw text chunks and search queries into dense vector embeddings.

Key Concepts:
-------------
1. **What is a Vector Embedding?**
   An embedding model is a neural network trained to project human language into a
   high-dimensional geometric space (vector space). In this space, texts with similar
   meanings reside near each other, even if they use completely different vocabulary.
   Example: "The puppy barked" and "A small dog made a noise" will have very high similarity,
   whereas keyword search (like BM25 or SQL LIKE) might fail to connect them.

2. **Model: sentence-transformers/all-MiniLM-L6-v2**
   - **Provider**: Hugging Face / UKPLab
   - **Dimension**: 384 float values per embedding.
   - **Advantages**: Extremely lightweight (~90MB), runs locally on CPU with millisecond latency,
     100% free, no API keys, no internet required after initial download.
"""

import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

console = Console()

# Module-level cache for the loaded model to prevent reloading weights on every function call
_model_instance: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-loads and caches the SentenceTransformer model.
    The first call will download the model weights from Hugging Face if not cached locally.
    Subsequent calls will reuse the in-memory model.
    """
    global _model_instance
    if _model_instance is None:
        console.print(
            f"[bold blue]Loading Hugging Face embedding model: [cyan]{EMBEDDING_MODEL_NAME}[/cyan]...[/bold blue]"
        )
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        console.print(
            f"[bold green][OK] Embedding model ready! Output dimension: {EMBEDDING_DIMENSION}[/bold green]"
        )
    return _model_instance


def get_embedding(text: str) -> List[float]:
    """
    Generates a normalized 384-dimensional vector embedding for a single text string.

    Args:
        text: The text string to embed.

    Returns:
        List of 384 float numbers representing the semantic embedding.
    """
    model = get_embedding_model()
    # normalize_embeddings=True ensures vectors have a length of 1 (unit vectors).
    # For unit vectors, Dot Product equals Cosine Similarity, which makes similarity
    # searches computationally faster.
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generates embeddings for multiple texts in batches.
    Batch processing is significantly faster than embedding texts one by one.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to process concurrently through the neural network.

    Returns:
        List of float vectors, one for each input text.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 5,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


if __name__ == "__main__":
    # Test script: Demonstrate semantic similarity calculation between sentences
    console.print("[bold yellow]Testing Hugging Face Embedder & Cosine Similarity...[/bold yellow]\n")

    sentence_a = "The astronaut discovered water ice beneath the red Martian soil."
    sentence_b = "Subsurface ice was located by scientists on the planet Mars."
    sentence_c = "I love eating chocolate ice cream on warm summer afternoons."

    # Compute individual embeddings
    vec_a = np.array(get_embedding(sentence_a))
    vec_b = np.array(get_embedding(sentence_b))
    vec_c = np.array(get_embedding(sentence_c))

    # Since vectors are normalized (magnitude = 1), cosine similarity is simply the dot product!
    sim_a_b = float(np.dot(vec_a, vec_b))
    sim_a_c = float(np.dot(vec_a, vec_c))

    console.print(f"[bold]Sentence A:[/bold] \"{sentence_a}\"")
    console.print(f"[bold]Sentence B (Semantically related):[/bold] \"{sentence_b}\"")
    console.print(f"[bold]Sentence C (Unrelated):[/bold] \"{sentence_c}\"\n")

    console.print(
        f"[green]Similarity between A & B (Mars topics): [bold]{sim_a_b:.4f}[/bold] (High)[/green]"
    )
    console.print(
        f"[red]Similarity between A & C (Mars vs Ice cream): [bold]{sim_a_c:.4f}[/bold] (Low)[/red]"
    )
    console.print(f"\nVector length verified: {len(vec_a)} dimensions")
