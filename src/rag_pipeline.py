"""
End-to-End RAG Pipeline with Google Gemini
===========================================
This module ties together all stages of the RAG (Retrieval-Augmented Generation) system:

1. Retrieval: Query embedding & pgvector cosine similarity search (`src.retriever`).
2. Augmentation: Grounded prompt template incorporating retrieved chunks and source tags.
3. Generation: LLM generation via the official Google Gemini SDK (`google-genai`).

Why RAG Solves LLM Hallucinations:
----------------------------------
LLMs are probabilistic word predictors trained on static data. When asked about
private documents, recent events, or specific numbers, they may generate fluent
yet completely fabricated answers ("hallucinations").

By explicitly supplying retrieved context and instructing the model:
"Answer ONLY based on the provided context below. If the context does not contain
the information, state that it is not available,"
the model acts as an analytical reasoning engine rather than a memorized knowledge store.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    validate_gemini_key,
)
from src.retriever import retrieve_relevant_chunks, format_context_for_llm

console = Console()

# Grounded RAG Prompt Template
RAG_SYSTEM_INSTRUCTION = """\
You are an expert, truthful AI assistant grounded strictly in verified reference documents.
Your task is to answer the user's question accurately using ONLY the provided Context below.

Rules:
1. Ground your answer completely in the provided context.
2. If the context does not contain enough information to answer the question, state honestly:
   "Based on the provided documents, I do not have enough information to answer this question."
3. Do not assume or invent facts outside the provided context.
4. Cite your sources where applicable (e.g., "[Source: filename.txt, Chunk #X]").
"""

RAG_PROMPT_TEMPLATE = """\
Context Information:
---------------------
{context}
---------------------

User Question: {question}

Helpful & Grounded Answer:"""


def get_gemini_client() -> genai.Client:
    """Initializes and returns the official Google GenAI client."""
    validate_gemini_key()
    return genai.Client(api_key=GEMINI_API_KEY)


def answer_question_rag(
    question: str,
    top_k: int = 3,
    min_similarity: float = 0.0,
    show_context: bool = True,
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline for a question:
    1. Retrieves top-k chunks from PostgreSQL pgvector.
    2. Builds an augmented prompt.
    3. Calls Google Gemini to generate the response.

    Args:
        question: The user's query.
        top_k: Number of chunks to retrieve.
        min_similarity: Minimum cosine similarity threshold.
        show_context: If True, prints retrieved chunks to the terminal.

    Returns:
        Dictionary containing answer, retrieved chunks, and the prompt used.
    """
    # Step 1: Retrieval
    console.print(f"[bold cyan]Searching pgvector for:[/bold cyan] \"{question}\"")
    retrieved_chunks = retrieve_relevant_chunks(
        query=question, top_k=top_k, min_similarity=min_similarity
    )

    if show_context:
        console.print(f"\n[bold yellow]─── Retrieved Context ({len(retrieved_chunks)} chunks) ───[/bold yellow]")
        for idx, item in enumerate(retrieved_chunks, 1):
            source = item["metadata"].get("source", "Unknown")
            chunk_num = item["metadata"].get("chunk_index", 0)
            score = item["similarity_score"]
            console.print(
                Panel(
                    item["content"],
                    title=f"Source [{idx}]: {source} (Chunk #{chunk_num}) | Cosine Similarity: {score:.4f}",
                    border_style="dim blue",
                )
            )

    # Step 2: Augmentation
    formatted_context = format_context_for_llm(retrieved_chunks)
    augmented_prompt = RAG_PROMPT_TEMPLATE.format(
        context=formatted_context,
        question=question,
    )

    # Step 3: Generation with Gemini
    console.print(f"[bold magenta]Synthesizing answer with Gemini ({GEMINI_MODEL})...[/bold magenta]")
    client = get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=augmented_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            temperature=0.2,  # Low temperature for factual, grounded precision
        ),
    )

    answer_text = response.text or "No response generated."

    return {
        "question": question,
        "answer": answer_text,
        "chunks": retrieved_chunks,
        "prompt": augmented_prompt,
    }


def interactive_session(top_k: int = 3) -> None:
    """Runs an interactive conversational loop in the terminal."""
    console.print(
        Panel.fit(
            "[bold green]Interactive RAG Q&A Session Started![/bold green]\n"
            "Ask questions against your pgvector knowledge base.\n"
            "Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.",
            title="RAG Interactive CLI",
            border_style="green",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]Your Question > [/bold cyan]").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting session. Happy learning![/yellow]")
                break

            result = answer_question_rag(question=question, top_k=top_k)

            console.print("\n[bold green]─── Gemini RAG Answer ───[/bold green]")
            console.print(Markdown(result["answer"]))

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
            break
        except Exception as err:
            console.print(f"[bold red]Error:[/bold red] {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query knowledge base using Gemini RAG pipeline.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What is the power source and scanning depth of the Zephyr-IV rover?",
        help="Question to ask",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive Q&A session",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to include in prompt",
    )
    parser.add_argument(
        "--hide-context",
        action="store_true",
        help="Hide the retrieved context panels from console",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_session(top_k=args.top_k)
    else:
        try:
            res = answer_question_rag(
                question=args.question,
                top_k=args.top_k,
                show_context=not args.hide_context,
            )
            console.print("\n[bold green]─── Gemini RAG Answer ───[/bold green]")
            console.print(Markdown(res["answer"]))
        except Exception as e:
            console.print(f"[bold red]RAG Pipeline error:[/bold red] {e}")
