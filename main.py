"""
RAG Practice - Main Entry Point & Command Center
================================================
This script provides an interactive menu or quick commands to run any phase
of the RAG workflow:

  1. Initialize Database & pgvector Table
  2. Test Text Chunking
  3. Test Hugging Face Embeddings & Cosine Similarity
  4. Ingest Documents into pgvector
  5. Search / Retrieve Relevant Chunks
  6. Ask a Question with Gemini RAG Pipeline (Single Query or Interactive)

You can also run any module directly:
  uv run python src/db.py
  uv run python src/chunker.py
  uv run python src/embedder.py
  uv run python src/ingest.py
  uv run python src/retriever.py
  uv run python src/rag_pipeline.py
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def show_menu():
    console.print(
        Panel.fit(
            "[bold cyan]RAG Practice Learning Hub[/bold cyan]\n"
            "[white]Hugging Face Embeddings + PostgreSQL (pgvector) + Google Gemini LLM[/white]\n\n"
            "[green]1.[/green] Initialize / Reset PostgreSQL pgvector Table ([dim]src/db.py[/dim])\n"
            "[green]2.[/green] Test Document Chunking ([dim]src/chunker.py[/dim])\n"
            "[green]3.[/green] Test Hugging Face Embedder & Cosine Similarity ([dim]src/embedder.py[/dim])\n"
            "[green]4.[/green] Ingest Sample Documents into pgvector ([dim]src/ingest.py[/dim])\n"
            "[green]5.[/green] Run Vector Similarity Search / Retrieval ([dim]src/retriever.py[/dim])\n"
            "[green]6.[/green] Run Full Gemini RAG Pipeline (Single Query) ([dim]src/rag_pipeline.py[/dim])\n"
            "[green]7.[/green] Start Interactive Gemini RAG Q&A Chat Loop ([dim]src/rag_pipeline.py -i[/dim])\n"
            "[red]0.[/red] Exit",
            title="[bold yellow]Menu[/bold yellow]",
            border_style="cyan",
        )
    )


def main():
    while True:
        show_menu()
        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="6")

        if choice == "0":
            console.print("[yellow]Exiting. Happy coding![/yellow]")
            break
        elif choice == "1":
            from src.db import init_db, get_document_count
            reset_choice = Prompt.ask("Reset existing table? (y/n)", choices=["y", "n"], default="n")
            init_db(drop_existing=(reset_choice == "y"))
            console.print(f"[cyan]Current documents in DB: {get_document_count()}[/cyan]\n")
        elif choice == "2":
            from src.config import BASE_DIR
            from src.chunker import load_and_chunk_file
            sample = BASE_DIR / "sample_data" / "mars_rover_chronicles.txt"
            chunks = load_and_chunk_file(sample, chunk_size=300, chunk_overlap=50)
            console.print(f"[green]Chunked {sample.name} into {len(chunks)} chunks.[/green]\n")
            for c in chunks[:3]:
                console.print(Panel(c.content, title=f"Chunk #{c.metadata['chunk_index']}"))
        elif choice == "3":
            from src.embedder import get_embedding
            import numpy as np
            s1 = "Astronauts landed on the Martian surface."
            s2 = "A spacecraft touched down on Mars."
            s3 = "A recipe for baking chocolate cookies."
            v1, v2, v3 = np.array(get_embedding(s1)), np.array(get_embedding(s2)), np.array(get_embedding(s3))
            console.print(f"S1: {s1}")
            console.print(f"S2: {s2}")
            console.print(f"S3: {s3}\n")
            console.print(f"[green]Similarity (S1 <-> S2): {float(np.dot(v1, v2)):.4f}[/green]")
            console.print(f"[red]Similarity (S1 <-> S3): {float(np.dot(v1, v3)):.4f}[/red]\n")
        elif choice == "4":
            from src.ingest import ingest_path
            from src.config import BASE_DIR
            ingest_path(BASE_DIR / "sample_data")
        elif choice == "5":
            from src.retriever import retrieve_relevant_chunks
            query = Prompt.ask("Enter search query", default="What did the Zephyr rover discover on Mars?")
            results = retrieve_relevant_chunks(query, top_k=3)
            for r in results:
                console.print(
                    Panel(
                        r["content"],
                        title=f"Source: {r['metadata'].get('source')} | Score: {r['similarity_score']:.4f}",
                    )
                )
        elif choice == "6":
            from src.rag_pipeline import answer_question_rag
            from rich.markdown import Markdown
            query = Prompt.ask("Enter question for Gemini RAG", default="What is the power source and scanning depth of the rover?")
            res = answer_question_rag(query, top_k=3)
            console.print("\n[bold green]─── Gemini Response ───[/bold green]")
            console.print(Markdown(res["answer"]))
        elif choice == "7":
            from src.rag_pipeline import interactive_session
            interactive_session()


if __name__ == "__main__":
    main()
