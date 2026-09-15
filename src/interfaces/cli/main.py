import asyncio
import sys
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agents.workflow import AgenticRAGWorkflow
from src.config.settings import get_settings
from src.domain.schemas import IngestRequest
from src.ingestion.pipeline import IngestionPipeline
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.database import create_database_repository
from src.storage.vector_store import create_vector_store

cli = typer.Typer(help="Agentic Research Assistant CLI - Ingest, Search, and Ask without Docker or Airflow")
console = Console(highlight=False, legacy_windows=False)


def _get_services():
    settings = get_settings()
    repo = create_database_repository(settings)
    vector_store = create_vector_store(settings)
    embeddings = EmbeddingsService(settings)
    llm = LLMService(settings)
    cache = CacheService(settings)
    observability = ObservabilityService(settings)
    return settings, repo, vector_store, embeddings, llm, cache, observability


@cli.command("ingest-file")
def ingest_file(
    file_paths: list[str] = typer.Argument(..., help="Path(s) to local PDF file(s) to parse, embed, and index"),
    title: str = typer.Option(None, "--title", "-t", help="Optional custom title (for single file)"),
):
    """Ingest one or more local PDF files directly into SQLite and Vector Store."""
    settings, repo, vector_store, embeddings, _, _, _ = _get_services()
    pipeline = IngestionPipeline(
        settings=settings,
        repository=repo,
        vector_store=vector_store,
        embeddings_service=embeddings,
    )
    for fp in file_paths:
        console.print(f"[cyan]Parsing & Indexing PDF:[/cyan] {fp}...")
        res = asyncio.run(pipeline.ingest_pdf_file(fp, title=title if len(file_paths) == 1 else None))
        if res.errors == 0:
            console.print(f"[bold green][OK] {res.message}[/bold green]")
        else:
            console.print(f"[bold red][!] {res.message}[/bold red]")


@cli.command("ingest-dir")
def ingest_directory(
    dir_path: str = typer.Argument("data/sample_papers", help="Directory containing PDF files to index"),
):
    """Ingest all PDF files inside a directory into SQLite and Vector Store."""
    import os
    import glob
    settings, repo, vector_store, embeddings, _, _, _ = _get_services()
    pipeline = IngestionPipeline(
        settings=settings,
        repository=repo,
        vector_store=vector_store,
        embeddings_service=embeddings,
    )
    pdf_files = glob.glob(os.path.join(dir_path, "*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {dir_path}[/yellow]")
        return

    console.print(f"[bold cyan]Found {len(pdf_files)} PDF files in {dir_path}[/bold cyan]")
    for fp in pdf_files:
        console.print(f"[cyan]Indexing:[/cyan] {os.path.basename(fp)}...")
        res = asyncio.run(pipeline.ingest_pdf_file(fp))
        if res.errors == 0:
            console.print(f"  [green]✓[/green] {res.message}")
        else:
            console.print(f"  [red]✗[/red] {res.message}")


@cli.command("seed")
def seed_benchmark_papers():
    """Seed seminal AI research papers into SQLite and Vector Store without network delays."""
    settings, repo, vector_store, embeddings, _, _, _ = _get_services()
    pipeline = IngestionPipeline(
        settings=settings,
        repository=repo,
        vector_store=vector_store,
        embeddings_service=embeddings,
    )
    console.print("[cyan]Seeding benchmark AI research papers...[/cyan]")
    res = asyncio.run(pipeline.ingest_sample_papers())
    console.print(f"[bold green][OK] {res.message}[/bold green]")


@cli.command()
def ingest(
    query: str = typer.Option("cat:cs.AI OR cat:cs.LG", "--query", "-q", help="arXiv search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of papers to fetch"),
    download_pdf: bool = typer.Option(True, "--download/--no-download", help="Download and parse full PDFs"),
):
    """Ingest papers directly from arXiv into database and vector store."""
    settings, repo, vector_store, embeddings, _, _, _ = _get_services()
    pipeline = IngestionPipeline(
        settings=settings,
        repository=repo,
        vector_store=vector_store,
        embeddings_service=embeddings,
    )

    console.print(Panel.fit(f"[bold green]Starting arXiv Ingestion[/bold green]\nQuery: [cyan]{query}[/cyan]\nLimit: [yellow]{limit}[/yellow]"))

    request = IngestRequest(query=query, max_results=limit, download_pdf=download_pdf)
    response = asyncio.run(pipeline.run(request))

    if response.errors == 0:
        console.print(f"[bold green][OK] {response.message}[/bold green]")
    else:
        console.print(f"[bold yellow][!] {response.message}[/bold yellow]")


@cli.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, bm25, or knn"),
    full: bool = typer.Option(False, "--full", "-f", help="Display full text and metadata for each individual chunk"),
):
    """Search and inspect indexed paper chunks using BM25, k-NN, or Hybrid search."""
    settings, _, vector_store, embeddings, _, _, _ = _get_services()

    console.print(f"[cyan]Searching for:[/cyan] '{query}' (mode: [yellow]{mode}[/yellow], top_k: {top_k})...")

    if mode == "bm25":
        hits = vector_store.search_bm25(query, top_k=top_k)
    elif mode == "knn":
        vec = asyncio.run(embeddings.embed_query(query))
        hits = vector_store.search_knn(vec, top_k=top_k)
    else:
        vec = asyncio.run(embeddings.embed_query(query))
        hits = vector_store.search_hybrid(query, vec, top_k=top_k)

    if not hits:
        console.print("[yellow]No matches found.[/yellow]")
        return

    if full:
        console.print(f"\n[bold green]Found {len(hits)} matching chunks:[/bold green]\n")
        for idx, hit in enumerate(hits, 1):
            sec = hit.section_title or "General"
            title_hdr = f"#{idx} | [{hit.arxiv_id}] {hit.title} | Chunk #{hit.chunk_index} (Section: {sec}) | Score: {hit.score:.4f}"
            console.print(Panel(hit.chunk_text, title=f"[bold cyan]{title_hdr}[/bold cyan]", border_style="blue"))
    else:
        table = Table(title=f"Search Results ({len(hits)} chunks)")
        table.add_column("Rank", justify="center", style="cyan")
        table.add_column("arXiv ID", style="magenta")
        table.add_column("Title", style="bold white")
        table.add_column("Chunk #", justify="center", style="yellow")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Snippet Preview", style="dim")

        for idx, hit in enumerate(hits, 1):
            snippet = hit.chunk_text[:100] + "..." if len(hit.chunk_text) > 100 else hit.chunk_text
            table.add_row(str(idx), hit.arxiv_id, hit.title[:35], f"#{hit.chunk_index}", f"{hit.score:.4f}", snippet)

        console.print(table)
        console.print("[dim]Tip: Pass '--full' / '-f' to view complete chunk contents and section metadata.[/dim]\n")


@cli.command()
def ask(
    query: str = typer.Argument(..., help="Research question"),
):
    """Ask a question through the self-correcting LangGraph Agentic RAG workflow."""
    settings, _, vector_store, embeddings, llm, cache, observability = _get_services()

    workflow = AgenticRAGWorkflow(
        settings=settings,
        llm_service=llm,
        vector_store=vector_store,
        embeddings_service=embeddings,
        cache_service=cache,
        observability_service=observability,
    )

    console.print(f"\n[bold cyan]Question:[/bold cyan] {query}")
    console.print("[dim]Running agentic reasoning workflow...[/dim]\n")

    response = asyncio.run(workflow.execute(query))

    console.print(Panel(response.answer, title="[bold green]AI Assistant Answer[/bold green]", expand=False))

    if response.sources:
        console.print("\n[bold]Citations & Sources:[/bold]")
        for src in response.sources:
            console.print(f" * [cyan]{src.arxiv_id}[/cyan]: [bold]{src.title}[/bold] (Chunk {src.chunk_index})")

    if response.reasoning_steps:
        console.print("\n[bold dim]Reasoning Steps:[/bold dim]")
        for step in response.reasoning_steps:
            console.print(f"  -> [dim]{step}[/dim]")

    console.print(f"\n[dim]Completed in {response.execution_time:.2f}s | Guardrail: {response.guardrail_score}/100[/dim]\n")


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", "-r"),
):
    """Start the FastAPI server."""
    uvicorn.run("src.interfaces.api.app:app", host=host, port=port, reload=reload)


@cli.command()
def ui(
    port: int = typer.Option(7860, "--port", "-p"),
):
    """Start the Gradio Web Interface."""
    from src.interfaces.ui.app import launch_ui
    launch_ui(server_port=port)


if __name__ == "__main__":
    cli()
