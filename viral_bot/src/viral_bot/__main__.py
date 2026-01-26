"""
Command-line interface for the Viral Health Post Bot.

Usage:
    python -m viral_bot run              # Run the bot once
    python -m viral_bot run --dry-run    # Run without writing to Sheets
    python -m viral_bot debug-item URL   # Debug pipeline for a single URL
    python -m viral_bot serve            # Start the health server
    python -m viral_bot backfill         # Backfill older content
    python -m viral_bot export_csv       # Export outputs to CSV
    python -m viral_bot stats            # Show database statistics
    python -m viral_bot config           # Show current configuration
"""

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import get_settings
from .logging_conf import setup_logging, get_logger
from .db import get_database
from .main import run_bot, debug_single_url
from .server import run_server
from .sheets import test_connection as test_sheets

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool):
    """Viral Health Post Idea Bot CLI."""
    level = "DEBUG" if debug else get_settings().log_level
    setup_logging(level=level)


@cli.command()
@click.option("--freshness", "-f", type=int, help="Freshness window in hours")
@click.option("--max-outputs", "-n", type=int, help="Maximum outputs to generate")
@click.option("--dry-run", is_flag=True, help="Run without writing to Google Sheets")
def run(freshness: int, max_outputs: int, dry_run: bool):
    """
    Run the bot once (fetch, evaluate, generate, export).

    Examples:
      python -m viral_bot run
      python -m viral_bot run --dry-run
      python -m viral_bot run -f 24 -n 3
    """
    if dry_run:
        console.print(Panel("[bold yellow]Starting Viral Bot Run (DRY RUN)[/bold yellow]"))
    else:
        console.print(Panel("[bold green]Starting Viral Bot Run[/bold green]"))

    try:
        stats = asyncio.run(run_bot(
            freshness_hours=freshness,
            max_outputs=max_outputs,
            dry_run=dry_run,
        ))

        # Display results
        table = Table(title="Run Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Show key metrics
        metrics = [
            ("run_id", stats.get("run_id", "")),
            ("items_fetched", stats.get("items_fetched", 0)),
            ("items_with_content", stats.get("items_with_content", 0)),
            ("items_after_pre_filter", stats.get("items_after_pre_filter", 0)),
            ("items_after_dedup", stats.get("items_after_dedup", 0)),
            ("items_evaluated", stats.get("items_evaluated", 0)),
            ("items_relevant", stats.get("items_relevant", 0)),
            ("items_after_post_filter", stats.get("items_after_post_filter", 0)),
            ("items_after_novelty", stats.get("items_after_novelty", 0)),
            ("items_generated", stats.get("items_generated", 0)),
            ("items_exported", stats.get("items_exported", 0)),
        ]

        for key, value in metrics:
            table.add_row(key, str(value))

        console.print(table)

        if stats.get("errors"):
            console.print("[red]Errors:[/red]")
            for error in stats["errors"]:
                console.print(f"  - {error}")

        if stats.get("status") == "SUCCESS":
            console.print("\n[bold green]Run completed successfully![/bold green]")
        else:
            console.print("\n[bold red]Run failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise click.Abort()


@cli.command("debug-item")
@click.argument("url")
@click.option("--json-output", "-j", is_flag=True, help="Output results as JSON")
def debug_item(url: str, json_output: bool):
    """
    Debug the full pipeline for a single URL.

    Shows each step of processing including content fetching,
    pre-AI filtering, AI evaluation, and generation.

    Examples:
      python -m viral_bot debug-item https://pubmed.ncbi.nlm.nih.gov/12345678/
      python -m viral_bot debug-item https://www.bbc.com/news/health-12345 --json-output
    """
    console.print(Panel(f"[bold cyan]Debugging URL[/bold cyan]\n{url}"))

    try:
        result = asyncio.run(debug_single_url(url))

        if json_output:
            # Custom JSON encoder for datetime
            def json_encoder(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            console.print(json.dumps(result, indent=2, default=json_encoder))

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise click.Abort()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", type=int, help="Port to bind to")
@click.option("--with-scheduler", is_flag=True, help="Enable built-in scheduler")
def serve(host: str, port: int, with_scheduler: bool):
    """Start the health check server."""
    console.print(Panel("[bold blue]Starting Server[/bold blue]"))

    settings = get_settings()
    port = port or settings.port

    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Scheduler: {'Enabled' if with_scheduler else 'Disabled'}")
    console.print()

    run_server(host=host, port=port, with_scheduler=with_scheduler)


@cli.command()
@click.option("--days", "-d", type=int, default=2, help="Days to backfill")
def backfill(days: int):
    """Backfill content from the past N days."""
    console.print(Panel(f"[bold yellow]Backfilling last {days} days[/bold yellow]"))

    freshness_hours = days * 24

    try:
        stats = asyncio.run(run_bot(freshness_hours=freshness_hours))

        console.print(f"Items fetched: {stats.get('items_fetched', 0)}")
        console.print(f"Items generated: {stats.get('items_generated', 0)}")
        console.print("[bold green]Backfill complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise click.Abort()


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="outputs.csv", help="Output file path")
@click.option("--days", "-d", type=int, default=7, help="Days to export")
def export_csv(output: str, days: int):
    """Export recent outputs to CSV."""
    console.print(Panel(f"[bold cyan]Exporting to {output}[/bold cyan]"))

    db = get_database()
    session = db.get_session()

    try:
        outputs = db.get_recent_outputs(session, days=days)

        if not outputs:
            console.print("[yellow]No outputs found[/yellow]")
            return

        # Write to CSV
        output_path = Path(output)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Headers
            writer.writerow([
                "id",
                "run_id",
                "created_at",
                "archetype",
                "headline",
                "image_suggestion",
                "virality_score",
                "confidence",
                "extracted_claim",
                "source_name",
                "source_url",
                "status",
            ])

            # Data
            for out in outputs:
                item = out.content_item
                writer.writerow([
                    out.id,
                    out.run_id,
                    out.created_at.isoformat() if out.created_at else "",
                    out.archetype,
                    out.headline,
                    out.image_suggestion or "",
                    out.virality_score or 0,
                    round(out.confidence, 2) if out.confidence else 0,
                    out.extracted_claim or "",
                    item.source if item else "",
                    item.url if item else "",
                    out.status,
                ])

        console.print(f"[green]Exported {len(outputs)} outputs to {output_path}[/green]")

    finally:
        session.close()


@cli.command()
def stats():
    """Show database statistics."""
    db = get_database()
    session = db.get_session()

    try:
        db_stats = db.get_stats(session)

        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Content Items", str(db_stats["total_items"]))
        table.add_row("Total Evaluations", str(db_stats["total_evaluations"]))
        table.add_row("Total Outputs", str(db_stats["total_outputs"]))
        table.add_row("Total Feedback", str(db_stats["total_feedback"]))
        table.add_row("Total Runs", str(db_stats["total_runs"]))
        table.add_row("Successful Runs", str(db_stats["successful_runs"]))

        console.print(table)

    finally:
        session.close()


@cli.command()
def test_sheets():
    """Test Google Sheets connection."""
    console.print(Panel("[bold yellow]Testing Google Sheets Connection[/bold yellow]"))

    try:
        from .sheets import test_connection

        if test_connection():
            console.print("[bold green]Connection successful![/bold green]")
        else:
            console.print("[bold red]Connection failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise click.Abort()


@cli.command()
@click.option("--all", "show_all", is_flag=True, help="Show disabled sources too")
def sources(show_all: bool):
    """List configured content sources."""
    from .sources import get_source_registry

    registry = get_source_registry()

    table = Table(title="Content Sources")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Enabled", style="green")
    table.add_column("Priority")

    sources_list = registry.list_sources(enabled_only=not show_all)

    for source in sources_list:
        enabled = "[green]Yes[/green]" if source.enabled else "[red]No[/red]"
        source_type = type(source).__name__.replace("Source", "")
        table.add_row(source.name, source_type, enabled, str(source.priority))

    console.print(table)

    stats = registry.get_stats()
    console.print(f"\nTotal: {stats['total_sources']} | Enabled: {stats['enabled_sources']}")
    console.print(f"By category: {stats['by_category']}")


@cli.command()
@click.argument("output_id", type=int)
@click.option("--likes", "-l", type=int, help="Likes count")
@click.option("--shares", "-s", type=int, help="Shares count")
@click.option("--saves", "-v", type=int, help="Saves count")
@click.option("--notes", "-n", type=str, help="Additional notes")
def feedback(output_id: int, likes: int, shares: int, saves: int, notes: str):
    """Add feedback for an output."""
    db = get_database()
    session = db.get_session()

    try:
        from .db import Output

        output = session.query(Output).filter(Output.id == output_id).first()

        if not output:
            console.print(f"[red]Output {output_id} not found[/red]")
            raise click.Abort()

        db.save_feedback(
            session,
            output_id=output_id,
            likes=likes,
            shares=shares,
            saves=saves,
            notes=notes,
        )
        session.commit()

        console.print(f"[green]Feedback saved for output {output_id}[/green]")

    finally:
        session.close()


@cli.command()
def config():
    """Show current configuration (excluding secrets)."""
    settings = get_settings()

    console.print(Panel("[bold blue]Current Configuration[/bold blue]"))

    # Bot settings
    console.print("\n[cyan]Bot Settings:[/cyan]")
    console.print(f"  freshness_hours:     {settings.freshness_hours}")
    console.print(f"  max_outputs_per_run: {settings.max_outputs_per_run}")
    console.print(f"  min_virality_score:  {settings.min_virality_score}")

    # Anti-generic settings
    console.print("\n[cyan]Anti-Generic Settings:[/cyan]")
    console.print(f"  min_body_chars_paper:  {settings.min_body_chars_paper}")
    console.print(f"  min_body_chars_news:   {settings.min_body_chars_news}")
    console.print(f"  min_differentiators:   {settings.min_differentiators}")
    console.print(f"  enable_pre_ai_filter:  {settings.enable_pre_ai_filter}")
    console.print(f"  enable_post_ai_filter: {settings.enable_post_ai_filter}")

    # Novelty settings
    console.print("\n[cyan]Novelty Settings:[/cyan]")
    console.print(f"  novelty_threshold:       {settings.novelty_threshold}")
    console.print(f"  novelty_retention_days:  {settings.novelty_retention_days}")
    console.print(f"  novelty_use_embeddings:  {settings.novelty_use_embeddings}")
    console.print(f"  novelty_penalty_factor:  {settings.novelty_penalty_factor}")

    # Mixing settings
    console.print("\n[cyan]Mixing Settings:[/cyan]")
    console.print(f"  max_study_stat_per_run:     {settings.max_study_stat_per_run}")
    console.print(f"  min_non_study_stat_per_run: {settings.min_non_study_stat_per_run}")
    console.print(f"  max_per_archetype:          {settings.max_per_archetype}")

    # OpenAI
    console.print("\n[cyan]OpenAI:[/cyan]")
    console.print(f"  model:           {settings.openai_model}")
    console.print(f"  embedding_model: {settings.openai_embedding_model}")

    # Scheduler
    console.print("\n[cyan]Scheduler:[/cyan]")
    console.print(f"  enable_scheduler: {settings.enable_scheduler}")
    console.print(f"  schedule_hours:   {settings.schedule_hours}")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
