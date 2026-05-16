"""``dbsprout doctor`` command — diagnose the local environment."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from dbsprout.cli.console import console
from dbsprout.doctor import run_all_checks


def _glyph(status: str) -> str:
    """Map a check status to a colored Rich glyph."""
    if status == "warn":
        return "[yellow]⚠[/yellow]"
    if status == "fail":
        return "[red]✖[/red]"
    return "[green]✔[/green]"


def doctor_command(
    db: str | None = typer.Option(
        None, "--db", help="Database URL to test.", envvar="DBSPROUT_TARGET_DB"
    ),
    config_path: str | None = typer.Option(
        "dbsprout.toml", "--config", help="Config file to scan for secrets."
    ),
) -> None:
    """Run environment health checks and print actionable fixes."""
    cfg = Path(config_path) if config_path else None
    results = run_all_checks(db_url=db, config_path=cfg)

    table = Table(title="dbsprout doctor")
    table.add_column("")
    table.add_column("Category", style="cyan")
    table.add_column("Check")
    table.add_column("Message")
    for r in results:
        msg = r.message if r.fix is None else f"{r.message}\n[dim]fix: {r.fix}[/dim]"
        table.add_row(_glyph(r.status), r.category, r.name, msg)
    console.print(table)

    passed = sum(1 for r in results if r.status == "pass")
    warned = sum(1 for r in results if r.status == "warn")
    failed = sum(1 for r in results if r.status == "fail")
    console.print(
        f"Summary: [green]{passed} passed[/green], "
        f"[yellow]{warned} warnings[/yellow], [red]{failed} failed[/red]"
    )
    if failed:
        raise typer.Exit(code=1)
