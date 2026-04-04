"""``dbsprout audit`` command — display LLM interaction audit log."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dbsprout.privacy.audit import AuditLog

console = Console()

_DEFAULT_LOG = Path(".dbsprout/audit.log")


def audit_command(
    last: int | None = typer.Option(
        None,
        "--last",
        "-n",
        help="Show only the N most recent entries.",
        min=1,
    ),
) -> None:
    """Show the LLM interaction audit log."""
    audit = AuditLog(path=_DEFAULT_LOG)
    events = audit.read(limit=last)

    if not events:
        console.print("[yellow]No audit entries found.[/yellow]")
        return

    table = Table(title=f"Audit Log ({len(events)} entries)")
    table.add_column("Timestamp", style="dim")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Tier")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Cached")
    table.add_column("Schema")

    for event in events:
        tokens = f"{event.tokens_sent}/{event.tokens_received}"
        cost = f"${event.cost_estimate:.4f}" if event.cost_estimate > 0 else "$0.00"
        cached = "[green]yes[/green]" if event.cached else "no"
        table.add_row(
            event.timestamp,
            event.provider,
            event.model or "-",
            event.privacy_tier or "-",
            tokens,
            cost,
            cached,
            (event.schema_hash or "-")[:8],
        )

    console.print(table)
