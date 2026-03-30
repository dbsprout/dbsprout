"""``dbsprout validate`` command — integrity validation with Rich report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.quality.integrity import validate_integrity
from dbsprout.schema.models import DatabaseSchema

console = Console()


def validate_command(  # noqa: PLR0913
    schema_snapshot: Path | None = typer.Option(
        None,
        "--schema-snapshot",
        help="Path to schema snapshot JSON. Default: .dbsprout/schema.json",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Path to dbsprout.toml.",
    ),
    rows: int = typer.Option(100, "--rows", "-n", help="Row count per table.", min=1),
    seed: int = typer.Option(42, "--seed", "-s", help="Global seed.", min=0),
    output_format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="Output format: rich (default) or json.",
    ),
    engine: str = typer.Option(
        "heuristic",
        "--engine",
        "-e",
        help="Generation engine: heuristic or spec.",
    ),
) -> None:
    """Validate integrity of generated seed data."""
    # Resolve schema
    snapshot_path = _resolve_schema_path(schema_snapshot)
    if snapshot_path is None or not snapshot_path.exists():
        console.print("[red]Error:[/red] No schema snapshot found.")
        console.print("Run [bold]dbsprout init[/bold] first, or use --schema-snapshot.")
        raise typer.Exit(code=1)

    raw = snapshot_path.read_text(encoding="utf-8")
    schema = DatabaseSchema.model_validate_json(raw)

    # Load config
    cfg_path = config_path or Path("dbsprout.toml")
    config = DBSproutConfig.from_toml(cfg_path if cfg_path.exists() else None)

    # Generate data
    result = orchestrate(schema, config, seed=seed, default_rows=rows, engine=engine)

    # Validate
    report = validate_integrity(result.tables_data, schema)

    # Output
    if output_format == "json":
        _print_json(report)
    else:
        _print_rich(report)

    if not report.passed:
        raise typer.Exit(code=1)


def _resolve_schema_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    default = Path(".dbsprout/schema.json")
    if default.exists():
        return default
    return None


def _print_rich(report: object) -> None:
    """Print Rich table with validation results."""
    from dbsprout.quality.integrity import IntegrityReport  # noqa: PLC0415

    assert isinstance(report, IntegrityReport)

    if not report.checks:
        console.print("[green]No checks to run (empty schema).[/green]")
        return

    table = Table(title="Integrity Validation")
    table.add_column("Check", style="bold")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Status")
    table.add_column("Details")

    for check in report.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.check, check.table, check.column, status, check.details)

    console.print(table)

    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    console.print(f"\n{passed}/{total} checks passed.")


def _print_json(report: object) -> None:
    """Print JSON output for CI integration."""
    from dbsprout.quality.integrity import IntegrityReport  # noqa: PLC0415

    assert isinstance(report, IntegrityReport)

    data = {
        "passed": report.passed,
        "checks": [
            {
                "check": c.check,
                "table": c.table,
                "column": c.column,
                "passed": c.passed,
                "details": c.details,
            }
            for c in report.checks
        ],
    }
    console.print(json.dumps(data, indent=2))
