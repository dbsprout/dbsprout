"""``dbsprout generate`` command — orchestrate data generation and output."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import GenerateResult, orchestrate
from dbsprout.schema.models import DatabaseSchema

console = Console()


def generate_command(  # noqa: PLR0913
    schema_snapshot: Path | None = typer.Option(
        None,
        "--schema-snapshot",
        help="Path to schema snapshot JSON. Default: .dbsprout/schema.json",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        help="Path to dbsprout.toml. Default: ./dbsprout.toml",
    ),
    rows: int = typer.Option(
        100,
        "--rows",
        "-n",
        help="Default row count per table.",
        min=1,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Global seed for deterministic output.",
        min=0,
    ),
    output_format: str = typer.Option(
        "sql",
        "--output-format",
        "-f",
        help="Output format: sql, csv, json, jsonl.",
    ),
    output_dir: Path = typer.Option(
        Path("./seeds"),
        "--output-dir",
        "-o",
        help="Output directory for generated files.",
    ),
    dialect: str = typer.Option(
        "postgresql",
        "--dialect",
        "-d",
        help="SQL dialect: postgresql, mysql, sqlite.",
    ),
) -> None:
    """Generate seed data from a schema snapshot."""
    # Load schema
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

    # Override config seed/rows from CLI
    effective_rows = rows
    effective_seed = seed

    # Orchestrate
    result = orchestrate(schema, config, seed=effective_seed, default_rows=effective_rows)

    if result.total_tables == 0:
        console.print("[yellow]No tables to generate.[/yellow]")
        raise typer.Exit(code=0)

    # Write output
    insertion_order = _flat_insertion_order(schema, result)
    _write_output(result, schema, insertion_order, output_dir, output_format, dialect)

    # Summary
    _print_summary(result, output_dir, output_format)


def _resolve_schema_path(explicit: Path | None) -> Path | None:
    """Resolve schema snapshot path."""
    if explicit is not None:
        return explicit
    default = Path(".dbsprout/schema.json")
    if default.exists():
        return default
    return None


def _flat_insertion_order(
    schema: DatabaseSchema,
    result: GenerateResult,
) -> list[str]:
    """Get flat insertion order for output file naming."""
    from graphlib import CycleError  # noqa: PLC0415

    from dbsprout.schema.graph import FKGraph, resolve_cycles  # noqa: PLC0415

    try:
        graph = FKGraph.from_schema(schema)
    except CycleError:
        graph = resolve_cycles(schema).graph

    flat: list[str] = []
    for batch in graph.insertion_order:
        flat.extend(sorted(batch))

    # Only include tables that were actually generated
    return [t for t in flat if t in result.tables_data]


def _write_output(  # noqa: PLR0913
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    output_dir: Path,
    output_format: str,
    dialect: str,
) -> None:
    """Write generated data using the selected output writer."""
    if output_format == "sql":
        from dbsprout.output.sql_writer import SQLWriter  # noqa: PLC0415

        SQLWriter().write(
            result.tables_data,
            schema,
            insertion_order,
            output_dir,
            dialect=dialect,
        )
    elif output_format == "csv":
        from dbsprout.output.csv_writer import CSVWriter  # noqa: PLC0415

        CSVWriter().write(result.tables_data, schema, insertion_order, output_dir)
    elif output_format in ("json", "jsonl"):
        from dbsprout.output.json_writer import JSONWriter  # noqa: PLC0415

        JSONWriter().write(
            result.tables_data,
            schema,
            insertion_order,
            output_dir,
            fmt=output_format,  # type: ignore[arg-type]
        )
    else:
        console.print(f"[red]Error:[/red] Unknown output format: {output_format}")
        raise typer.Exit(code=1)


def _print_summary(
    result: GenerateResult,
    output_dir: Path,
    output_format: str,
) -> None:
    """Print generation summary."""
    table = Table(title="Generation Complete", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Tables", str(result.total_tables))
    table.add_row("Total rows", str(result.total_rows))
    table.add_row("Duration", f"{result.duration_seconds:.3f}s")
    table.add_row("Output", str(output_dir))
    table.add_row("Format", output_format)

    console.print(table)
