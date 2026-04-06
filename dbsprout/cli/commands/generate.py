"""``dbsprout generate`` command — orchestrate data generation and output."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import GenerateResult, orchestrate
from dbsprout.schema.models import DatabaseSchema

if TYPE_CHECKING:
    from dbsprout.quality.integrity import IntegrityReport

console = Console()


def generate_command(  # noqa: PLR0913
    schema_snapshot: Path | None = None,
    config_path: Path | None = None,
    rows: int = 100,
    seed: int = 42,
    output_format: str = "sql",
    output_dir: Path = Path("./seeds"),
    dialect: str = "postgresql",
    engine: str = "heuristic",
    privacy: str = "local",  # noqa: ARG001
    target_db: str | None = None,
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

    # Orchestrate
    result = orchestrate(schema, config, seed=seed, default_rows=rows, engine=engine)

    if result.total_tables == 0:
        console.print("[yellow]No tables to generate.[/yellow]")
        raise typer.Exit(code=0)

    # Write output
    _write_output(
        result, schema, result.insertion_order, output_dir, output_format, dialect, target_db
    )

    # Validate integrity
    from dbsprout.quality.integrity import validate_integrity  # noqa: PLC0415

    report = validate_integrity(result.tables_data, schema)
    _print_validation(report)

    # Summary
    _print_summary(result, output_dir, output_format)

    if not report.passed:
        console.print("[red]Integrity validation FAILED.[/red]")
        raise typer.Exit(code=1)


def _resolve_schema_path(explicit: Path | None) -> Path | None:
    """Resolve schema snapshot path."""
    if explicit is not None:
        return explicit
    default = Path(".dbsprout/schema.json")
    if default.exists():
        return default
    return None


def _write_output(  # noqa: PLR0913
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    output_dir: Path,
    output_format: str,
    dialect: str,
    target_db: str | None = None,
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
    elif output_format == "parquet":
        from dbsprout.output.parquet_writer import ParquetWriter  # noqa: PLC0415

        ParquetWriter().write(result.tables_data, schema, insertion_order, output_dir)
    elif output_format == "direct":
        if not target_db:
            console.print("[red]Error:[/red] --db is required when using --output-format direct")
            raise typer.Exit(code=1)
        _run_direct_insert(result, schema, insertion_order, target_db)
    else:
        console.print(f"[red]Error:[/red] Unknown output format: {output_format}")
        raise typer.Exit(code=1)


def _print_validation(report: IntegrityReport) -> None:
    """Print integrity validation results as a Rich table."""
    if not report.checks:
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


def _detect_direct_dialect(url: str) -> str:
    """Detect database dialect from a connection URL prefix."""
    lower = url.lower()
    if lower.startswith(("postgresql", "postgres")):
        return "postgresql"
    if lower.startswith("mysql"):
        return "mysql"
    return lower.split("://")[0].split("+")[0] if "://" in lower else "unknown"


def _run_direct_insert(
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    target_db: str,
) -> None:
    """Dispatch direct insertion to the appropriate dialect writer."""
    from dbsprout.output.pg_copy import InsertResult  # noqa: PLC0415

    dialect = _detect_direct_dialect(target_db)

    insert_result: InsertResult
    if dialect == "postgresql":
        from dbsprout.output.pg_copy import PgCopyWriter  # noqa: PLC0415

        insert_result = PgCopyWriter().write(result.tables_data, schema, insertion_order, target_db)
    elif dialect == "mysql":
        from dbsprout.output.mysql_load_data import MysqlLoadDataWriter  # noqa: PLC0415

        insert_result = MysqlLoadDataWriter().write(
            result.tables_data, schema, insertion_order, target_db
        )
    else:
        console.print(f"[red]Error:[/red] Direct insertion not supported for dialect: {dialect}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Inserted {insert_result.total_rows} rows "
        f"into {insert_result.tables_inserted} tables "
        f"in {insert_result.duration_seconds:.3f}s[/green]"
    )


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
