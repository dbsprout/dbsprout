"""``dbsprout validate`` command — integrity + fidelity validation with Rich report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.quality.integrity import IntegrityReport, validate_integrity
from dbsprout.schema.models import DatabaseSchema

if TYPE_CHECKING:
    from dbsprout.quality.fidelity import FidelityReport

console = Console()


def validate_command(  # noqa: PLR0913
    schema_snapshot: Path | None = None,
    config_path: Path | None = None,
    rows: int = 100,
    seed: int = 42,
    output_format: str = "rich",
    engine: str = "heuristic",
    reference_data: Path | None = None,
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

    # Validate integrity
    integrity_report = validate_integrity(result.tables_data, schema)

    # Validate fidelity (optional — requires --reference-data and [stats] extra)
    fidelity_report: FidelityReport | None = None
    if reference_data is not None:
        fidelity_report = _run_fidelity(result.tables_data, reference_data, schema)

    # Output
    if output_format == "json":
        _print_json(integrity_report, fidelity_report)
    else:
        _print_rich(integrity_report)
        if fidelity_report is not None:
            _print_fidelity_rich(fidelity_report)

    if not integrity_report.passed:
        raise typer.Exit(code=1)


def _resolve_schema_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    default = Path(".dbsprout/schema.json")
    if default.exists():
        return default
    return None


def _print_rich(report: IntegrityReport) -> None:
    """Print Rich table with validation results."""

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


def _run_fidelity(
    tables_data: dict[str, list[dict[str, Any]]],
    reference_path: Path,
    schema: DatabaseSchema,
) -> FidelityReport:
    """Load reference data and compute fidelity metrics."""
    from dbsprout.quality.fidelity import (  # noqa: PLC0415
        FidelityReport,
        load_reference_csv,
        validate_fidelity,
    )

    if not reference_path.exists():
        console.print(f"[red]Error:[/red] Reference data not found: {reference_path}")
        return FidelityReport()

    # Load reference CSV — one file per table, match by table name
    ref_data: dict[str, list[dict[str, Any]]] = {}
    if reference_path.is_dir():
        for table in schema.tables:
            csv_path = reference_path / f"{table.name}.csv"
            if csv_path.exists():
                ref_data[table.name] = load_reference_csv(csv_path, table.name)
    else:
        # Single file — assume table name from stem
        table_name = reference_path.stem
        ref_data[table_name] = load_reference_csv(reference_path, table_name)

    return validate_fidelity(tables_data, ref_data, schema)


def _print_fidelity_rich(report: FidelityReport) -> None:
    """Print fidelity metrics as a Rich table."""
    if not report.metrics:
        return

    table = Table(title="Fidelity Validation")
    table.add_column("Metric", style="bold")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Score")
    table.add_column("Details")

    for m in report.metrics:
        score_str = f"{m.score:.3f}"
        if m.score >= 0.8:
            score_display = f"[green]{score_str}[/green]"
        elif m.score >= 0.5:
            score_display = f"[yellow]{score_str}[/yellow]"
        else:
            score_display = f"[red]{score_str}[/red]"
        table.add_row(m.metric, m.table, m.column, score_display, m.details)

    console.print(table)
    status = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    console.print(f"\nFidelity overall: {report.overall_score:.3f} {status}")


def _print_json(
    integrity_report: IntegrityReport,
    fidelity_report: FidelityReport | None = None,
) -> None:
    """Print JSON output for CI integration."""
    data: dict[str, Any] = {
        "integrity": {
            "passed": integrity_report.passed,
            "checks": [
                {
                    "check": c.check,
                    "table": c.table,
                    "column": c.column,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in integrity_report.checks
            ],
        },
    }
    if fidelity_report is not None:
        data["fidelity"] = {
            "passed": fidelity_report.passed,
            "overall_score": fidelity_report.overall_score,
            "metrics": [
                {
                    "metric": m.metric,
                    "table": m.table,
                    "column": m.column,
                    "score": m.score,
                    "details": m.details,
                }
                for m in fidelity_report.metrics
            ],
        }
    data["passed"] = integrity_report.passed and (
        fidelity_report.passed if fidelity_report else True
    )
    console.print(json.dumps(data, indent=2))
