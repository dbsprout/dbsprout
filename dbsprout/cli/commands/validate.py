"""``dbsprout validate`` command — integrity + fidelity validation with Rich report."""

from __future__ import annotations

import sys
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
    from dbsprout.quality.detection import DetectionReport
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
    detection: bool = False,
    output: Path | None = None,
    compact: bool = False,
) -> None:
    """Validate integrity of generated seed data."""
    # Validate --output requires --format json
    if output is not None and output_format != "json":
        console.print("[red]Error:[/red] --output requires --format json.")
        raise typer.Exit(code=1)

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

    # Validate detection (optional — requires --reference-data and [stats] extra)
    detection_report: DetectionReport | None = None
    if detection:
        if reference_data is None:
            console.print("[red]Error:[/red] --detection requires --reference-data.")
            raise typer.Exit(code=1)
        detection_report = _run_detection(result.tables_data, reference_data, schema, seed)

    # Output
    if output_format == "json":
        _print_json(
            integrity_report,
            fidelity_report,
            detection_report,
            schema=schema,
            tables_data=result.tables_data,
            engine=engine,
            seed=seed,
            output=output,
            compact=compact,
        )
    else:
        _print_rich(integrity_report)
        if fidelity_report is not None:
            _print_fidelity_rich(fidelity_report)
        if detection_report is not None:
            _print_detection_rich(detection_report)

    if not integrity_report.passed:
        raise typer.Exit(code=1)
    if fidelity_report is not None and not fidelity_report.passed:
        raise typer.Exit(code=1)
    if detection_report is not None and not detection_report.passed:
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


def _load_reference_data(
    reference_path: Path,
    schema: DatabaseSchema,
) -> dict[str, list[dict[str, Any]]] | None:
    """Load reference CSV data for fidelity/detection comparison.

    Returns None if the reference path does not exist (caller handles error).
    """
    from dbsprout.quality.fidelity import load_reference_csv  # noqa: PLC0415

    if not reference_path.exists():
        console.print(f"[red]Error:[/red] Reference data not found: {reference_path}")
        return None

    ref_data: dict[str, list[dict[str, Any]]] = {}
    if reference_path.is_dir():
        for table in schema.tables:
            csv_path = (reference_path / f"{table.name}.csv").resolve()
            if not csv_path.is_relative_to(reference_path.resolve()):
                continue
            if csv_path.exists():
                ref_data[table.name] = load_reference_csv(csv_path, table.name)
    else:
        table_name = reference_path.stem
        ref_data[table_name] = load_reference_csv(reference_path, table_name)

    return ref_data


def _run_fidelity(
    tables_data: dict[str, list[dict[str, Any]]],
    reference_path: Path,
    schema: DatabaseSchema,
) -> FidelityReport:
    """Load reference data and compute fidelity metrics."""
    from dbsprout.quality.fidelity import (  # noqa: PLC0415
        FidelityReport,
        validate_fidelity,
    )

    ref_data = _load_reference_data(reference_path, schema)
    if ref_data is None:
        return FidelityReport()

    return validate_fidelity(tables_data, ref_data, schema)


def _run_detection(
    tables_data: dict[str, list[dict[str, Any]]],
    reference_path: Path,
    schema: DatabaseSchema,
    seed: int = 42,
) -> DetectionReport:
    """Load reference data and compute detection (C2ST) metrics."""
    from dbsprout.quality.detection import (  # noqa: PLC0415
        DetectionReport,
        validate_detection,
    )

    ref_data = _load_reference_data(reference_path, schema)
    if ref_data is None:
        return DetectionReport()

    return validate_detection(tables_data, ref_data, schema, seed=seed)


def _print_detection_rich(report: DetectionReport) -> None:
    """Print detection metrics as a Rich table."""
    if not report.metrics:
        return

    table = Table(title="Detection Validation (C2ST)")
    table.add_column("Metric", style="bold")
    table.add_column("Table")
    table.add_column("Accuracy")
    table.add_column("Details")

    for m in report.metrics:
        acc_str = f"{m.accuracy:.3f}"
        if m.accuracy <= 0.55:
            acc_display = f"[green]{acc_str}[/green]"
        elif m.accuracy <= 0.7:
            acc_display = f"[yellow]{acc_str}[/yellow]"
        else:
            acc_display = f"[red]{acc_str}[/red]"
        table.add_row(m.metric, m.table, acc_display, m.details)

    console.print(table)
    status = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    console.print(f"\nDetection overall: {report.overall_score:.3f} {status}")


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


def _print_json(  # noqa: PLR0913
    integrity_report: IntegrityReport,
    fidelity_report: FidelityReport | None = None,
    detection_report: DetectionReport | None = None,
    *,
    schema: DatabaseSchema | None = None,
    tables_data: dict[str, list[dict[str, Any]]] | None = None,
    engine: str = "heuristic",
    seed: int = 42,
    output: Path | None = None,
    compact: bool = False,
) -> None:
    """Serialize QualityReport as JSON and write to stdout or file."""
    from dbsprout.quality.report import QualityReport  # noqa: PLC0415

    row_counts = {t: len(rows) for t, rows in (tables_data or {}).items()}
    schema_hash = schema.schema_hash() if schema else ""

    report = QualityReport.from_reports(
        integrity=integrity_report,
        schema_hash=schema_hash,
        row_counts=row_counts,
        engine=engine,
        seed=seed,
        fidelity=fidelity_report,
        detection=detection_report,
    )

    json_str = report.model_dump_json() if compact else report.model_dump_json(indent=2)

    if output is not None:
        try:
            output.write_text(json_str + "\n", encoding="utf-8")
            typer.echo(f"Report written to {output}", err=True)
        except OSError as exc:
            console.print(f"[red]Error:[/red] Cannot write to {output}: {exc}")
            raise typer.Exit(code=1) from exc
    else:
        sys.stdout.write(json_str + "\n")
