"""``dbsprout validate`` command — integrity + fidelity validation with Rich report."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from dbsprout.config.models import DBSproutConfig
from dbsprout.core import service
from dbsprout.schema.models import DatabaseSchema

if TYPE_CHECKING:
    from dbsprout.quality.detection import DetectionReport
    from dbsprout.quality.fidelity import FidelityReport
    from dbsprout.quality.integrity import IntegrityReport

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

    # --detection requires --reference-data (checked before any generation).
    if detection and reference_data is None:
        console.print("[red]Error:[/red] --detection requires --reference-data.")
        raise typer.Exit(code=1)

    # Load reference rows (CLI owns file IO + the not-found message); the core
    # service computes fidelity/detection from the already-loaded dict.
    ref_rows: dict[str, list[dict[str, Any]]] | None = None
    if reference_data is not None:
        ref_rows = _load_reference_data(reference_data, schema)

    # Generate + validate through the core service facade (single seam).
    outcome = service.run_validation(
        schema,
        config,
        seed=seed,
        default_rows=rows,
        engine=engine,
        reference_data=ref_rows,
        detection=detection,
    )
    integrity_report = outcome.integrity
    fidelity_report: FidelityReport | None = outcome.fidelity
    detection_report: DetectionReport | None = outcome.detection

    # Parity: when --reference-data was requested but the path was missing,
    # the pre-S-106 CLI still emitted an *empty* (passed) report rather than
    # omitting it. Reconstruct that so the JSON/Rich output is unchanged.
    if reference_data is not None and ref_rows is None:
        fidelity_report = _empty_fidelity()
        if detection:
            detection_report = _empty_detection()

    _emit_reports(
        integrity_report,
        fidelity_report,
        detection_report,
        schema=schema,
        tables_data=outcome.tables_data,
        output_format=output_format,
        engine=engine,
        seed=seed,
        output=output,
        compact=compact,
    )


def _emit_reports(  # noqa: PLR0913
    integrity_report: IntegrityReport,
    fidelity_report: FidelityReport | None,
    detection_report: DetectionReport | None,
    *,
    schema: DatabaseSchema,
    tables_data: dict[str, list[dict[str, Any]]],
    output_format: str,
    engine: str,
    seed: int,
    output: Path | None,
    compact: bool,
) -> None:
    """Render the reports (JSON or Rich) and raise the appropriate exit code.

    Exit code 1 if any computed report failed; otherwise returns normally.
    """
    if output_format == "json":
        _print_json(
            integrity_report,
            fidelity_report,
            detection_report,
            schema=schema,
            tables_data=tables_data,
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

    reports = (integrity_report, fidelity_report, detection_report)
    if any(r is not None and not r.passed for r in reports):
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
        table.add_row(
            escape(check.check),
            escape(check.table),
            escape(check.column),
            status,
            escape(check.details),
        )

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
                ref_data[table.name] = load_reference_csv(csv_path)
    else:
        table_name = reference_path.stem
        ref_data[table_name] = load_reference_csv(reference_path)

    return ref_data


def _empty_fidelity() -> FidelityReport:
    """An empty (passed) fidelity report — emitted when reference data is missing."""
    from dbsprout.quality.fidelity import FidelityReport  # noqa: PLC0415

    return FidelityReport()


def _empty_detection() -> DetectionReport:
    """An empty (passed) detection report — emitted when reference data is missing."""
    from dbsprout.quality.detection import DetectionReport  # noqa: PLC0415

    return DetectionReport()


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
        table.add_row(escape(m.metric), escape(m.table), acc_display, escape(m.details))

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
        table.add_row(
            escape(m.metric),
            escape(m.table),
            escape(m.column),
            score_display,
            escape(m.details),
        )

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
        sys.stdout.flush()
