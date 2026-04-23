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
    from dbsprout.cli.sources import SchemaSource
    from dbsprout.migrate.snapshot import SnapshotStore
    from dbsprout.migrate.update_models import UpdateResult
    from dbsprout.output.models import InsertResult
    from dbsprout.quality.integrity import IntegrityReport

console = Console()

# Snapshot directory is always project-root-relative so that ``init``, ``diff``,
# and ``generate --incremental`` agree on where snapshots live, regardless of
# ``--output-dir`` (which is the seeds directory for ``generate``).
_SNAPSHOT_DIR = Path(".dbsprout") / "snapshots"


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
    upsert: bool = False,
    insert_method: str = "auto",
    file: str | None = None,
    incremental: bool = False,
    snapshot: str | None = None,
) -> None:
    """Generate seed data from a schema snapshot."""
    # Validate insert_method
    valid_methods = {"auto", "copy", "load_data", "batch"}
    if insert_method not in valid_methods:
        console.print(
            f"[red]Error:[/red] Invalid --insert-method: {insert_method}. "
            f"Use: {', '.join(sorted(valid_methods))}"
        )
        raise typer.Exit(code=1)

    if incremental:
        _run_incremental(
            db=target_db,
            file=file,
            snapshot=snapshot,
            config_path=config_path,
            seed=seed,
            default_rows=rows,
            engine=engine,
            output_format=output_format,
            output_dir=output_dir,
            dialect=dialect,
            upsert=upsert,
            insert_method=insert_method,
        )
        return

    # Load schema (with snapshot fallback)
    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415

    snap_store = SnapshotStore()
    schema = _resolve_schema(schema_snapshot, snap_store)
    if schema is None:
        console.print("[red]Error:[/red] No schema snapshot found.")
        console.print("Run [bold]dbsprout init[/bold] first, or use --schema-snapshot.")
        raise typer.Exit(code=1)

    # Auto-snapshot: save if no snapshot exists or schema changed
    try:
        latest = snap_store.load_latest()
        if latest is None or latest.schema_hash() != schema.schema_hash():
            snap_store.save(schema)
    except OSError as exc:
        console.print(f"[yellow]Warning: could not save snapshot:[/yellow] {exc}")

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
        result,
        schema,
        result.insertion_order,
        output_dir,
        output_format,
        dialect,
        target_db,
        upsert,
        insert_method,
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


def _resolve_schema(
    explicit: Path | None,
    store: SnapshotStore,
) -> DatabaseSchema | None:
    """Load schema from explicit path, default file, or latest snapshot."""
    import json  # noqa: PLC0415

    path = _resolve_schema_path(explicit)
    if path is not None and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        schema_data = raw.get("schema", raw)
        return DatabaseSchema.model_validate(schema_data)
    return store.load_latest()


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
    upsert: bool = False,
    insert_method: str = "auto",
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
            upsert=upsert,
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
        _run_direct_insert(result, schema, insertion_order, target_db, insert_method)
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
    if lower.startswith("sqlite"):
        return "sqlite"
    if lower.startswith("mssql"):
        return "mssql"
    return lower.split("://")[0].split("+")[0] if "://" in lower else "unknown"


def _run_direct_insert(
    result: GenerateResult,
    schema: DatabaseSchema,
    insertion_order: list[str],
    target_db: str,
    insert_method: str = "auto",
) -> None:
    """Dispatch direct insertion to the appropriate dialect writer.

    Parameters
    ----------
    result:
        Generated data from the orchestrator.
    schema:
        Unified database schema.
    insertion_order:
        Table names in FK-safe insertion order.
    target_db:
        SQLAlchemy connection URL.
    insert_method:
        One of ``"auto"``, ``"copy"``, ``"load_data"``, ``"batch"``.
        ``"auto"`` selects the fastest available writer for the dialect.
    """
    dialect = _detect_direct_dialect(target_db)

    # Validate method/dialect compatibility
    if insert_method == "copy" and dialect != "postgresql":
        console.print("[red]Error:[/red] --insert-method copy is only available for PostgreSQL.")
        raise typer.Exit(code=1)
    if insert_method == "load_data" and dialect != "mysql":
        console.print("[red]Error:[/red] --insert-method load_data is only available for MySQL.")
        raise typer.Exit(code=1)

    method_name: str
    insert_result: InsertResult

    if insert_method == "batch":
        from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

        method_name = f"SQLAlchemy batch INSERT ({dialect})"
        insert_result = SaBatchWriter().write(
            result.tables_data, schema, insertion_order, target_db
        )

    elif insert_method == "copy" or (insert_method == "auto" and dialect == "postgresql"):
        try:
            import psycopg  # type: ignore[import-not-found,unused-ignore]  # noqa: F401, PLC0415

            from dbsprout.output.pg_copy import PgCopyWriter  # noqa: PLC0415

            method_name = (
                "PostgreSQL COPY (auto-detected)"
                if insert_method == "auto"
                else "PostgreSQL COPY (user-selected)"
            )
            insert_result = PgCopyWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] psycopg not installed, falling back to batch INSERT."
            )
            from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

            method_name = "SQLAlchemy batch INSERT (fallback)"
            insert_result = SaBatchWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )

    elif insert_method == "load_data" or (insert_method == "auto" and dialect == "mysql"):
        try:
            import pymysql  # type: ignore[import-untyped]  # noqa: F401, PLC0415

            from dbsprout.output.mysql_load_data import (  # noqa: PLC0415
                MysqlLoadDataWriter,
            )

            method_name = (
                "MySQL LOAD DATA (auto-detected)"
                if insert_method == "auto"
                else "MySQL LOAD DATA (user-selected)"
            )
            insert_result = MysqlLoadDataWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] pymysql not installed, falling back to batch INSERT."
            )
            from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

            method_name = "SQLAlchemy batch INSERT (fallback)"
            insert_result = SaBatchWriter().write(
                result.tables_data, schema, insertion_order, target_db
            )

    else:
        # sqlite, mssql, unknown -- all use SaBatchWriter
        from dbsprout.output.sa_batch import SaBatchWriter  # noqa: PLC0415

        method_name = f"SQLAlchemy batch INSERT ({dialect})"
        insert_result = SaBatchWriter().write(
            result.tables_data, schema, insertion_order, target_db
        )

    console.print(f"[blue]Insert method:[/blue] {method_name}")
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


def _persist_result(  # noqa: PLR0913
    result: GenerateResult,
    schema: DatabaseSchema,
    output_dir: Path,
    output_format: str,
    dialect: str,
    upsert: bool,
    insert_method: str,
    target_db: str | None = None,
) -> None:
    """Write output, validate integrity, print summary.

    Raises ``typer.Exit(1)`` if integrity validation fails.
    """
    from dbsprout.quality.integrity import validate_integrity  # noqa: PLC0415

    if result.total_tables == 0:  # pragma: no cover — pre-checked by callers
        console.print("[yellow]No tables to generate.[/yellow]")
        raise typer.Exit(code=0)

    _write_output(
        result,
        schema,
        result.insertion_order,
        output_dir,
        output_format,
        dialect,
        target_db,
        upsert,
        insert_method,
    )
    report = validate_integrity(result.tables_data, schema)
    _print_validation(report)
    _print_summary(result, output_dir, output_format)

    if not report.passed:  # pragma: no cover — heuristic engine produces valid data
        console.print("[red]Integrity validation FAILED.[/red]")
        raise typer.Exit(code=1)


def _insertion_order_for(schema: DatabaseSchema) -> list[str]:
    """Compute FK-safe insertion order for a schema."""
    from graphlib import CycleError  # noqa: PLC0415

    from dbsprout.schema.graph import FKGraph, resolve_cycles  # noqa: PLC0415

    try:
        graph = FKGraph.from_schema(schema)
    except CycleError:  # pragma: no cover — covered by orchestrator tests
        graph = resolve_cycles(schema).graph

    out: list[str] = []
    for batch in graph.insertion_order:
        out.extend(sorted(batch))
    return out


def _print_actions_applied(result: UpdateResult, unchanged_tables: list[str]) -> None:
    """Render the list of applied actions as a Rich table, plus unchanged tables."""
    table = Table(title="Incremental Updates")
    table.add_column("Action", style="bold")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Description")

    for action in result.actions_applied:
        table.add_row(
            action.action.value,
            action.change.table_name,
            action.change.column_name or "",
            action.description,
        )
    console.print(table)
    console.print(
        f"[blue]Rows modified:[/blue] {result.rows_modified} | "
        f"added: {result.rows_added} | removed: {result.rows_removed}"
    )
    if unchanged_tables:
        console.print(
            f"[dim]Unchanged tables (skipped): {', '.join(sorted(unchanged_tables))}[/dim]"
        )


def _load_schema_from_source(source: SchemaSource) -> DatabaseSchema:
    """Introspect DB or parse file, returning the new ``DatabaseSchema``."""
    from dbsprout.schema.introspect import introspect  # noqa: PLC0415
    from dbsprout.schema.parsers import parse_schema_file  # noqa: PLC0415

    if source.kind == "db":
        return introspect(source.raw_value)
    return parse_schema_file(Path(source.raw_value))


def _run_full_gen_from_source(  # noqa: PLR0913
    source: SchemaSource,
    config_path: Path | None,
    seed: int,
    default_rows: int,
    engine: str,
    output_format: str,
    output_dir: Path,
    dialect: str,
    upsert: bool,
    insert_method: str,
) -> None:
    """Full-gen path used as a fallback when no prior snapshot exists."""
    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415

    schema = _load_schema_from_source(source)

    cfg_path = config_path or Path("dbsprout.toml")
    config = DBSproutConfig.from_toml(cfg_path if cfg_path.exists() else None)

    result = orchestrate(schema, config, seed=seed, default_rows=default_rows, engine=engine)
    _persist_result(result, schema, output_dir, output_format, dialect, upsert, insert_method)

    try:
        SnapshotStore(base_dir=_SNAPSHOT_DIR).save(schema)
    except (ValueError, OSError) as exc:  # pragma: no cover — defensive
        console.print(f"[yellow]Warning: could not save snapshot:[/yellow] {exc}")


def _run_incremental(  # noqa: PLR0913
    db: str | None,
    file: str | None,
    snapshot: str | None,
    config_path: Path | None,
    seed: int,
    default_rows: int,
    engine: str,
    output_format: str,
    output_dir: Path,
    dialect: str,
    upsert: bool,
    insert_method: str,
) -> None:
    """Execute the --incremental path: diff against a prior snapshot and apply updates."""
    from dbsprout.cli.sources import resolve_schema_source  # noqa: PLC0415
    from dbsprout.migrate.differ import SchemaDiffer  # noqa: PLC0415
    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415
    from dbsprout.migrate.updater import IncrementalUpdater  # noqa: PLC0415

    if db is not None and file is not None:
        console.print("[red]Error:[/red] Provide only one of --db or --file.")
        raise typer.Exit(code=2)

    source = resolve_schema_source(db, file, output_dir)
    store = SnapshotStore(base_dir=_SNAPSHOT_DIR)
    old_schema = store.load_by_hash(snapshot) if snapshot is not None else store.load_latest()

    if old_schema is None:
        console.print(
            "[yellow]Warning:[/yellow] No prior snapshot found — falling back to full generation."
        )
        _run_full_gen_from_source(
            source=source,
            config_path=config_path,
            seed=seed,
            default_rows=default_rows,
            engine=engine,
            output_format=output_format,
            output_dir=output_dir,
            dialect=dialect,
            upsert=upsert,
            insert_method=insert_method,
        )
        return

    new_schema = _load_schema_from_source(source)

    cfg_path = config_path or Path("dbsprout.toml")
    config = DBSproutConfig.from_toml(cfg_path if cfg_path.exists() else None)

    existing = orchestrate(old_schema, config, seed=seed, default_rows=default_rows, engine=engine)

    changes = SchemaDiffer.diff(old_schema, new_schema)
    if not changes:
        console.print("[green]No schema changes detected — output unchanged.[/green]")
        _persist_result(
            existing,
            new_schema,
            output_dir,
            output_format,
            dialect,
            upsert,
            insert_method,
        )
        try:
            store.save(new_schema)
        except (ValueError, OSError) as exc:  # pragma: no cover — defensive
            console.print(f"[yellow]Warning: could not save snapshot:[/yellow] {exc}")
        return

    import time  # noqa: PLC0415

    updater = IncrementalUpdater(new_schema, config, seed=seed)
    apply_start = time.monotonic()
    update_result = updater.update(changes, existing.tables_data)
    apply_duration = time.monotonic() - apply_start

    insertion_order = _insertion_order_for(new_schema)
    applied_result = GenerateResult(
        tables_data=update_result.tables_data,
        insertion_order=[t for t in insertion_order if t in update_result.tables_data],
        total_rows=sum(len(rows) for rows in update_result.tables_data.values()),
        total_tables=len(update_result.tables_data),
        duration_seconds=round(apply_duration, 3),
    )

    changed_tables = {a.change.table_name for a in update_result.actions_applied}
    unchanged_tables = [t for t in applied_result.tables_data if t not in changed_tables]

    _persist_result(
        applied_result,
        new_schema,
        output_dir,
        output_format,
        dialect,
        upsert,
        insert_method,
    )
    _print_actions_applied(update_result, unchanged_tables)
    try:
        store.save(new_schema)
    except (ValueError, OSError) as exc:  # pragma: no cover — defensive
        console.print(f"[yellow]Warning: could not save snapshot:[/yellow] {exc}")
