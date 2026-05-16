"""``dbsprout init`` command — introspect a database and generate config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.panel import Panel
from rich.table import Table

from dbsprout.cli.console import console
from dbsprout.schema.graph import ResolvedGraph, UnresolvableCycleError, resolve_cycles
from dbsprout.schema.introspect import introspect

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema


def init_command(  # noqa: PLR0913
    db: str | None = typer.Option(None, "--db", help="Database connection URL"),
    file: str | None = typer.Option(None, "--file", "-f", help="SQL DDL file path (.sql)"),
    django: bool = False,
    django_apps: str | None = None,
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", help="Output directory for config and snapshots"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing files"),
) -> None:
    """Introspect a database schema and generate configuration."""
    # ── Validate arguments ───────────────────────────────────────────────
    sources = sum([db is not None, file is not None, django])
    if sources > 1:
        console.print("[red]Error:[/red] Provide only one of --db, --file, or --django.")
        raise typer.Exit(code=1)

    if sources == 0:
        console.print("[red]Error:[/red] Provide --db <url>, --file <path>, or --django.")
        raise typer.Exit(code=1)

    if django:
        app_labels = [a.strip() for a in django_apps.split(",")] if django_apps else None
        return _init_from_django(app_labels, output_dir, dry_run)

    if file is not None:
        return _init_from_file(file, output_dir, dry_run)

    assert db is not None  # validated above: exactly one source is set

    from dbsprout.schema.parsers.mongodb import is_mongo_url  # noqa: PLC0415

    if is_mongo_url(db):
        return _init_from_mongo(db, output_dir, dry_run)

    import sqlalchemy as sa  # noqa: PLC0415

    # ── Sanitize URL for display/storage ────────────────────────────────
    try:
        safe_url = sa.engine.make_url(db).render_as_string(hide_password=True)
    except Exception:
        safe_url = "<invalid URL>"

    # ── Introspect ───────────────────────────────────────────────────────
    try:
        schema = introspect(db)
    except (ValueError, sa.exc.SQLAlchemyError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if not schema.tables:
        console.print("[yellow]Warning:[/yellow] No tables found in database.")
        _write_config(schema, safe_url, output_dir, dry_run)
        raise typer.Exit(code=0)

    # ── Resolve cycles ───────────────────────────────────────────────────
    try:
        resolved = resolve_cycles(schema)
    except UnresolvableCycleError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    # ── Display schema summary ───────────────────────────────────────────
    _display_schema_table(schema)
    _display_insertion_order(resolved)
    _display_cycle_warnings(resolved)
    _display_self_refs(resolved)

    # ── Write files ──────────────────────────────────────────────────────
    _write_config(schema, safe_url, output_dir, dry_run)
    _write_snapshot(schema, output_dir, dry_run)

    console.print("\n[green bold]Done![/green bold] Run `dbsprout generate` to create seed data.")
    return None


# ── Display helpers ──────────────────────────────────────────────────────


def _display_schema_table(schema: DatabaseSchema) -> None:
    """Print a Rich table with schema summary."""
    table = Table(title="Schema Summary", show_lines=False)
    table.add_column("Table", style="cyan")
    table.add_column("Columns", justify="right")
    table.add_column("FKs", justify="right")
    table.add_column("Primary Key", style="dim")

    for t in sorted(schema.tables, key=lambda x: x.name):
        pk_str = ", ".join(t.primary_key) if t.primary_key else "—"
        table.add_row(t.name, str(len(t.columns)), str(len(t.foreign_keys)), pk_str)

    console.print(table)


def _display_insertion_order(resolved: ResolvedGraph) -> None:
    """Print the batched insertion order."""
    if not resolved.graph.insertion_order:
        return

    lines: list[str] = []
    for i, batch in enumerate(resolved.graph.insertion_order, 1):
        tables = ", ".join(batch)
        lines.append(f"  [bold]{i}.[/bold] {tables}")

    panel = Panel("\n".join(lines), title="Insertion Order", border_style="blue")
    console.print(panel)


def _display_cycle_warnings(resolved: ResolvedGraph) -> None:
    """Print warnings about deferred FKs from cycle breaking."""
    if not resolved.deferred_fks:
        return

    lines: list[str] = []
    for dfk in resolved.deferred_fks:
        cols = ", ".join(dfk.foreign_key.columns)
        lines.append(
            f"  [yellow]{dfk.source_table}.{cols}[/yellow] → "
            f"{dfk.foreign_key.ref_table} (deferred for two-pass insertion)"
        )

    panel = Panel(
        "\n".join(lines),
        title="Cycle Resolution — Deferred FKs",
        border_style="yellow",
    )
    console.print(panel)


def _display_self_refs(resolved: ResolvedGraph) -> None:
    """Print advisory about self-referencing FKs."""
    self_refs = resolved.graph.self_referencing
    if not self_refs:
        return
    tables = ", ".join(sorted(self_refs.keys()))
    console.print(f"[dim]Self-referencing FKs:[/dim] {tables}")


# ── File writing helpers ─────────────────────────────────────────────────


def _write_config(
    schema: DatabaseSchema,
    db_url: str,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Generate and write dbsprout.toml."""
    dialect = schema.dialect or "unknown"
    table_names = schema.table_names()

    snapshot_path_str = ""
    if schema.tables:
        snapshot_path_str = ".dbsprout/snapshots/"

    toml_content = f"""\
# DBSprout configuration — generated by `dbsprout init`

[schema]
dialect = "{dialect}"
source = "{db_url}"
snapshot = "{snapshot_path_str}"

[generation]
default_rows = 100
seed = 42
output_format = "sql"
output_dir = "./seeds"
"""

    if table_names:
        toml_content += "\n# Per-table overrides (uncomment and customize):\n"
        for name in table_names:
            # Quote table names that contain special TOML characters
            safe_name = f'"{name}"' if any(c in name for c in '.]["\n') else name
            toml_content += f"# [tables.{safe_name}]\n# rows = 100\n"

    if dry_run:
        console.print("[dim]Dry run — skipping file writes[/dim]")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        toml_path = output_dir / "dbsprout.toml"
        if toml_path.exists():
            console.print("[yellow]Warning:[/yellow] Overwriting existing dbsprout.toml")
        toml_path.write_text(toml_content)
        console.print(f"[green]Wrote[/green] {toml_path}")
    except OSError as exc:
        console.print(f"[red]Error writing config:[/red] {exc}")
        raise typer.Exit(code=1) from None


def _write_snapshot(
    schema: DatabaseSchema,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Write schema snapshot JSON via SnapshotStore."""
    if not schema.tables or dry_run:
        return

    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415

    try:
        snap_dir = output_dir / ".dbsprout" / "snapshots"
        store = SnapshotStore(base_dir=snap_dir)
        info = store.save(schema)
        console.print(f"[green]Wrote[/green] {info.path}")
    except OSError as exc:
        console.print(f"[red]Error writing snapshot:[/red] {exc}")
        raise typer.Exit(code=1) from None


# ── File-based init ──────────────────────────────────────────────────────


def _init_from_file(file_path: str, output_dir: Path, dry_run: bool) -> None:
    """Initialize from a schema file instead of a live database."""
    from dbsprout.schema.parsers import parse_schema_file  # noqa: PLC0415

    try:
        schema = parse_schema_file(Path(file_path))
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if not schema.tables:
        console.print("[yellow]Warning:[/yellow] No tables found in DDL file.")
        _write_config(schema, file_path, output_dir, dry_run)
        raise typer.Exit(code=0)

    try:
        resolved = resolve_cycles(schema)
    except UnresolvableCycleError as exc:  # pragma: no cover — tested via --db path
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    _display_schema_table(schema)
    _display_insertion_order(resolved)
    _display_cycle_warnings(resolved)
    _display_self_refs(resolved)

    _write_config(schema, file_path, output_dir, dry_run)
    _write_snapshot(schema, output_dir, dry_run)

    console.print("\n[green bold]Done![/green bold] Run `dbsprout generate` to create seed data.")


# ── Django-based init ───────────────────────────────────────────────────


def _init_from_django(
    app_labels: list[str] | None,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Initialize from Django model introspection."""
    from dbsprout.schema.parsers.django import parse_django_models  # noqa: PLC0415

    try:
        schema = parse_django_models(app_labels=app_labels)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if not schema.tables:
        console.print("[yellow]Warning:[/yellow] No Django models found.")
        raise typer.Exit(code=0)

    try:
        resolved = resolve_cycles(schema)
    except UnresolvableCycleError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    _display_schema_table(schema)
    _display_insertion_order(resolved)
    _display_cycle_warnings(resolved)
    _display_self_refs(resolved)

    _write_config(schema, "django", output_dir, dry_run)
    _write_snapshot(schema, output_dir, dry_run)

    console.print("\n[green bold]Done![/green bold] Run `dbsprout generate` to create seed data.")


# ── MongoDB-based init ───────────────────────────────────────────────────


def _init_from_mongo(db: str, output_dir: Path, dry_run: bool) -> None:
    """Initialize from MongoDB document-sampling schema inference."""
    from dbsprout.schema.parsers.mongodb import infer_mongo_schema  # noqa: PLC0415

    try:
        schema = infer_mongo_schema(db)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    safe_url = schema.source or "mongodb"

    if not schema.tables:
        console.print("[yellow]Warning:[/yellow] No collections found in database.")
        _write_config(schema, safe_url, output_dir, dry_run)
        raise typer.Exit(code=0)

    try:
        resolved = resolve_cycles(schema)
    except UnresolvableCycleError as exc:  # pragma: no cover — tested via --db path
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    _display_schema_table(schema)
    _display_insertion_order(resolved)
    _display_cycle_warnings(resolved)
    _display_self_refs(resolved)

    _write_config(schema, safe_url, output_dir, dry_run)
    _write_snapshot(schema, output_dir, dry_run)

    console.print("\n[green bold]Done![/green bold] Run `dbsprout generate` to create seed data.")
