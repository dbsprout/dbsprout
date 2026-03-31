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


def init_command(
    db: str | None = typer.Option(None, "--db", help="Database connection URL"),
    file: str | None = typer.Option(None, "--file", "-f", help="SQL DDL file path (.sql)"),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", help="Output directory for config and snapshots"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without writing files"),
) -> None:
    """Introspect a database schema and generate configuration."""
    import sqlalchemy as sa  # noqa: PLC0415

    # ── Validate arguments ───────────────────────────────────────────────
    if file is not None and db is not None:
        console.print("[red]Error:[/red] Provide --db or --file, not both.")
        raise typer.Exit(code=1)

    if file is not None:
        return _init_from_file(file, output_dir, dry_run)

    if db is None:
        console.print("[red]Error:[/red] Provide --db <connection_string> or --file <path.sql>.")
        raise typer.Exit(code=1)

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
        snapshot_path_str = f".dbsprout/snapshots/{schema.schema_hash()}.json"

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
    """Write schema snapshot JSON."""
    if not schema.tables:
        return

    if dry_run:
        return

    try:
        snapshot_dir = output_dir / ".dbsprout" / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        schema_hash = schema.schema_hash()
        snapshot_path = snapshot_dir / f"{schema_hash}.json"
        snapshot_path.write_text(schema.model_dump_json(indent=2))
        console.print(f"[green]Wrote[/green] {snapshot_path}")
    except OSError as exc:
        console.print(f"[red]Error writing snapshot:[/red] {exc}")
        raise typer.Exit(code=1) from None


# ── File-based init ──────────────────────────────────────────────────────


def _init_from_file(file_path: str, output_dir: Path, dry_run: bool) -> None:
    """Initialize from a DDL file instead of a live database."""
    from dbsprout.schema.parsers.ddl import parse_ddl  # noqa: PLC0415

    max_ddl_bytes = 10 * 1024 * 1024  # 10 MB

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(code=1)

    if path.stat().st_size > max_ddl_bytes:  # pragma: no cover — requires >10MB file
        console.print(f"[red]Error:[/red] File too large (>10 MB): {file_path}")
        raise typer.Exit(code=1)

    try:
        file_text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".dbml":
            from dbsprout.schema.parsers.dbml import parse_dbml  # noqa: PLC0415

            schema = parse_dbml(file_text, source_file=str(file_path))
        else:
            schema = parse_ddl(file_text, source_file=str(file_path))
    except (ValueError, OSError) as exc:
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
