"""``dbsprout diff`` command — report schema changes since the last snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import typer

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange, SchemaChangeType
    from dbsprout.schema.models import DatabaseSchema


def _resolve_source(
    db: str | None,
    file: str | None,
    output_dir: Path,
) -> tuple[Literal["db", "file"], str]:
    """Resolve the schema source from flags or ``dbsprout.toml`` config.

    Precedence: ``--db`` > ``--file`` > ``config.schema.source`` > exit 2.
    """
    from dbsprout.cli.console import console  # noqa: PLC0415

    if db is not None:
        return ("db", db)
    if file is not None:
        return ("file", file)

    from dbsprout.config import load_config  # noqa: PLC0415

    try:
        cfg = load_config(output_dir / "dbsprout.toml")
    except ValueError:
        console.print("[red]Error:[/red] No schema source. Provide --db or --file.")
        raise typer.Exit(code=2) from None

    src = cfg.schema_.source
    if not src:
        console.print("[red]Error:[/red] No schema source. Provide --db or --file.")
        raise typer.Exit(code=2)

    import sqlalchemy as sa  # noqa: PLC0415

    try:
        url = sa.engine.make_url(src)
    except sa.exc.ArgumentError:
        return ("file", src)
    if url.drivername:
        return ("db", src)
    return ("file", src)


def _summarize(changes: list[SchemaChange]) -> dict[str, int]:
    """Count changes by SchemaChangeType plus a total.

    Returns a dict with one key per SchemaChangeType variant (value used
    as the key), all initialized to 0, plus a "total" key.
    """
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    summary: dict[str, int] = {"total": len(changes)}
    for ct in SchemaChangeType:
        summary[ct.value] = sum(1 for c in changes if c.change_type == ct)
    return summary


_SUMMARY_LABELS: list[tuple[str, str]] = [
    ("table_added", "+tables"),
    ("table_removed", "-tables"),
    ("column_added", "+cols"),
    ("column_removed", "-cols"),
    ("column_type_changed", "~type"),
    ("column_nullability_changed", "~null"),
    ("column_default_changed", "~default"),
    ("foreign_key_added", "+fks"),
    ("foreign_key_removed", "-fks"),
    ("index_added", "+idx"),
    ("index_removed", "-idx"),
    ("enum_changed", "~enums"),
]


def _render_no_changes(
    old_schema: DatabaseSchema,
    safe_new_source: str,
    output_format: str,
) -> None:
    """Render the empty-diff result in either rich or json format."""
    from dbsprout.cli.console import console  # noqa: PLC0415

    if output_format == "json":
        import json  # noqa: PLC0415
        from datetime import datetime, timezone  # noqa: PLC0415

        payload = {
            "summary": {"total": 0},
            "old_snapshot": old_schema.schema_hash()[:8],
            "new_source": safe_new_source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "changes": [],
        }
        print(json.dumps(payload, indent=2))  # noqa: T201
    else:
        console.print("[green]✓ No changes detected.[/green]")


def _render_json(
    changes: list[SchemaChange],
    old_schema: DatabaseSchema,
    safe_new_source: str,
) -> None:
    """Render a non-empty diff result as JSON to stdout."""
    import json  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    payload = {
        "summary": _summarize(changes),
        "old_snapshot": old_schema.schema_hash()[:8],
        "new_source": safe_new_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "changes": [c.model_dump(mode="json") for c in changes],
    }
    print(json.dumps(payload, indent=2))  # noqa: T201


def _render_rich(
    changes: list[SchemaChange],
    old_schema: DatabaseSchema,
    safe_new_source: str,
) -> None:
    """Render a non-empty diff result to the Rich console."""
    _render_rich_header(changes, old_schema, safe_new_source)
    _render_rich_changes(changes)


def _render_rich_header(
    changes: list[SchemaChange],
    old_schema: DatabaseSchema,
    safe_new_source: str,
) -> None:
    """Build the summary Panel at the top of the report."""
    from datetime import datetime, timezone  # noqa: PLC0415

    from rich.panel import Panel  # noqa: PLC0415

    from dbsprout.cli.console import console  # noqa: PLC0415

    summary = _summarize(changes)
    old_hash = old_schema.schema_hash()[:8]
    old_tables = len(old_schema.tables)
    timestamp = datetime.now(timezone.utc).isoformat()

    parts: list[str] = []
    for key, label in _SUMMARY_LABELS:
        count = summary.get(key, 0)
        if count > 0:
            parts.append(f"{label}: {count}")
    summary_line = ", ".join(parts) if parts else "no changes"

    body = (
        f"old: {old_hash} ({old_tables} tables)\n"
        f"new: {safe_new_source}\n"
        f"summary: {summary_line}\n"
        f"generated: {timestamp}"
    )
    console.print(Panel(body, title="Schema Drift", border_style="cyan"))


def _render_rich_changes(changes: list[SchemaChange]) -> None:
    """Print grouped change sections under the header panel."""
    from dbsprout.cli.console import console  # noqa: PLC0415
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    groups: dict[str, list[SchemaChange]] = {
        "Tables": [],
        "Columns": [],
        "Foreign Keys": [],
        "Indexes": [],
        "Enums": [],
    }
    for c in changes:
        if c.change_type in {
            SchemaChangeType.TABLE_ADDED,
            SchemaChangeType.TABLE_REMOVED,
        }:
            groups["Tables"].append(c)
        elif c.change_type in {
            SchemaChangeType.COLUMN_ADDED,
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.COLUMN_TYPE_CHANGED,
            SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
            SchemaChangeType.COLUMN_DEFAULT_CHANGED,
        }:
            groups["Columns"].append(c)
        elif c.change_type in {
            SchemaChangeType.FOREIGN_KEY_ADDED,
            SchemaChangeType.FOREIGN_KEY_REMOVED,
        }:
            groups["Foreign Keys"].append(c)
        elif c.change_type in {
            SchemaChangeType.INDEX_ADDED,
            SchemaChangeType.INDEX_REMOVED,
        }:
            groups["Indexes"].append(c)
        elif c.change_type == SchemaChangeType.ENUM_CHANGED:
            groups["Enums"].append(c)

    for group_name, group_changes in groups.items():
        if not group_changes:
            continue
        console.print(f"\n[bold]{group_name}[/bold]")
        for c in group_changes:
            prefix = _change_prefix(c.change_type)
            line = _format_change_line(c)
            console.print(f"  {prefix} {line}")


def _change_prefix(ct: SchemaChangeType) -> str:
    """Return the colour-coded marker for a change type."""
    if ct.value.endswith("_added"):
        return "[green]+[/green]"
    if ct.value.endswith("_removed"):
        return "[red]-[/red]"
    return "[yellow]~[/yellow]"


def _format_change_line(c: SchemaChange) -> str:
    """Format a single change into a human-readable line."""
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    ct = c.change_type
    col_ref = f"{c.table_name}.{c.column_name}"
    formatters: dict[SchemaChangeType, str] = {
        SchemaChangeType.ENUM_CHANGED: f"enum {c.column_name or '<unknown>'}",
        SchemaChangeType.COLUMN_ADDED: col_ref,
        SchemaChangeType.COLUMN_REMOVED: col_ref,
        SchemaChangeType.COLUMN_TYPE_CHANGED: f"{col_ref}: {c.old_value} → {c.new_value}",
        SchemaChangeType.COLUMN_NULLABILITY_CHANGED: (
            f"{col_ref}: nullable {c.old_value} → {c.new_value}"
        ),
        SchemaChangeType.COLUMN_DEFAULT_CHANGED: (
            f"{col_ref}: default {c.old_value!r} → {c.new_value!r}"
        ),
        SchemaChangeType.FOREIGN_KEY_ADDED: f"{c.table_name} foreign key",
        SchemaChangeType.FOREIGN_KEY_REMOVED: f"{c.table_name} foreign key",
        SchemaChangeType.INDEX_ADDED: f"{c.table_name} index",
        SchemaChangeType.INDEX_REMOVED: f"{c.table_name} index",
        SchemaChangeType.TABLE_ADDED: c.table_name,
        SchemaChangeType.TABLE_REMOVED: c.table_name,
    }
    return formatters.get(ct, str(ct.value))


def diff_command(
    db: str | None,
    file: str | None,
    snapshot: str | None,
    output_format: str,
    output_dir: Path,
) -> None:
    """Report schema changes since the last snapshot."""
    from dbsprout.cli.console import console  # noqa: PLC0415

    if db is not None and file is not None:
        console.print("[red]Error:[/red] Provide only one of --db or --file.")
        raise typer.Exit(code=2)

    if output_format not in ("rich", "json"):
        console.print("[red]Error:[/red] Invalid format. Use 'rich' or 'json'.")
        raise typer.Exit(code=2)

    _source_kind, _source_value = _resolve_source(db, file, output_dir)

    from dbsprout.migrate.snapshot import SnapshotStore  # noqa: PLC0415

    store = SnapshotStore(base_dir=output_dir / ".dbsprout" / "snapshots")
    if snapshot is not None:
        old_schema = store.load_by_hash(snapshot)
        if old_schema is None:
            console.print(f"[red]Error:[/red] Snapshot not found: {snapshot}")
            raise typer.Exit(code=2)
    else:
        old_schema = store.load_latest()
        if old_schema is None:
            console.print("[red]Error:[/red] No snapshots found. Run 'dbsprout init' first.")
            raise typer.Exit(code=2)

    # ── New schema resolution (Task 5: db, Task 6: file) ────────────
    import importlib  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415

    # NOTE: ``dbsprout.schema.__init__`` re-exports the ``introspect`` function,
    # which shadows the submodule attribute lookup. Use ``importlib`` so
    # ``@patch("dbsprout.schema.introspect.introspect")`` still works in tests.
    introspect_module = importlib.import_module("dbsprout.schema.introspect")

    if _source_kind == "db":
        try:
            new_schema = introspect_module.introspect(_source_value)
            safe_new_source = sa.engine.make_url(_source_value).render_as_string(hide_password=True)
        except (ValueError, sa.exc.SQLAlchemyError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=2) from None
    else:  # _source_kind == "file"
        new_schema = _parse_schema_file(_source_value)
        safe_new_source = _source_value

    from dbsprout.migrate.differ import SchemaDiffer  # noqa: PLC0415

    assert new_schema is not None, "guaranteed by source resolution branch above"
    changes = SchemaDiffer.diff(old_schema, new_schema)

    if not changes:
        _render_no_changes(old_schema, safe_new_source, output_format)
        raise typer.Exit(code=0)

    if output_format == "json":
        _render_json(changes, old_schema, safe_new_source)
    else:
        _render_rich(changes, old_schema, safe_new_source)

    raise typer.Exit(code=1)  # drift detected → CI signal


def _parse_schema_file(file_path: str) -> DatabaseSchema:
    """Parse a schema file based on suffix. Raises ``typer.Exit(2)`` on error.

    Supports: ``.sql`` (DDL), ``.dbml``, ``.mermaid``/``.mmd``,
    ``.puml``/``.plantuml``/``.pu``, ``.prisma``. Unknown suffixes fall back to
    SQL DDL. Mirrors :func:`dbsprout.cli.commands.init._init_from_file` dispatch.
    """
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.cli.console import console  # noqa: PLC0415

    max_ddl_bytes = 10 * 1024 * 1024  # 10 MB cap — matches init_command

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(code=2)
    if path.stat().st_size > max_ddl_bytes:  # pragma: no cover — requires >10MB file
        console.print(f"[red]Error:[/red] File too large (>10 MB): {file_path}")
        raise typer.Exit(code=2)

    try:
        file_text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".dbml":
            from dbsprout.schema.parsers.dbml import parse_dbml  # noqa: PLC0415

            return parse_dbml(file_text, source_file=str(file_path))
        if suffix in (".mermaid", ".mmd"):
            from dbsprout.schema.parsers.mermaid import parse_mermaid  # noqa: PLC0415

            return parse_mermaid(file_text, source_file=str(file_path))
        if suffix in (".puml", ".plantuml", ".pu"):
            from dbsprout.schema.parsers.plantuml import parse_plantuml  # noqa: PLC0415

            return parse_plantuml(file_text, source_file=str(file_path))
        if suffix == ".prisma":
            from dbsprout.schema.parsers.prisma import parse_prisma  # noqa: PLC0415

            return parse_prisma(file_text, source_file=str(file_path))
        # Default: SQL DDL
        from dbsprout.schema.parsers.ddl import parse_ddl  # noqa: PLC0415

        return parse_ddl(file_text, source_file=str(file_path))
    except (ValueError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from None
