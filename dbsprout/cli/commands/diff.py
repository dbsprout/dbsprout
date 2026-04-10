"""``dbsprout diff`` command — report schema changes since the last snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import typer

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange
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
        raise typer.Exit(code=0)

    raise NotImplementedError


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
