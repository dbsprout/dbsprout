"""``dbsprout diff`` command — report schema changes since the last snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import typer

if TYPE_CHECKING:
    from pathlib import Path


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

    raise NotImplementedError
