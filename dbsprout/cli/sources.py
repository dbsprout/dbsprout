"""Shared CLI helper — resolve schema source from flags or config.

Used by ``dbsprout diff`` and ``dbsprout generate --incremental`` so the
two commands always accept the same precedence (``--db`` > ``--file`` >
``config.schema.source``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import typer

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SchemaSource:
    """Resolved schema source."""

    kind: Literal["db", "file"]
    raw_value: str
    display_value: str  # password-sanitized for DB URLs, path as-is for files


def resolve_schema_source(
    db: str | None,
    file: str | None,
    output_dir: Path,
) -> SchemaSource:
    """Resolve the schema source from flags or ``dbsprout.toml``.

    Precedence: ``--db`` > ``--file`` > ``config.schema.source``.
    Raises ``typer.Exit(code=2)`` if none is set.
    """
    from dbsprout.cli.console import console  # noqa: PLC0415

    if db is not None:
        return _make_db_source(db)
    if file is not None:
        return SchemaSource(kind="file", raw_value=file, display_value=file)

    from pydantic import ValidationError  # noqa: PLC0415

    from dbsprout.config import load_config  # noqa: PLC0415

    try:
        cfg = load_config(output_dir / "dbsprout.toml")
    except (ValueError, OSError, ValidationError):
        console.print("[red]Error:[/red] No schema source. Provide --db or --file.")
        raise typer.Exit(code=2) from None

    src = cfg.schema_.source
    if not src:
        console.print("[red]Error:[/red] No schema source. Provide --db or --file.")
        raise typer.Exit(code=2)

    if "://" in src:
        return _make_db_source(src)
    return SchemaSource(kind="file", raw_value=src, display_value=src)


def _make_db_source(url: str) -> SchemaSource:
    """Build a ``SchemaSource`` for a DB URL, sanitizing the password."""
    import sqlalchemy as sa  # noqa: PLC0415

    try:
        display = sa.engine.make_url(url).render_as_string(hide_password=True)
    except sa.exc.ArgumentError:
        display = "<invalid URL>"
    return SchemaSource(kind="db", raw_value=url, display_value=display)
