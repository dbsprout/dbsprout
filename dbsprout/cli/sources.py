"""Shared helper — resolve schema source from flags or config.

Used by ``dbsprout diff`` and ``dbsprout generate --incremental`` so the
two commands always accept the same precedence (``--db`` > ``--file`` >
``config.schema.source``).

The helper is CLI-framework-agnostic: on failure it raises
:class:`SchemaSourceError`. CLI callers translate that into their preferred
``typer.Exit`` code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


class SchemaSourceError(ValueError):
    """No schema source could be resolved from flags or config."""


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
    Raises :class:`SchemaSourceError` if none is set.
    """
    if db is not None:
        return _make_db_source(db)
    if file is not None:
        return SchemaSource(kind="file", raw_value=file, display_value=file)

    from pydantic import ValidationError  # noqa: PLC0415

    from dbsprout.config import load_config  # noqa: PLC0415

    try:
        cfg = load_config(output_dir / "dbsprout.toml")
    except (ValueError, OSError, ValidationError) as exc:
        msg = "No schema source. Provide --db or --file."
        raise SchemaSourceError(msg) from exc

    src = cfg.schema_.source
    if not src:
        msg = "No schema source. Provide --db or --file."
        raise SchemaSourceError(msg)

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
