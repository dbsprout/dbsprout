"""Alembic migration parser.

Converts an Alembic project (``alembic.ini`` + ``versions/*.py``) into a
``list[SchemaChange]`` via AST parsing only — no migration code is executed.
Also offers an opt-in ``compare_metadata`` wrapper that delegates to
``alembic.autogenerate`` for live-DB comparisons.
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import MetaData

    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlembicParser:
    """Parse Alembic migration histories into ``SchemaChange`` lists."""

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        """Return the full forward history (base → head) of schema changes."""
        versions_dir = _discover_versions_dir(project_path)
        revisions = _collect_revisions(versions_dir)
        ordered = _linearize_revisions(revisions)
        changes: list[SchemaChange] = []
        for rev in ordered:
            changes.extend(_parse_upgrade(rev))
        return changes

    def compare_metadata(self, db_url: str, metadata: MetaData) -> list[SchemaChange]:
        """Diff a SQLAlchemy MetaData against a live DB via Alembic autogenerate."""
        raise NotImplementedError  # filled in Task 12


def _discover_versions_dir(project_path: Path) -> Path:
    """Resolve the Alembic versions directory.

    Order: ``alembic.ini[alembic].script_location/versions`` →
    ``./alembic/versions`` → ``./migrations/versions``.
    """
    ini = project_path / "alembic.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if cp.has_option("alembic", "script_location"):
            loc = project_path / cp.get("alembic", "script_location") / "versions"
            if loc.is_dir():
                return loc

    for fallback in (
        project_path / "alembic" / "versions",
        project_path / "migrations" / "versions",
    ):
        if fallback.is_dir():
            return fallback

    raise MigrationParseError("No Alembic versions/ directory found", file_path=project_path)


def _collect_revisions(versions_dir: Path) -> list[_Revision]:  # noqa: ARG001
    return []


def _linearize_revisions(revisions: list[_Revision]) -> list[_Revision]:  # noqa: ARG001
    return []


def _parse_upgrade(rev: _Revision) -> list[SchemaChange]:  # noqa: ARG001
    return []


@dataclass(frozen=True)
class _Revision:
    """Internal — parsed revision header + AST for a single Alembic file."""

    path: Path
    revision: str
    down_revision: str | None
    # ast.Module stored loosely as object to keep dataclass frozen-hashable-friendly
    module: object
