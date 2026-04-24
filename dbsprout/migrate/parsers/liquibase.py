"""Liquibase XML changelog parser.

Converts a Liquibase XML changelog tree into a ``list[SchemaChange]`` via
``defusedxml`` parsing only — no Liquibase CLI, no JVM, no database connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)

_MAX_CHANGELOG_BYTES = 1024 * 1024  # 1 MB per-file cap

_DEFAULT_CHANGELOG_PATHS: tuple[str, ...] = (
    "db/changelog/db.changelog-master.xml",
    "src/main/resources/db/changelog/db.changelog-master.xml",
    "changelog.xml",
)


@dataclass(frozen=True)
class LiquibaseMigrationParser:
    """Parse Liquibase XML changelog trees into ``SchemaChange`` lists.

    ``changelog_file`` is resolved relative to the ``project_path`` passed to
    ``detect_changes``. If ``None``, the parser probes the default changelog
    paths and uses the first match.
    """

    changelog_file: str | None = None

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        root = _resolve_root(project_path, self.changelog_file)
        if root is None:
            searched = ", ".join(
                [self.changelog_file]
                if self.changelog_file is not None
                else list(_DEFAULT_CHANGELOG_PATHS),
            )
            raise MigrationParseError(
                f"no Liquibase changelog found under {project_path}; searched: {searched}",
            )
        return []  # placeholder until op handlers land


def _resolve_root(project_path: Path, changelog_file: str | None) -> Path | None:
    if changelog_file is not None:
        candidate = (project_path / changelog_file).resolve()
        return candidate if candidate.is_file() else None
    for default in _DEFAULT_CHANGELOG_PATHS:
        candidate = (project_path / default).resolve()
        if candidate.is_file():
            return candidate
    return None
