"""Flyway migration parser.

Converts a Flyway project (``db/migration/V*__*.sql``) into a
``list[SchemaChange]`` via sqlglot parsing only — no migration SQL executes,
no Flyway runtime dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)

_MAX_MIGRATION_BYTES = 1024 * 1024  # 1 MB

_DEFAULT_LOCATIONS: tuple[str, ...] = (
    "db/migration",
    "src/main/resources/db/migration",
    "migrations",
)


@dataclass(frozen=True)
class FlywayMigrationParser:
    """Parse Flyway versioned SQL migration histories into ``SchemaChange`` lists."""

    dialect: str = "postgres"
    locations: tuple[str, ...] | None = None
    placeholders: tuple[tuple[str, str], ...] = ()

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        files = _discover_migration_files(project_path, self.locations)
        if not files:
            searched = ", ".join(self.locations or _DEFAULT_LOCATIONS)
            raise MigrationParseError(
                f"no V*__*.sql found under {project_path}; searched: {searched}",
            )
        raise NotImplementedError("walker lands in later tasks")


def _discover_migration_files(
    project_path: Path,  # noqa: ARG001
    locations: tuple[str, ...] | None,  # noqa: ARG001
) -> list[Path]:
    return []


# ---------------------------------------------------------------------------
# FK ledger placeholder (to be filled in Task 6)
# ---------------------------------------------------------------------------


@dataclass
class _FKLedger:
    by_key: dict[tuple[str, str], SchemaChange] = field(default_factory=dict)

    def record(self, change: SchemaChange) -> None:
        detail = change.detail or {}
        name = detail.get("constraint_name")
        if name:
            self.by_key[(change.table_name, str(name))] = change

    def resolve(self, table: str, constraint_name: str) -> SchemaChange | None:
        return self.by_key.get((table, constraint_name))
