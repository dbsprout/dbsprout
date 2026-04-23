"""Migration parser Protocol and shared errors.

Each framework (Alembic, Django, Flyway, Liquibase, Prisma) provides a
concrete ``MigrationParser`` implementation that converts its migration
history into a ``list[SchemaChange]`` the incremental updater can consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange


class MigrationParseError(Exception):
    """Raised when a migration source cannot be parsed.

    Attributes:
        file_path: Path of the offending file, or ``None`` for
            project-level failures (multiple heads, no versions dir, etc.).
    """

    def __init__(self, message: str, file_path: Path | None = None) -> None:
        super().__init__(message)
        self.file_path = file_path


@runtime_checkable
class MigrationParser(Protocol):
    """Protocol every framework-specific migration parser implements."""

    def detect_changes(self, project_path: Path) -> list[SchemaChange]: ...


__all__ = ["MigrationParseError", "MigrationParser"]
