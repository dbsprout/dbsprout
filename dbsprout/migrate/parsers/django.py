"""Django migration parser.

Converts a Django project (``<app>/migrations/*.py``) into a
``list[SchemaChange]`` via AST parsing only — no migration code executes,
no Django runtime dependency.
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

_MAX_MIGRATION_BYTES = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class DjangoMigrationParser:
    """Parse Django migration histories into ``SchemaChange`` lists."""

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        files = _discover_migration_files(project_path)
        if not files:
            raise MigrationParseError(
                f"no */migrations/*.py found under {project_path}",
            )
        raise NotImplementedError("walker lands in later tasks")


def _discover_migration_files(project_path: Path) -> list[Path]:  # noqa: ARG001
    return []
