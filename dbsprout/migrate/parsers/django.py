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


def _discover_migration_files(project_path: Path) -> list[Path]:
    """Return every migration file under ``project_path``.

    Filters out ``__init__.py``, ``__pycache__`` entries, files larger than
    1 MB, and anything that resolves outside ``project_path`` (symlink guard).
    """
    project_resolved = project_path.resolve()
    found: list[Path] = []
    for path in sorted(project_path.rglob("migrations/*.py")):
        if path.name == "__init__.py":
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            logger.debug("skipping unresolvable migration path %s", path)
            continue
        if project_resolved not in resolved.parents and resolved != project_resolved:
            logger.debug("skipping symlink-escape migration path %s", path)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            logger.debug("skipping unreadable migration path %s", path)
            continue
        if size > _MAX_MIGRATION_BYTES:
            logger.debug("skipping oversize migration file %s (%d bytes)", path, size)
            continue
        found.append(path)
    return found
