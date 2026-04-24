"""Prisma migration parser.

Converts a Prisma project (``prisma/migrations/{timestamp}_{name}/migration.sql``)
into a ``list[SchemaChange]`` via sqlglot parsing only — no migration SQL executes,
no Prisma CLI dependency.

Statement dispatch is delegated to :mod:`dbsprout.migrate.parsers._sql_walker`,
shared with the Flyway parser. This module owns discovery, ordering by
timestamped subdirectory name, and SQL dialect resolution from
``migration_lock.toml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:  # Python 3.11+ stdlib
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]

from dbsprout.migrate.parsers import MigrationParseError
from dbsprout.migrate.parsers._sql_walker import (
    MAX_MIGRATION_BYTES,
    _FKLedger,
    is_contained,
    parse_sql_text,
    walk_statements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)

# Prisma ``provider`` values map to sqlglot dialect identifiers. CockroachDB
# emits PostgreSQL-compatible migration SQL, so we reuse the postgres dialect.
PROVIDER_DIALECTS: dict[str, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "sqlserver": "tsql",
    "cockroachdb": "postgres",
}


@dataclass(frozen=True)
class PrismaMigrationParser:
    """Parse a Prisma migration history into a ``list[SchemaChange]``.

    ``dialect`` overrides the provider recorded in ``migration_lock.toml``.
    ``migrations_dir`` is the project-relative directory containing the
    per-migration subdirectories (default ``"prisma/migrations"``).
    """

    dialect: str | None = None
    migrations_dir: str = "prisma/migrations"

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        """Return the ordered forward history of schema changes under ``project_path``.

        Raises :class:`MigrationParseError` when the migrations directory is
        missing, no ``migration.sql`` files are discovered, a subdirectory has
        no ``migration.sql``, ``migration_lock.toml`` is malformed, the
        provider is MongoDB, or sqlglot fails to parse any file.
        """
        project_root = project_path.resolve()
        mig_root = (project_path / self.migrations_dir).resolve()
        if not is_contained(mig_root, project_root):
            raise MigrationParseError(
                f"migrations_dir '{self.migrations_dir}' escapes project root {project_path}",
            )
        if not mig_root.is_dir():
            raise MigrationParseError(
                f"no {self.migrations_dir}/ under {project_path}",
            )
        files = _discover_migrations(mig_root, project_root)
        if not files:
            raise MigrationParseError(
                f"no migration.sql found under {mig_root}",
            )
        dialect = _resolve_dialect(mig_root, override=self.dialect)
        ledger = _FKLedger()
        changes: list[SchemaChange] = []
        for file in files:
            text = file.read_text(encoding="utf-8")
            stmts = parse_sql_text(text, dialect=dialect, file_path=file)
            changes.extend(walk_statements(stmts, ledger=ledger))
        return changes


# ---------------------------------------------------------------------------
# Dialect resolution
# ---------------------------------------------------------------------------


def _resolve_dialect(mig_root: Path, *, override: str | None) -> str:
    """Return the sqlglot dialect for migrations under ``mig_root``.

    Constructor ``override`` takes precedence. Otherwise, read
    ``migration_lock.toml`` and map its ``provider`` field via
    :data:`PROVIDER_DIALECTS`. Defaults to ``"postgres"`` with a debug log when
    the lock file is missing. Raises :class:`MigrationParseError` on malformed
    TOML, unknown providers, or ``provider = "mongodb"``.
    """
    if override is not None:
        return override
    lock = mig_root / "migration_lock.toml"
    if not lock.is_file():
        logger.debug("no migration_lock.toml under %s; defaulting to postgres", mig_root)
        return "postgres"
    try:
        data = tomllib.loads(lock.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise MigrationParseError(
            f"invalid migration_lock.toml: {exc}",
            file_path=lock,
        ) from exc
    provider = data.get("provider")
    if not isinstance(provider, str):
        raise MigrationParseError(
            "invalid migration_lock.toml: missing or non-string provider",
            file_path=lock,
        )
    if provider == "mongodb":
        raise MigrationParseError(
            "MongoDB provider has no SQL migrations",
            file_path=lock,
        )
    if provider not in PROVIDER_DIALECTS:
        raise MigrationParseError(
            f"unknown Prisma provider '{provider}' in migration_lock.toml",
            file_path=lock,
        )
    return PROVIDER_DIALECTS[provider]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_migrations(mig_root: Path, project_root: Path) -> list[Path]:
    """Return discovered ``migration.sql`` paths ordered by subdirectory stem.

    Each subdirectory under ``mig_root`` must contain a ``migration.sql``; a
    missing file raises :class:`MigrationParseError`. Symlinked migration files,
    oversize files, and files resolving outside ``project_root`` are skipped
    with a debug log. Subdirectories that are symlinks to paths outside
    ``project_root`` raise :class:`MigrationParseError` (path-traversal defence).
    """
    files: list[Path] = []
    subdirs = sorted(p for p in mig_root.iterdir() if p.is_dir())
    for sub in subdirs:
        size = 0
        mig: Path | None = None
        try:
            if sub.is_symlink():
                resolved_sub = sub.resolve()
                if not is_contained(resolved_sub, project_root):
                    raise MigrationParseError(
                        f"migration subdir '{sub.name}' escapes project root",
                        file_path=sub,
                    )
                logger.debug("skipping symlinked subdir %s", sub)
                continue
            mig = sub / "migration.sql"
            if not mig.exists() and not mig.is_symlink():
                raise MigrationParseError(
                    f"{sub.name} has no migration.sql",
                    file_path=sub,
                )
            if mig.is_symlink():
                logger.debug("skipping symlinked migration %s", mig)
                continue
            resolved = mig.resolve()
            if not is_contained(resolved, project_root):
                logger.debug("skipping out-of-tree migration %s", mig)
                continue
            size = resolved.stat().st_size
        except OSError:
            logger.debug("cannot stat %s; skipping", sub)
            continue
        if mig is None:  # pragma: no cover - defensive: only reached if try body short-circuits
            continue
        if size > MAX_MIGRATION_BYTES:
            logger.debug("%s exceeds 1 MB size cap; skipping", mig)
            continue
        # Append the already-validated resolved path so a later TOCTOU swap of
        # ``sub/migration.sql`` to an out-of-tree symlink can't redirect the read.
        files.append(resolved)
    return files
