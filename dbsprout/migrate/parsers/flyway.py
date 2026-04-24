"""Flyway migration parser.

Converts a Flyway project (``db/migration/V*__*.sql``) into a
``list[SchemaChange]`` via sqlglot parsing only — no migration SQL executes,
no Flyway runtime dependency.

Statement dispatch lives in :mod:`dbsprout.migrate.parsers._sql_walker` so it
can be shared with other SQL-migration parsers (e.g. Prisma). Flyway retains
versioned-file discovery, ``V*__*.sql`` / ``R__`` / ``U__`` filename handling,
and ``${placeholder}`` substitution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dbsprout.migrate.parsers import MigrationParseError

# Production use needs only these two shared primitives plus the size cap and
# path-containment helper. The remaining re-imports below exist so tests and
# any external consumers that relied on the pre-refactor private API surface
# keep resolving; ruff F401 is suppressed on the block because the symbols are
# intentional re-exports.
from dbsprout.migrate.parsers._sql_walker import (  # noqa: F401  (re-export for back-compat)
    MAX_MIGRATION_BYTES,
    _alter_column_default_change,
    _alter_column_nullability_change,
    _alter_column_type_change,
    _column_def_to_dict,
    _column_ref_to_fk,
    _extract_inline_fks,
    _FKLedger,
    _handle_add_column,
    _handle_add_constraint,
    _handle_alter_column,
    _handle_alter_table,
    _handle_create_index,
    _handle_create_table,
    _handle_drop_column,
    _handle_drop_constraint,
    _handle_drop_index,
    _handle_drop_table,
    _handle_rename_column,
    _handle_rename_table,
    _split_qualified,
    _strip_quotes,
    _table_fk_to_detail,
    is_contained,
    parse_sql_text,
    walk_statements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlglot import exp

    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)

# Re-export the shared size cap under its historical private name so callers and
# tests that imported ``_MAX_MIGRATION_BYTES`` keep working after the refactor.
_MAX_MIGRATION_BYTES = MAX_MIGRATION_BYTES

# Alias the walker entry point to match the pre-refactor private name so tests
# that imported ``_walk_statements`` from this module continue to resolve.
_walk_statements = walk_statements

_DEFAULT_LOCATIONS: tuple[str, ...] = (
    "db/migration",
    "src/main/resources/db/migration",
    "migrations",
)

_VERSIONED_RE = re.compile(r"^V(?P<version>[0-9][0-9_.]*)__(?P<description>.+)\.sql$")
_REPEATABLE_RE = re.compile(r"^R__(?P<description>.+)\.sql$")
_UNDO_RE = re.compile(r"^U(?P<version>[0-9][0-9_.]*)__(?P<description>.+)\.sql$")
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class FlywayMigrationParser:
    """Parse Flyway versioned SQL migration histories into ``SchemaChange`` lists.

    ``placeholders`` values are substituted verbatim into migration SQL text
    before parsing. The parser itself never executes SQL, but callers that
    later replay emitted ``SchemaChange`` records against a live database are
    responsible for ensuring placeholder values do not contain SQL
    metacharacters (``;``, ``'``, ``--``). Pass values from a trusted source
    (``flyway.conf``, CI secrets), not from end-user input.
    """

    dialect: str = "postgres"
    locations: tuple[str, ...] | None = None
    placeholders: tuple[tuple[str, str], ...] = ()

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        """Return the ordered forward history of schema changes under ``project_path``.

        Walks Flyway versioned SQL migrations in the configured ``locations`` (or
        default probes), substitutes ``placeholders``, and dispatches each SQL
        statement to the shared sqlglot walker. Raises ``MigrationParseError``
        when no migrations are discovered, versions collide, placeholders are
        unresolved, or sqlglot fails to parse a file.
        """
        files = _discover_migration_files(project_path, self.locations)
        if not files:
            searched = ", ".join(self.locations or _DEFAULT_LOCATIONS)
            raise MigrationParseError(
                f"no V*__*.sql found under {project_path}; searched: {searched}",
            )
        placeholders = dict(self.placeholders)
        ledger = _FKLedger()
        changes: list[SchemaChange] = []
        for file in files:
            stmts = _parse_file(file, dialect=self.dialect, placeholders=placeholders)
            changes.extend(walk_statements(stmts, ledger=ledger))
        return changes


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _parse_version(raw: str) -> tuple[int, ...]:
    segments = re.split(r"[._]", raw)
    try:
        return tuple(int(s) for s in segments if s != "")
    except ValueError as exc:
        raise MigrationParseError(f"invalid Flyway version '{raw}'") from exc


_VERSION_PAD_LEN = 16


def _version_sort_key(version: tuple[int, ...]) -> tuple[int, ...]:
    """Right-pad the version tuple so shorter versions sort before longer ones with same prefix.

    Pads to _VERSION_PAD_LEN segments; oversize versions raise MigrationParseError
    rather than silently sorting incorrectly.
    """
    if len(version) > _VERSION_PAD_LEN:
        raise MigrationParseError(
            f"Flyway version has {len(version)} segments; "
            f"max supported is {_VERSION_PAD_LEN}: {version}",
        )
    return version + (0,) * (_VERSION_PAD_LEN - len(version))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _resolve_locations(project_path: Path, locations: tuple[str, ...] | None) -> list[Path]:
    resolved_root = project_path.resolve()
    if locations is not None:
        dirs: list[Path] = []
        for loc in locations:
            candidate = (project_path / loc).resolve()
            if not is_contained(candidate, resolved_root):
                raise MigrationParseError(
                    f"location '{loc}' escapes project root {project_path}",
                )
            dirs.append(candidate)
        return dirs
    for default in _DEFAULT_LOCATIONS:
        candidate = (project_path / default).resolve()
        if candidate.is_dir() and is_contained(candidate, resolved_root):
            return [candidate]
    return []


def _discover_migration_files(
    project_path: Path,
    locations: tuple[str, ...] | None,
) -> list[Path]:
    resolved_root = project_path.resolve()
    dirs = _resolve_locations(project_path, locations)
    by_version: dict[tuple[int, ...], Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for sql_file in sorted(d.rglob("*.sql")):
            name = sql_file.name
            try:
                if sql_file.is_symlink():
                    logger.debug("skipping symlink %s", sql_file)
                    continue
                resolved_file = sql_file.resolve()
                if not is_contained(resolved_file, resolved_root):
                    logger.debug("skipping out-of-tree file %s", sql_file)
                    continue
                size = sql_file.stat().st_size
            except OSError:
                logger.debug("cannot stat %s; skipping", sql_file)
                continue
            if size > MAX_MIGRATION_BYTES:
                logger.debug("%s exceeds 1 MB size cap; skipping", sql_file)
                continue
            if _REPEATABLE_RE.match(name):
                logger.debug("repeatable migration %s skipped (out of scope)", sql_file)
                continue
            if _UNDO_RE.match(name):
                logger.debug("undo migration %s skipped (out of scope)", sql_file)
                continue
            m = _VERSIONED_RE.match(name)
            if not m:
                logger.debug("non-Flyway filename %s skipped", sql_file)
                continue
            version = _parse_version(m.group("version"))
            if version in by_version:
                raise MigrationParseError(
                    f"duplicate Flyway version {version}: {by_version[version]} vs {sql_file}",
                )
            by_version[version] = sql_file
    return [by_version[v] for v in sorted(by_version, key=_version_sort_key)]


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------


def _substitute_placeholders(text: str, mapping: dict[str, str]) -> str:
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        return mapping.get(key, m.group(0))  # leave unresolved for the checker

    return _PLACEHOLDER_RE.sub(_sub, text)


def _check_unresolved(text: str, file_path: Path) -> None:
    m = _PLACEHOLDER_RE.search(text)
    if m:
        raise MigrationParseError(
            f"unresolved placeholder {m.group(0)} in {file_path}; "
            f"pass placeholders={{'{m.group(1)}': '...'}} to FlywayMigrationParser",
            file_path=file_path,
        )


# ---------------------------------------------------------------------------
# File-level parser (placeholder-aware wrapper over the shared walker)
# ---------------------------------------------------------------------------


def _parse_file(
    file_path: Path,
    *,
    dialect: str,
    placeholders: dict[str, str],
) -> list[exp.Expression]:
    text = file_path.read_text(encoding="utf-8")
    text = _substitute_placeholders(text, placeholders)
    _check_unresolved(text, file_path)
    return parse_sql_text(text, dialect=dialect, file_path=file_path)


__all__ = ["FlywayMigrationParser"]
