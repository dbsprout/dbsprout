"""Flyway migration parser.

Converts a Flyway project (``db/migration/V*__*.sql``) into a
``list[SchemaChange]`` via sqlglot parsing only — no migration SQL executes,
no Flyway runtime dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError as SqlglotParseError
from sqlglot.errors import TokenError

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

_VERSIONED_RE = re.compile(r"^V(?P<version>[0-9][0-9_.]*)__(?P<description>.+)\.sql$")
_REPEATABLE_RE = re.compile(r"^R__(?P<description>.+)\.sql$")
_UNDO_RE = re.compile(r"^U(?P<version>[0-9][0-9_.]*)__(?P<description>.+)\.sql$")
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


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


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _parse_version(raw: str) -> tuple[int, ...]:
    segments = re.split(r"[._]", raw)
    try:
        return tuple(int(s) for s in segments if s != "")
    except ValueError as exc:
        raise MigrationParseError(f"invalid Flyway version '{raw}'") from exc


def _version_sort_key(version: tuple[int, ...]) -> tuple[int, ...]:
    """Right-pad to length 8 so shorter versions sort before longer ones with same prefix."""
    return version + (0,) * (8 - len(version))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _resolve_locations(project_path: Path, locations: tuple[str, ...] | None) -> list[Path]:
    if locations is not None:
        return [project_path / loc for loc in locations]
    for default in _DEFAULT_LOCATIONS:
        candidate = project_path / default
        if candidate.is_dir():
            return [candidate]
    return []


def _discover_migration_files(
    project_path: Path,
    locations: tuple[str, ...] | None,
) -> list[Path]:
    dirs = _resolve_locations(project_path, locations)
    by_version: dict[tuple[int, ...], Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for sql_file in sorted(d.rglob("*.sql")):
            name = sql_file.name
            try:
                size = sql_file.stat().st_size
            except OSError:
                logger.debug("cannot stat %s; skipping", sql_file)
                continue
            if size > _MAX_MIGRATION_BYTES:
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
# Identifier helpers
# ---------------------------------------------------------------------------


def _strip_quotes(ident: str) -> str:
    s = ident.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "`"):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        return s[1:-1]
    return s


def _split_qualified(ident: str) -> tuple[str | None, str]:
    parts = [_strip_quotes(p) for p in ident.split(".") if p]
    if not parts:
        return None, ""
    if len(parts) == 1:
        return None, parts[0]
    return parts[-2], parts[-1]


# ---------------------------------------------------------------------------
# sqlglot file parser
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
    try:
        raw_stmts = sqlglot.parse(text, read=dialect)
    except (SqlglotParseError, TokenError) as exc:
        raise MigrationParseError(
            f"could not parse {file_path} (dialect={dialect}): {exc}",
            file_path=file_path,
        ) from exc
    return cast("list[exp.Expression]", [s for s in raw_stmts if s is not None])


# ---------------------------------------------------------------------------
# FK ledger
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


# ---------------------------------------------------------------------------
# Statement walker (stub — dispatch added per-task)
# ---------------------------------------------------------------------------


def _walk_statements(
    stmts: list[exp.Expression],
    *,
    dialect: str,  # noqa: ARG001
    ledger: _FKLedger,  # noqa: ARG001
) -> list[SchemaChange]:
    out: list[SchemaChange] = []
    for stmt in stmts:
        logger.debug("skipping unsupported statement: %s", type(stmt).__name__)
    return out
