"""Liquibase XML changelog parser.

Converts a Liquibase XML changelog tree into a ``list[SchemaChange]`` via
``defusedxml`` parsing only — no Liquibase CLI, no JVM, no database connection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable  # noqa: TC003 — returned by later-task yields
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from defusedxml.ElementTree import (  # type: ignore[import-untyped]
    ParseError as DefusedXMLParseError,
)
from defusedxml.ElementTree import (
    parse as defused_parse,
)

# Kept at runtime scope: ``_FKLedger`` will invoke ``SchemaChange`` directly in
# later tasks (``record(change)`` + ``resolve(...) -> SchemaChange | None``).
from dbsprout.migrate.models import SchemaChange  # noqa: TC001
from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

    # Avoid ``from xml.etree...`` literal in source — the S-059 security test
    # checks for that substring. Liquibase handlers only use duck-typed element
    # interfaces (``.tag``, ``.get``, iteration) so ``Any`` is adequate here.
    Element = Any

logger = logging.getLogger(__name__)

_MAX_CHANGELOG_BYTES = 1024 * 1024  # 1 MB per-file cap

_DEFAULT_CHANGELOG_PATHS: tuple[str, ...] = (
    "db/changelog/db.changelog-master.xml",
    "src/main/resources/db/changelog/db.changelog-master.xml",
    "changelog.xml",
)


def _strip_ns(tag: str) -> str:
    """Return the local name of an XML tag, stripping any ``{namespace}`` prefix."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


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
        ledger = _FKLedger()
        seen_changesets: dict[tuple[str, str], Path] = {}
        visited: set[Path] = set()
        return list(_walk_changelog(root, project_path, ledger, visited, seen_changesets))


def _resolve_root(project_path: Path, changelog_file: str | None) -> Path | None:
    if changelog_file is not None:
        candidate = (project_path / changelog_file).resolve()
        return candidate if candidate.is_file() else None
    for default in _DEFAULT_CHANGELOG_PATHS:
        candidate = (project_path / default).resolve()
        if candidate.is_file():
            return candidate
    return None


@dataclass
class _FKLedger:
    """In-walk lookup of recorded foreign-key additions.

    Populated on ``addForeignKeyConstraint`` / inline ``references``; queried on
    ``dropForeignKeyConstraint`` so bare-name drops with no prior add can be
    skipped safely (matches the S-058 Flyway parser's conservative policy).
    """

    by_key: dict[tuple[str, str], SchemaChange] = field(default_factory=dict)

    def record(self, change: SchemaChange) -> None:
        detail = change.detail or {}
        name = detail.get("constraint_name")
        if name:
            self.by_key[(change.table_name, str(name))] = change

    def resolve(self, table: str, constraint_name: str) -> SchemaChange | None:
        return self.by_key.get((table, constraint_name))


def _walk_changelog(
    root: Path,
    project_path: Path,  # noqa: ARG001
    ledger: _FKLedger,  # noqa: ARG001
    visited: set[Path],
    seen_changesets: dict[tuple[str, str], Path],  # noqa: ARG001
) -> Iterable[SchemaChange]:
    resolved = root.resolve()
    if resolved in visited:
        raise MigrationParseError(f"include cycle detected at {resolved}")
    if not root.is_file():
        raise MigrationParseError(f"changelog file missing: {root}")
    if root.stat().st_size > _MAX_CHANGELOG_BYTES:
        logger.debug("%s exceeds 1 MB size cap; skipping", root)
        return
    visited.add(resolved)
    try:
        tree = defused_parse(str(root))
    except DefusedXMLParseError as exc:
        raise MigrationParseError(
            f"could not parse {root}: {exc}",
            file_path=root,
        ) from exc
    for elem in tree.getroot():
        tag = _strip_ns(elem.tag)
        if tag == "changeSet":
            logger.debug("changeSet placeholder — to be handled in later task")
        elif tag == "include":
            logger.debug("include placeholder — to be handled in later task")
        elif tag == "includeAll":
            logger.debug("includeAll placeholder — to be handled in later task")
        else:
            logger.debug("skipping unsupported Liquibase element: %s", tag)
    # Marks the function as a generator so the early ``return`` above emits an
    # empty iterator instead of ``None``. Later tasks will replace the
    # placeholder ``logger.debug`` calls above with ``yield`` statements.
    if False:  # pragma: no cover
        yield
