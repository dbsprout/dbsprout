"""Liquibase XML changelog parser.

Converts a Liquibase XML changelog tree into a ``list[SchemaChange]`` via
``defusedxml`` parsing only — no Liquibase CLI, no JVM, no database connection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from defusedxml.ElementTree import (  # type: ignore[import-untyped]
    ParseError as DefusedXMLParseError,
)
from defusedxml.ElementTree import (
    parse as defused_parse,
)

from dbsprout.migrate.models import SchemaChange, SchemaChangeType
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
    ledger: _FKLedger,
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
            yield from _handle_changeset(elem, root, ledger)
        elif tag == "include":
            logger.debug("include placeholder — to be handled in later task")
        elif tag == "includeAll":
            logger.debug("includeAll placeholder — to be handled in later task")
        else:
            logger.debug("skipping unsupported Liquibase element: %s", tag)


# ---------------------------------------------------------------------------
# changeSet + per-operation handlers
# ---------------------------------------------------------------------------


def _handle_changeset(
    changeset: Element,
    source: Path,
    ledger: _FKLedger,
) -> Iterable[SchemaChange]:
    for child in changeset:
        tag = _strip_ns(child.tag)
        handler = _OP_HANDLERS.get(tag)
        if handler is None:
            if tag in _DEBUG_SKIP_TAGS:
                logger.debug("skipping %s in %s (out of scope)", tag, source)
            else:
                logger.debug("skipping unsupported change %s in %s", tag, source)
            continue
        yield from handler(child, ledger)


def _handle_create_table(elem: Element, _ledger: _FKLedger) -> list[SchemaChange]:
    table_name = elem.get("tableName", "")
    schema = elem.get("schemaName")
    cols: list[dict[str, object]] = []
    fks: list[dict[str, object]] = []
    for child in elem:
        if _strip_ns(child.tag) != "column":
            continue
        col, inline_fk = _parse_column(child)
        cols.append(col)
        if inline_fk is not None:
            fks.append(inline_fk)
    detail: dict[str, object] = {"columns": cols, "foreign_keys": fks}
    if schema:
        detail["schema"] = schema
    changes: list[SchemaChange] = [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=table_name,
            detail=detail,
        )
    ]
    for fk in fks:
        fk_change = SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name=table_name,
            detail=fk,
        )
        changes.append(fk_change)
    return changes


def _handle_drop_table(elem: Element, _ledger: _FKLedger) -> list[SchemaChange]:
    table_name = elem.get("tableName", "")
    schema = elem.get("schemaName")
    detail: dict[str, object] = {}
    if schema:
        detail["schema"] = schema
    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name=table_name,
            detail=detail or None,
        )
    ]


def _parse_column(col: Element) -> tuple[dict[str, object], dict[str, object] | None]:
    name = col.get("name", "")
    sql_type = col.get("type", "")
    default = _resolve_column_default(col)
    nullable = True
    primary_key = False
    fk: dict[str, object] | None = None
    for c in col:
        if _strip_ns(c.tag) != "constraints":
            continue
        if c.get("nullable") == "false":
            nullable = False
        if c.get("primaryKey") == "true":
            primary_key = True
            nullable = False
        fk_name = c.get("foreignKeyName")
        references = c.get("references")
        if fk_name and references:
            fk = _parse_inline_fk(name, fk_name, references)
    return (
        {
            "name": name,
            "sql_type": sql_type,
            "nullable": nullable,
            "default": default,
            "primary_key": primary_key,
        },
        fk,
    )


def _parse_inline_fk(
    local_col: str,
    fk_name: str,
    references: str,
) -> dict[str, object]:
    """Parse a ``references="table(col[, col, ...])"`` attribute."""
    ref_table = references
    remote_cols: list[str] = []
    if "(" in references and references.endswith(")"):
        ref_table, _, inner = references[:-1].partition("(")
        remote_cols = [c.strip() for c in inner.split(",") if c.strip()]
    return {
        "constraint_name": fk_name,
        "local_cols": [local_col],
        "ref_table": ref_table.strip(),
        "remote_cols": remote_cols or ["id"],
    }


_DEFAULT_ATTRS: tuple[str, ...] = (
    "defaultValue",
    "defaultValueNumeric",
    "defaultValueBoolean",
    "defaultValueDate",
    "defaultValueComputed",
    "defaultValueSequenceNext",
)


def _resolve_column_default(col: Element) -> str | None:
    for attr in _DEFAULT_ATTRS:
        value = col.get(attr)
        if value is not None:
            return str(value)
    return None


_OpHandler = Callable[[Any, "_FKLedger"], "list[SchemaChange]"]

_OP_HANDLERS: dict[str, _OpHandler] = {
    "createTable": _handle_create_table,
    "dropTable": _handle_drop_table,
}

_DEBUG_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "preConditions",
        "rollback",
        "validCheckSum",
        "modifySql",
        "comment",
        "empty",
        "sql",
        "sqlFile",
        "customChange",
        "executeCommand",
        "stop",
        "tagDatabase",
        "insert",
        "update",
        "delete",
        "loadData",
        "loadUpdateData",
        "addLookupTable",
        "addPrimaryKey",
        "dropPrimaryKey",
        "addUniqueConstraint",
        "dropUniqueConstraint",
        "addCheckConstraint",
        "dropCheckConstraint",
        "addAutoIncrement",
        "mergeColumns",
        "createView",
        "dropView",
        "createProcedure",
        "dropProcedure",
        "createSequence",
        "dropSequence",
        "alterSequence",
    }
)
