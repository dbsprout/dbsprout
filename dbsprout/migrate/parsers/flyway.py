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

from dbsprout.migrate.models import SchemaChange, SchemaChangeType
from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

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
        placeholders = dict(self.placeholders)
        ledger = _FKLedger()
        changes: list[SchemaChange] = []
        for file in files:
            stmts = _parse_file(file, dialect=self.dialect, placeholders=placeholders)
            changes.extend(_walk_statements(stmts, dialect=self.dialect, ledger=ledger))
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
# Column definition helper
# ---------------------------------------------------------------------------


def _column_def_to_dict(col: exp.ColumnDef) -> dict[str, object]:
    constraints = col.args.get("constraints") or []
    nullable = True
    default: str | None = None
    pk = False
    for c in constraints:
        kind = c.args.get("kind")
        if isinstance(kind, exp.NotNullColumnConstraint):
            nullable = False
        elif isinstance(kind, exp.PrimaryKeyColumnConstraint):
            pk = True
            nullable = False
        elif isinstance(kind, exp.DefaultColumnConstraint):
            default = kind.args["this"].sql(dialect=None) if kind.args.get("this") else None
    dtype = col.args.get("kind")
    sql_type = dtype.sql(dialect=None) if isinstance(dtype, exp.DataType) else ""
    return {
        "name": _strip_quotes(col.name),
        "sql_type": sql_type,
        "nullable": nullable,
        "default": default,
        "primary_key": pk,
    }


def _extract_inline_fks(create: exp.Create) -> list[dict[str, object]]:
    fks: list[dict[str, object]] = []
    schema = create.this  # exp.Schema
    if not isinstance(schema, exp.Schema):
        return fks
    for expr in schema.expressions:
        # Column-level REFERENCES
        if isinstance(expr, exp.ColumnDef):
            for c in expr.args.get("constraints") or []:
                kind = c.args.get("kind")
                if isinstance(kind, exp.Reference):
                    ref = kind.args["this"]
                    # sqlglot wraps REFERENCES target as exp.Schema with .this = exp.Table
                    if isinstance(ref, exp.Schema):
                        ref_table_node = ref.this
                        ref_table = (
                            _split_qualified(ref_table_node.name)[1]
                            if isinstance(ref_table_node, exp.Table)
                            else ""
                        )
                        remote_cols = [
                            _strip_quotes(rc.name) for rc in (ref.args.get("expressions") or [])
                        ] or ["id"]
                    else:
                        ref_name = ref.name if hasattr(ref, "name") else str(ref)
                        _ref_schema, ref_table = _split_qualified(ref_name)
                        remote_cols = [
                            _strip_quotes(rc.name) for rc in (ref.args.get("expressions") or [])
                        ] or ["id"]
                    fks.append(
                        {
                            "constraint_name": None,
                            "local_cols": [_strip_quotes(expr.name)],
                            "ref_table": ref_table,
                            "remote_cols": remote_cols,
                        }
                    )
        # Table-level FOREIGN KEY constraint
        elif isinstance(expr, exp.ForeignKey):
            fk_local = [_strip_quotes(e.name) for e in (expr.args.get("expressions") or [])]
            fk_ref = expr.args.get("reference")
            fk_ref_table = ""
            fk_remote: list[str] = []
            if isinstance(fk_ref, exp.Reference):
                fk_ref_this = fk_ref.this
                if isinstance(fk_ref_this, exp.Table):
                    fk_ref_table = _split_qualified(fk_ref_this.name)[1]
                fk_remote = [_strip_quotes(e.name) for e in (fk_ref.args.get("expressions") or [])]
            fks.append(
                {
                    "constraint_name": None,
                    "local_cols": fk_local,
                    "ref_table": fk_ref_table,
                    "remote_cols": fk_remote,
                }
            )
    return fks


# ---------------------------------------------------------------------------
# CREATE/DROP TABLE handlers
# ---------------------------------------------------------------------------


def _handle_create_table(node: exp.Create) -> list[SchemaChange]:
    schema_expr = node.this
    if not isinstance(schema_expr, exp.Schema):
        return []
    table_expr = schema_expr.this
    if not isinstance(table_expr, exp.Table):  # pragma: no cover
        return []
    schema_name, table_name = _split_qualified(table_expr.sql(dialect=None))
    cols = [_column_def_to_dict(e) for e in schema_expr.expressions if isinstance(e, exp.ColumnDef)]
    fks = _extract_inline_fks(node)
    detail: dict[str, object] = {"columns": cols, "foreign_keys": fks}
    if schema_name:
        detail["schema"] = schema_name
    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=table_name,
            detail=detail,
        )
    ]


def _handle_drop_table(node: exp.Drop) -> list[SchemaChange]:
    table = node.this
    if not isinstance(table, exp.Table):  # pragma: no cover
        return []
    schema_name, table_name = _split_qualified(table.sql(dialect=None))
    detail: dict[str, object] = {}
    if schema_name:
        detail["schema"] = schema_name
    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name=table_name,
            detail=detail or None,
        )
    ]


# ---------------------------------------------------------------------------
# Statement walker
# ---------------------------------------------------------------------------


def _handle_add_column(
    table_name: str,
    schema_name: str | None,
    col: exp.ColumnDef,
) -> list[SchemaChange]:
    col_dict = _column_def_to_dict(col)
    detail: dict[str, object] = {k: v for k, v in col_dict.items() if k != "name"}
    if schema_name:
        detail["schema"] = schema_name
    changes: list[SchemaChange] = [
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name=table_name,
            column_name=str(col_dict["name"]),
            detail=detail,
        )
    ]
    # Inline REFERENCES → emit additional FOREIGN_KEY_ADDED
    for c in col.args.get("constraints") or []:
        kind = c.args.get("kind")
        if isinstance(kind, exp.Reference):
            ref = kind.args["this"]
            if isinstance(ref, exp.Schema):
                ref_table_node = ref.this
                ref_table = (
                    _split_qualified(ref_table_node.name)[1]
                    if isinstance(ref_table_node, exp.Table)
                    else ""
                )
                remote_cols = [
                    _strip_quotes(rc.name) for rc in (ref.args.get("expressions") or [])
                ] or ["id"]
            else:
                ref_name = ref.name if hasattr(ref, "name") else str(ref)
                _, ref_table = _split_qualified(ref_name)
                remote_cols = [
                    _strip_quotes(rc.name) for rc in (ref.args.get("expressions") or [])
                ] or ["id"]
            changes.append(
                SchemaChange(
                    change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
                    table_name=table_name,
                    detail={
                        "constraint_name": None,
                        "local_cols": [col_dict["name"]],
                        "ref_table": ref_table,
                        "remote_cols": remote_cols,
                    },
                )
            )
    return changes


def _handle_drop_column(table_name: str, drop: exp.Drop) -> list[SchemaChange]:
    col_expr = drop.this
    col_name = _strip_quotes(col_expr.name) if col_expr else ""
    return [
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name=table_name,
            column_name=col_name,
        )
    ]


def _handle_alter_column(
    table_name: str,
    action: exp.AlterColumn,
) -> list[SchemaChange]:
    col_name = _strip_quotes(action.this.name) if action.this else ""
    out: list[SchemaChange] = []
    dtype = action.args.get("dtype")
    if isinstance(dtype, exp.DataType):
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                table_name=table_name,
                column_name=col_name,
                new_value=dtype.sql(dialect=None),
            )
        )
    allow_null = action.args.get("allow_null")
    if allow_null is False:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
                table_name=table_name,
                column_name=col_name,
                new_value="NOT NULL",
            )
        )
    elif allow_null is True:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
                table_name=table_name,
                column_name=col_name,
                new_value="NULL",
            )
        )
    # SET DEFAULT or DROP DEFAULT
    if "default" in action.args:
        default_expr = action.args.get("default")
        new_value = (
            default_expr.sql(dialect=None) if isinstance(default_expr, exp.Expression) else None
        )
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
                table_name=table_name,
                column_name=col_name,
                new_value=new_value,
            )
        )
    elif action.args.get("drop") is True and dtype is None and allow_null is None:
        # DROP DEFAULT (no "default" key, but drop=True and no other dimension)
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
                table_name=table_name,
                column_name=col_name,
                new_value=None,
            )
        )
    if not out:
        logger.debug("ALTER COLUMN %s on %s produced no recognised dimension", col_name, table_name)
    return out


def _handle_add_constraint(
    table_name: str,
    action: exp.AddConstraint,
    ledger: _FKLedger,
) -> list[SchemaChange]:
    # In sqlglot >=25, AddConstraint.expressions holds the Constraint node(s).
    fk: exp.ForeignKey | None = None
    constraint_name: str | None = None
    for expr in action.expressions:
        if isinstance(expr, exp.Constraint):
            constraint_name = _strip_quotes(expr.name) if expr.name else None
            for inner in expr.expressions:
                if isinstance(inner, exp.ForeignKey):
                    fk = inner
                    break
        elif isinstance(expr, exp.ForeignKey):
            fk = expr
    if fk is None:
        logger.debug("ADD CONSTRAINT on %s is not a FK; skipping", table_name)
        return []
    local_cols = [_strip_quotes(e.name) for e in (fk.args.get("expressions") or [])]
    ref = fk.args.get("reference")
    ref_table = ""
    remote_cols: list[str] = []
    if isinstance(ref, exp.Reference):
        ref_this = ref.this
        if isinstance(ref_this, exp.Schema):
            # Schema wraps Table
            tbl = ref_this.this
            ref_table = _split_qualified(tbl.name)[1] if isinstance(tbl, exp.Table) else ""
            remote_cols = [_strip_quotes(e.name) for e in (ref_this.args.get("expressions") or [])]
        elif isinstance(ref_this, exp.Table):
            ref_table = _split_qualified(ref_this.name)[1]
            remote_cols = [_strip_quotes(e.name) for e in (ref.args.get("expressions") or [])]
    change = SchemaChange(
        change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
        table_name=table_name,
        detail={
            "constraint_name": constraint_name,
            "local_cols": local_cols,
            "ref_table": ref_table,
            "remote_cols": remote_cols,
        },
    )
    ledger.record(change)
    return [change]


def _handle_drop_constraint(
    table_name: str,
    action: exp.Drop,
    ledger: _FKLedger,
) -> list[SchemaChange]:
    name_node = action.this
    name = _strip_quotes(name_node.name) if name_node else ""
    existing = ledger.resolve(table_name, name)
    if existing is None:
        logger.debug(
            "DROP CONSTRAINT %s on %s not in FK ledger; skipping (likely CHECK/UNIQUE)",
            name,
            table_name,
        )
        return []
    return [
        SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
            table_name=table_name,
            detail={"constraint_name": name},
        )
    ]


def _handle_alter_table(
    node: exp.Alter,
    ledger: _FKLedger,
) -> list[SchemaChange]:
    out: list[SchemaChange] = []
    table_expr = node.this
    schema_name, table_name = _split_qualified(
        table_expr.sql(dialect=None) if isinstance(table_expr, exp.Table) else ""
    )
    actions = node.args.get("actions") or []
    for action in actions:
        if isinstance(action, exp.ColumnDef):
            out.extend(_handle_add_column(table_name, schema_name, action))
        elif isinstance(action, exp.AlterColumn):
            out.extend(_handle_alter_column(table_name, action))
        elif isinstance(action, exp.AlterRename):
            out.extend(_handle_rename_table(table_name, action))
        elif isinstance(action, exp.RenameColumn):
            out.extend(_handle_rename_column(table_name, action))
        elif isinstance(action, exp.AddConstraint):
            out.extend(_handle_add_constraint(table_name, action, ledger))
        elif isinstance(action, exp.Drop):
            drop_kind = action.args.get("kind") or ""
            if isinstance(drop_kind, str) and drop_kind.upper() == "COLUMN":
                out.extend(_handle_drop_column(table_name, action))
            elif isinstance(drop_kind, str) and drop_kind.upper() == "CONSTRAINT":
                out.extend(_handle_drop_constraint(table_name, action, ledger))
            else:
                logger.debug("skipping unsupported ALTER TABLE DROP action: kind=%s", drop_kind)
        else:
            logger.debug(
                "skipping unsupported ALTER TABLE action: %s",
                type(action).__name__,
            )
    return out


def _handle_rename_table(
    old_table: str,
    rename_action: exp.AlterRename,
) -> list[SchemaChange]:
    new_expr = rename_action.this
    _, new_table = _split_qualified(
        new_expr.sql(dialect=None) if isinstance(new_expr, exp.Expression) else ""
    )
    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name=old_table,
            detail={"rename_of": new_table},
        ),
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=new_table,
            detail={"rename_of": old_table},
        ),
    ]


def _handle_rename_column(
    table_name: str,
    action: exp.RenameColumn,
) -> list[SchemaChange]:
    old_col = action.this
    old_name = _strip_quotes(old_col.name) if old_col else ""
    new_col = action.args.get("to")
    new_name = _strip_quotes(new_col.name) if new_col is not None else ""
    return [
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name=table_name,
            column_name=old_name,
            detail={"rename_of": new_name},
        ),
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name=table_name,
            column_name=new_name,
            detail={"rename_of": old_name},
        ),
    ]


def _handle_create_index(node: exp.Create) -> list[SchemaChange]:
    ix = node.this
    if not isinstance(ix, exp.Index):
        return []
    index_name = _strip_quotes(ix.name)
    table_expr = ix.args.get("table")
    if not isinstance(table_expr, exp.Table):  # pragma: no cover
        return []
    _, table_name = _split_qualified(table_expr.sql(dialect=None))
    params = ix.args.get("params")
    cols: list[str] = []
    if isinstance(params, exp.IndexParameters):
        for col_node in params.args.get("columns") or []:
            # Columns are wrapped in Ordered nodes
            inner = col_node.this if isinstance(col_node, exp.Ordered) else col_node
            if isinstance(inner, exp.Column):
                cols.append(_strip_quotes(inner.name))
            else:
                cols.append(_strip_quotes(inner.sql(dialect=None)))
    return [
        SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name=table_name,
            detail={"index_name": index_name, "cols": cols},
        )
    ]


def _handle_drop_index(node: exp.Drop) -> list[SchemaChange]:
    target = node.this
    name = _strip_quotes(target.name) if target else ""
    return [
        SchemaChange(
            change_type=SchemaChangeType.INDEX_REMOVED,
            table_name="",
            detail={"index_name": name},
        )
    ]


def _walk_statements(
    stmts: list[exp.Expression],
    *,
    dialect: str,
    ledger: _FKLedger,
) -> list[SchemaChange]:
    out: list[SchemaChange] = []
    for stmt in stmts:
        kind = (stmt.args.get("kind") or "").upper()
        if isinstance(stmt, exp.Create) and kind == "TABLE":
            out.extend(_handle_create_table(stmt))
        elif isinstance(stmt, exp.Drop) and kind == "TABLE":
            out.extend(_handle_drop_table(stmt))
        elif isinstance(stmt, exp.Create) and kind == "INDEX":
            out.extend(_handle_create_index(stmt))
        elif isinstance(stmt, exp.Drop) and kind == "INDEX":
            out.extend(_handle_drop_index(stmt))
        elif isinstance(stmt, exp.Alter) and kind == "TABLE":
            out.extend(_handle_alter_table(stmt, ledger))
        else:
            logger.debug(
                "skipping unsupported statement: %s kind=%s",
                type(stmt).__name__,
                kind or "N/A",
            )
    _ = dialect  # available for future dialect-specific handling
    return out
