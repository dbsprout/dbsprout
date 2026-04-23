"""Alembic migration parser.

Converts an Alembic project (``alembic.ini`` + ``versions/*.py``) into a
``list[SchemaChange]`` via AST parsing only — no migration code is executed.
Also offers an opt-in ``compare_metadata`` wrapper that delegates to
``alembic.autogenerate`` for live-DB comparisons.
"""

from __future__ import annotations

import ast
import collections
import configparser
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dbsprout.migrate.models import SchemaChange
from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import MetaData

logger = logging.getLogger(__name__)

_MAX_REVISION_BYTES = 1024 * 1024  # 1 MB

_UNSET: object = object()


@dataclass(frozen=True)
class AlembicParser:
    """Parse Alembic migration histories into ``SchemaChange`` lists."""

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        """Return the full forward history (base → head) of schema changes."""
        versions_dir = _discover_versions_dir(project_path)
        revisions = _collect_revisions(versions_dir)
        ordered = _linearize_revisions(revisions)
        changes: list[SchemaChange] = []
        for rev in ordered:
            changes.extend(_parse_upgrade(rev))
        return changes

    def compare_metadata(self, db_url: str, metadata: MetaData) -> list[SchemaChange]:
        """Diff a SQLAlchemy MetaData against a live DB via Alembic autogenerate."""
        raise NotImplementedError  # filled in Task 12


def _discover_versions_dir(project_path: Path) -> Path:
    """Resolve the Alembic versions directory.

    Order: ``alembic.ini[alembic].script_location/versions`` →
    ``./alembic/versions`` → ``./migrations/versions``.
    """
    ini = project_path / "alembic.ini"
    if ini.exists():
        cp = configparser.ConfigParser()
        cp.read(ini)
        if cp.has_option("alembic", "script_location"):
            loc = project_path / cp.get("alembic", "script_location") / "versions"
            if loc.is_dir():
                return loc

    for fallback in (
        project_path / "alembic" / "versions",
        project_path / "migrations" / "versions",
    ):
        if fallback.is_dir():
            return fallback

    raise MigrationParseError("No Alembic versions/ directory found", file_path=project_path)


def _collect_revisions(versions_dir: Path) -> list[_Revision]:
    """Parse every ``.py`` file in *versions_dir* into ``_Revision`` records."""
    revs: list[_Revision] = []
    for f in sorted(versions_dir.glob("*.py")):
        if f.name.startswith("__"):
            continue
        if f.stat().st_size > _MAX_REVISION_BYTES:
            raise MigrationParseError(
                f"Revision file too large (> {_MAX_REVISION_BYTES} bytes): {f}",
                file_path=f,
            )
        module = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        rev, down = _extract_revision_ids(module, f)
        revs.append(_Revision(path=f, revision=rev, down_revision=down, module=module))
    return revs


def _extract_revision_ids(module: ast.Module, path: Path) -> tuple[str, str | None]:
    rev: str | None = None
    down: object = _UNSET
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "revision" and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                rev = node.value.value
        elif target.id == "down_revision" and isinstance(node.value, ast.Constant):  # noqa: SIM102
            if node.value.value is None or isinstance(node.value.value, str):
                down = node.value.value
    if rev is None:
        raise MigrationParseError(
            f'Revision file missing `revision = "..."` assignment: {path}',
            file_path=path,
        )
    if down is _UNSET:
        raise MigrationParseError(
            f"Revision file missing `down_revision = ...` assignment: {path}",
            file_path=path,
        )
    return rev, down  # type: ignore[return-value]


def _linearize_revisions(revisions: list[_Revision]) -> list[_Revision]:
    """Walk ``revisions`` from base to head via ``down_revision`` edges."""
    if not revisions:
        return []

    by_id: dict[str, _Revision] = {r.revision: r for r in revisions}
    children: dict[str | None, list[str]] = collections.defaultdict(list)
    for r in revisions:
        children[r.down_revision].append(r.revision)

    roots = [r.revision for r in revisions if r.down_revision is None]
    if len(roots) == 0:
        raise MigrationParseError("No root revision (all revisions reference a down_revision)")
    if len(roots) > 1:
        raise MigrationParseError(f"Multiple root revisions: {sorted(roots)} — expected one")

    heads = [r.revision for r in revisions if not children.get(r.revision)]
    if len(heads) > 1:
        raise MigrationParseError(f"Multiple heads: {sorted(heads)} — merge before parsing")

    ordered: list[_Revision] = []
    current: str | None = roots[0]
    while current is not None:
        ordered.append(by_id[current])
        kids = children.get(current, [])
        if len(kids) > 1:
            raise MigrationParseError(f"Branching at revision {current}: children={sorted(kids)}")
        current = kids[0] if kids else None
    return ordered


OpHandler = Callable[[ast.Call], list[SchemaChange]]
_OP_HANDLERS: dict[str, OpHandler] = {}  # populated in Tasks 7-11


def _parse_upgrade(rev: _Revision) -> list[SchemaChange]:
    assert isinstance(rev.module, ast.Module)
    upgrade_fn = _find_upgrade(rev.module, rev.path)
    changes: list[SchemaChange] = []
    for node in ast.walk(upgrade_fn):
        if isinstance(node, ast.Call) and _is_op_call(node):
            assert isinstance(node.func, ast.Attribute)
            verb = node.func.attr
            handler = _OP_HANDLERS.get(verb)
            if handler is None:
                logger.debug("Skipping unrecognized op.%s in %s", verb, rev.path)
                continue
            changes.extend(handler(node))
    return changes


def _find_upgrade(module: ast.Module, path: Path) -> ast.FunctionDef:
    fns = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "upgrade"]
    if not fns:
        raise MigrationParseError(f"Missing upgrade() function in {path}", file_path=path)
    if len(fns) > 1:
        raise MigrationParseError(f"Multiple upgrade() functions in {path}", file_path=path)
    return fns[0]


def _is_op_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "op"
    )


def _literal(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise MigrationParseError(f"Expected string literal, got {ast.unparse(node)}")


def _literal_list(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise MigrationParseError(f"Expected list/tuple literal, got {ast.unparse(node)}")
    return [_literal(el) for el in node.elts]


def _is_sa_column_call(node: ast.AST) -> bool:
    """Detect ``sa.Column(...)`` or ``Column(...)`` nodes."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name) and node.func.id == "Column":
        return True
    return bool(isinstance(node.func, ast.Attribute) and node.func.attr == "Column")


def _extract_column_spec(node: ast.Call) -> dict[str, object]:
    """Extract ``{name, alembic_type, nullable?, server_default?}`` from ``Column(...)``."""
    spec: dict[str, object] = {}
    args = node.args
    if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        spec["name"] = args[0].value
    else:
        spec["name"] = ast.unparse(args[0]) if args else ""
    if len(args) >= 2:
        spec["alembic_type"] = ast.unparse(args[1])
    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
    if "nullable" in kw:
        try:
            spec["nullable"] = bool(ast.literal_eval(kw["nullable"]))
        except ValueError:
            spec["nullable"] = ast.unparse(kw["nullable"])
    if "server_default" in kw:
        spec["server_default"] = ast.unparse(kw["server_default"])
    return spec


def _handle_create_table(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    name = _literal(node.args[0])
    columns: list[dict[str, object]] = []
    for arg in node.args[1:]:
        if _is_sa_column_call(arg):
            assert isinstance(arg, ast.Call)
            columns.append(_extract_column_spec(arg))
    detail = {"columns": columns} if columns else None
    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=name,
            detail=detail,
        )
    ]


def _handle_drop_table(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    return [
        SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name=_literal(node.args[0]),
        )
    ]


_OP_HANDLERS["create_table"] = _handle_create_table
_OP_HANDLERS["drop_table"] = _handle_drop_table


def _handle_add_column(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    table = _literal(node.args[0])
    col_node = node.args[1]
    if not _is_sa_column_call(col_node):
        raise MigrationParseError(
            f"op.add_column expected Column(...) arg, got {ast.unparse(col_node)}"
        )
    assert isinstance(col_node, ast.Call)
    spec = _extract_column_spec(col_node)
    name = str(spec.pop("name"))
    return [
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name=table,
            column_name=name,
            detail=spec or None,
        )
    ]


def _handle_drop_column(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    return [
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name=_literal(node.args[0]),
            column_name=_literal(node.args[1]),
        )
    ]


_OP_HANDLERS["add_column"] = _handle_add_column
_OP_HANDLERS["drop_column"] = _handle_drop_column


def _handle_alter_column(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    table = _literal(node.args[0])
    column = _literal(node.args[1])
    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
    out: list[SchemaChange] = []
    if "type_" in kw:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                table_name=table,
                column_name=column,
                new_value=ast.unparse(kw["type_"]),
            )
        )
    if "nullable" in kw:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
                table_name=table,
                column_name=column,
                new_value=ast.unparse(kw["nullable"]),
            )
        )
    if "server_default" in kw:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
                table_name=table,
                column_name=column,
                new_value=ast.unparse(kw["server_default"]),
            )
        )
    return out


_OP_HANDLERS["alter_column"] = _handle_alter_column


def _handle_create_foreign_key(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    if len(node.args) < 5:
        raise MigrationParseError(
            "op.create_foreign_key expects (name, source, referent, local_cols, remote_cols)"
        )
    name = _literal(node.args[0])
    source = _literal(node.args[1])
    referent = _literal(node.args[2])
    local_cols = _literal_list(node.args[3])
    remote_cols = _literal_list(node.args[4])
    return [
        SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
            table_name=source,
            detail={
                "name": name,
                "ref_table": referent,
                "local_cols": local_cols,
                "remote_cols": remote_cols,
            },
        )
    ]


def _handle_drop_constraint(node: ast.Call) -> list[SchemaChange]:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
    type_node = kw.get("type_")
    if type_node is None or _safe_literal(type_node) != "foreignkey":
        return []
    name = _literal(node.args[0])
    if len(node.args) >= 2:
        table = _literal(node.args[1])
    elif "table_name" in kw:
        table = _literal(kw["table_name"])
    else:
        raise MigrationParseError(
            "op.drop_constraint requires a table name (positional or table_name=)"
        )
    return [
        SchemaChange(
            change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
            table_name=table,
            detail={"name": name},
        )
    ]


def _safe_literal(node: ast.AST) -> str | None:
    """Return the string value of a Constant node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


_OP_HANDLERS["create_foreign_key"] = _handle_create_foreign_key
_OP_HANDLERS["drop_constraint"] = _handle_drop_constraint


@dataclass(frozen=True)
class _Revision:
    """Internal — parsed revision header + AST for a single Alembic file."""

    path: Path
    revision: str
    down_revision: str | None
    # ast.Module stored loosely as object to keep dataclass frozen-hashable-friendly
    module: object
