"""Django migration parser.

Converts a Django project (``<app>/migrations/*.py``) into a
``list[SchemaChange]`` via AST parsing only — no migration code executes,
no Django runtime dependency.
"""

from __future__ import annotations

import ast
import graphlib
import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dbsprout.migrate.models import SchemaChange  # noqa: TC001
from dbsprout.migrate.parsers import MigrationParseError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_MIGRATION_BYTES = 1024 * 1024  # 1 MB

_PREFIX_RE = re.compile(r"^(\d+)_")


@dataclass(frozen=True)
class _ParsedMigration:
    path: Path
    app_label: str
    name: str
    prefix: int
    dependencies: tuple[tuple[str, str], ...]
    operations: tuple[ast.Call, ...]


@dataclass(frozen=True)
class DjangoMigrationParser:
    """Parse Django migration histories into ``SchemaChange`` lists."""

    def detect_changes(self, project_path: Path) -> list[SchemaChange]:
        files = _discover_migration_files(project_path)
        if not files:
            raise MigrationParseError(
                f"no */migrations/*.py found under {project_path}",
            )
        parsed: list[_ParsedMigration] = []
        for path in files:
            app_label = path.parent.parent.name
            logger.debug("using directory name %s as app_label for %s", app_label, path)
            mig = _parse_migration_file(path, app_label=app_label)
            if mig is not None:
                parsed.append(mig)
        ordered = _linearize_migrations(parsed)
        return _translate_operations(ordered)


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


def _parse_migration_file(path: Path, *, app_label: str) -> _ParsedMigration | None:
    """Return a parsed migration, or ``None`` if the file has no ``Migration`` class."""
    stem = path.stem
    prefix_match = _PREFIX_RE.match(stem)
    if not prefix_match:
        raise MigrationParseError(
            f"migration filename missing numeric prefix: {stem}",
            file_path=path,
        )
    prefix = int(prefix_match.group(1))

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MigrationParseError("unreadable migration file", file_path=path) from exc

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MigrationParseError("unparseable migration file", file_path=path) from exc

    migration_cls: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Migration":
            migration_cls = node
            break
    if migration_cls is None:
        logger.debug("no Migration class in %s — skipping", path)
        return None

    deps = _extract_dependencies(migration_cls)
    ops = _extract_operations(migration_cls)
    return _ParsedMigration(
        path=path,
        app_label=app_label,
        name=stem,
        prefix=prefix,
        dependencies=deps,
        operations=ops,
    )


def _extract_dependencies(cls_node: ast.ClassDef) -> tuple[tuple[str, str], ...]:
    value = _find_class_assign_value(cls_node, "dependencies")
    if not isinstance(value, (ast.List, ast.Tuple)):
        return ()
    result: list[tuple[str, str]] = []
    for elt in value.elts:
        if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
            logger.debug("non-tuple dependency element; skipping")
            continue
        app_node, name_node = elt.elts
        if not (isinstance(app_node, ast.Constant) and isinstance(name_node, ast.Constant)):
            logger.debug("non-literal dependency element; skipping")
            continue
        if not (isinstance(app_node.value, str) and isinstance(name_node.value, str)):
            continue
        result.append((app_node.value, name_node.value))
    return tuple(result)


def _extract_operations(cls_node: ast.ClassDef) -> tuple[ast.Call, ...]:
    value = _find_class_assign_value(cls_node, "operations")
    if not isinstance(value, (ast.List, ast.Tuple)):
        return ()
    return tuple(elt for elt in value.elts if isinstance(elt, ast.Call))


def _find_class_assign_value(cls_node: ast.ClassDef, target_name: str) -> ast.expr | None:
    """Return the RHS expression of ``target_name = ...`` inside ``cls_node``.

    Handles both plain ``ast.Assign`` and annotated ``ast.AnnAssign`` forms.
    Forward-declaration ``AnnAssign`` (``name: type`` with no value) returns ``None``.
    """
    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    return stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if isinstance(target, ast.Name) and target.id == target_name and stmt.value is not None:
                return stmt.value
    return None


def _linearize_migrations(parsed: list[_ParsedMigration]) -> list[_ParsedMigration]:
    """Topologically sort migrations across apps, with in-app prefix order."""
    # Bucket by app, sort each bucket by numeric prefix (non-mutating rebind).
    buckets: dict[str, list[_ParsedMigration]] = defaultdict(list)
    for mig in parsed:
        buckets[mig.app_label].append(mig)
    by_app: dict[str, list[_ParsedMigration]] = {
        app: sorted(migs, key=lambda m: m.prefix) for app, migs in buckets.items()
    }

    # Duplicate-prefix detection on sorted buckets.
    for app, migs in by_app.items():
        for prev, curr in itertools.pairwise(migs):
            if prev.prefix == curr.prefix:
                raise MigrationParseError(
                    f"duplicate migration prefix {curr.prefix} in app {app}: "
                    f"{prev.path}, {curr.path}",
                    file_path=prev.path,
                )

    known: dict[tuple[str, str], _ParsedMigration] = {(m.app_label, m.name): m for m in parsed}

    # Pre-compute position within sorted siblings to avoid O(n) list.index scans.
    position: dict[tuple[str, str], int] = {
        (mig.app_label, mig.name): idx for migs in by_app.values() for idx, mig in enumerate(migs)
    }

    # Deterministic insertion order: (app_label, prefix).
    sorter: graphlib.TopologicalSorter[tuple[str, str]] = graphlib.TopologicalSorter()
    for key in sorted(known, key=lambda k: (k[0], known[k].prefix)):
        mig = known[key]
        siblings = by_app[mig.app_label]
        idx = position[(mig.app_label, mig.name)]
        intra = [(mig.app_label, siblings[idx - 1].name)] if idx > 0 else []
        cross = [dep for dep in mig.dependencies if dep in known]
        sorter.add(key, *intra, *cross)

    try:
        ordered_keys = list(sorter.static_order())
    except graphlib.CycleError as exc:
        cycle = " ↔ ".join(f"{a}:{n}" for a, n in exc.args[1][:2])
        raise MigrationParseError(f"cycle in migration dependencies: {cycle}") from exc

    return [known[k] for k in ordered_keys]


# ---------------------------------------------------------------------------
# Op → SchemaChange translation
# ---------------------------------------------------------------------------

_TableNameLedger = dict[tuple[str, str], str]  # (app, model) → table_name
_FieldLedger = dict[tuple[str, str, str], "_FieldSnapshot"]  # (app, model, field) → snapshot


@dataclass(frozen=True)
class _FieldSnapshot:
    django_type: str
    nullable: bool
    default: str | None
    is_fk: bool
    ref_table: str | None


def _translate_operations(ordered: list[_ParsedMigration]) -> list[SchemaChange]:
    """Walk ordered migrations, emit SchemaChange list."""
    table_names: _TableNameLedger = {}
    field_state: _FieldLedger = {}
    changes: list[SchemaChange] = []
    for mig in ordered:
        for op in mig.operations:
            _dispatch_op(op, mig=mig, tables=table_names, fields=field_state, out=changes)
    return changes


def _dispatch_op(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    op_name = _op_name(op)
    handler = _HANDLERS.get(op_name)
    if handler is None:
        logger.debug("skipping unsupported op %s in %s", op_name, mig.path)
        return
    handler(op, mig=mig, tables=tables, fields=fields, out=out)


def _op_name(op: ast.Call) -> str:
    func = op.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return "<unknown>"


_OpHandler = Callable[..., None]  # runtime alias used in _HANDLERS below
_HANDLERS: dict[str, _OpHandler] = {}  # populated in subsequent tasks
