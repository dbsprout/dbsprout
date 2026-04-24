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

from dbsprout.migrate.models import SchemaChange
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
    type_name: str
    django_type: str
    base_type: str  # django_type with null= and default= kwargs stripped
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


# ---------------------------------------------------------------------------
# Task 7: CreateModel + DeleteModel handlers
# ---------------------------------------------------------------------------


def _handle_create_model(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    name_node = kw.get("name")
    if not _is_str(name_node):
        logger.debug("CreateModel with non-literal name in %s", mig.path)
        return
    model_name: str = name_node.value  # type: ignore[union-attr]

    options = _kwargs_dict(kw.get("options"))
    db_table = options.get("db_table") if options else None
    table_name = db_table or _default_table_name(mig.app_label, model_name)
    tables[(mig.app_label, model_name)] = table_name

    field_nodes = kw.get("fields")
    field_dicts, fk_dicts = _extract_fields(
        field_nodes, mig=mig, model=model_name, fields_ledger=fields, tables=tables
    )

    out.append(
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=table_name,
            detail={
                "fields": field_dicts,
                "foreign_keys": fk_dicts,
                "db_table": db_table,
            },
        ),
    )
    # Emit standalone FOREIGN_KEY_ADDED for each FK defined in the initial fields.
    for fk in fk_dicts:
        col = fk["column"]
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
                table_name=table_name,
                column_name=str(col),
                detail={
                    "ref_table": fk["ref_table"],
                    "local_cols": fk["local_cols"],
                    "remote_cols": fk["remote_cols"],
                },
            ),
        )


def _handle_delete_model(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    name_node = kw.get("name")
    if not _is_str(name_node):
        return
    model_name: str = name_node.value  # type: ignore[union-attr]
    table_name = tables.pop(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    # Clean up field ledger entries under this model.
    for key in [k for k in fields if k[0] == mig.app_label and k[1] == model_name]:
        fields.pop(key)
    out.append(SchemaChange(change_type=SchemaChangeType.TABLE_REMOVED, table_name=table_name))


_HANDLERS["CreateModel"] = _handle_create_model
_HANDLERS["DeleteModel"] = _handle_delete_model


def _default_table_name(app_label: str, model_name: str) -> str:
    return f"{app_label}_{model_name.lower()}"


def _kwargs(op: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in op.keywords if kw.arg is not None}


def _kwargs_dict(node: ast.AST | None) -> dict[str, str | None]:
    if not isinstance(node, ast.Dict):
        return {}
    result: dict[str, str | None] = {}
    for k, v in zip(node.keys, node.values, strict=False):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            raw = v.value if isinstance(v, ast.Constant) else None
            result[k.value] = str(raw) if isinstance(raw, str) else None
    return result


def _is_str(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _extract_fields(
    fields_node: ast.AST | None,
    *,
    mig: _ParsedMigration,
    model: str,
    fields_ledger: _FieldLedger,
    tables: _TableNameLedger,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    field_dicts: list[dict[str, object]] = []
    fk_dicts: list[dict[str, object]] = []
    if not isinstance(fields_node, (ast.List, ast.Tuple)):
        return field_dicts, fk_dicts
    for tup in fields_node.elts:
        if not isinstance(tup, ast.Tuple) or len(tup.elts) != 2:
            continue
        name_node, field_call = tup.elts
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            continue
        if not isinstance(field_call, ast.Call):
            continue
        col_name: str = name_node.value
        snapshot = _field_snapshot(field_call, mig=mig, tables=tables)
        fields_ledger[(mig.app_label, model, col_name)] = snapshot
        field_dicts.append(
            {
                "name": col_name,
                "django_type": snapshot.django_type,
                "nullable": snapshot.nullable,
                "default": snapshot.default,
            },
        )
        if snapshot.is_fk and snapshot.ref_table:
            fk_dicts.append(
                {
                    "column": col_name,
                    "ref_table": snapshot.ref_table,
                    "local_cols": [col_name],
                    "remote_cols": ["id"],
                },
            )
    return field_dicts, fk_dicts


def _field_snapshot(
    field_call: ast.Call, *, mig: _ParsedMigration, tables: _TableNameLedger
) -> _FieldSnapshot:
    type_name = _op_name(field_call)
    kw = _kwargs(field_call)
    django_type = ast.unparse(field_call)
    # base_type: rebuild the call with null= and default= stripped so that
    # param-only changes (e.g. max_length=200 → max_length=300) are detectable.
    stripped_keywords = [
        kw_node for kw_node in field_call.keywords if kw_node.arg not in {"null", "default"}
    ]
    stripped_call = ast.Call(func=field_call.func, args=field_call.args, keywords=stripped_keywords)
    base_type = ast.unparse(stripped_call)
    nullable_node = kw.get("null")
    nullable = isinstance(nullable_node, ast.Constant) and nullable_node.value is True
    default_node = kw.get("default")
    default = ast.unparse(default_node) if default_node is not None else None
    is_fk = type_name in {"ForeignKey", "OneToOneField"}
    ref_table: str | None = None
    if is_fk and field_call.args:
        first = field_call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            ref_table = _resolve_ref(first.value, mig=mig, tables=tables)
    return _FieldSnapshot(
        type_name=type_name,
        django_type=django_type,
        base_type=base_type,
        nullable=nullable,
        default=default,
        is_fk=is_fk,
        ref_table=ref_table,
    )


def _resolve_ref(raw: str, *, mig: _ParsedMigration, tables: _TableNameLedger) -> str:
    if "." in raw:
        app, model = raw.split(".", 1)
        return tables.get((app, model), _default_table_name(app, model))
    return tables.get((mig.app_label, raw), _default_table_name(mig.app_label, raw))


# ---------------------------------------------------------------------------
# Task 8: AddField handler (plain, FK, M2M)
# ---------------------------------------------------------------------------


def _handle_add_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    name_node = kw.get("name")
    field_node = kw.get("field")
    if not (
        isinstance(model_node, ast.Constant)
        and isinstance(model_node.value, str)
        and isinstance(name_node, ast.Constant)
        and isinstance(name_node.value, str)
        and isinstance(field_node, ast.Call)
    ):
        logger.debug("AddField with non-literal args in %s", mig.path)
        return
    model_name: str = model_node.value
    column_name: str = name_node.value

    if _op_name(field_node) == "ManyToManyField":
        _emit_m2m_through(field_node, mig, (model_name, column_name), tables, out)
        return

    snap = _field_snapshot(field_node, mig=mig, tables=tables)
    fields[(mig.app_label, model_name, column_name)] = snap
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name=table_name,
            column_name=column_name,
            detail={
                "django_type": snap.django_type,
                "nullable": snap.nullable,
                "default": snap.default,
            },
        ),
    )
    if snap.is_fk and snap.ref_table:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.FOREIGN_KEY_ADDED,
                table_name=table_name,
                column_name=column_name,
                detail={
                    "ref_table": snap.ref_table,
                    "local_cols": [column_name],
                    "remote_cols": ["id"],
                },
            ),
        )


def _emit_m2m_through(
    field_node: ast.Call,
    mig: _ParsedMigration,
    model_col_pair: tuple[str, str],
    tables: _TableNameLedger,
    out: list[SchemaChange],
) -> None:
    """Emit implicit through-table TABLE_ADDED for a ManyToManyField."""
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    model, column = model_col_pair
    # Skip explicit through=... — user's own CreateModel covers it.
    if any(kw.arg == "through" for kw in field_node.keywords):
        return
    this_table = tables.get((mig.app_label, model), _default_table_name(mig.app_label, model))
    # Resolve target model from first positional arg.
    target_raw: str | None = None
    if (
        field_node.args
        and isinstance(field_node.args[0], ast.Constant)
        and isinstance(field_node.args[0].value, str)
    ):
        target_raw = field_node.args[0].value
    if target_raw is None:
        logger.debug("M2M without literal target in %s", mig.path)
        return
    target_table = _resolve_ref(target_raw, mig=mig, tables=tables)
    through_name = f"{mig.app_label}_{model.lower()}_{column}"
    model_col = f"{model.lower()}_id"
    target_col = f"{column}_id"
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=through_name,
            detail={
                "fields": [
                    {
                        "name": "id",
                        "django_type": "models.AutoField(primary_key=True)",
                        "nullable": False,
                        "default": None,
                    },
                    {
                        "name": model_col,
                        "django_type": "models.ForeignKey",
                        "nullable": False,
                        "default": None,
                    },
                    {
                        "name": target_col,
                        "django_type": "models.ForeignKey",
                        "nullable": False,
                        "default": None,
                    },
                ],
                "foreign_keys": [
                    {
                        "column": model_col,
                        "ref_table": this_table,
                        "local_cols": [model_col],
                        "remote_cols": ["id"],
                    },
                    {
                        "column": target_col,
                        "ref_table": target_table,
                        "local_cols": [target_col],
                        "remote_cols": ["id"],
                    },
                ],
                "db_table": None,
            },
        ),
    )


_HANDLERS["AddField"] = _handle_add_field


# ---------------------------------------------------------------------------
# Task 9: RemoveField handler
# ---------------------------------------------------------------------------


def _handle_remove_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    name_node = kw.get("name")
    if not (
        isinstance(model_node, ast.Constant)
        and isinstance(model_node.value, str)
        and isinstance(name_node, ast.Constant)
        and isinstance(name_node.value, str)
    ):
        return
    model_name: str = model_node.value
    column_name: str = name_node.value
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    prev = fields.pop((mig.app_label, model_name, column_name), None)
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name=table_name,
            column_name=column_name,
        ),
    )
    if prev is not None and prev.is_fk:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.FOREIGN_KEY_REMOVED,
                table_name=table_name,
                column_name=column_name,
                detail={"ref_table": prev.ref_table},
            ),
        )


_HANDLERS["RemoveField"] = _handle_remove_field


# ---------------------------------------------------------------------------
# Task 10: AlterField handler with per-dimension ledger diffing
# ---------------------------------------------------------------------------


def _handle_alter_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    name_node = kw.get("name")
    field_node = kw.get("field")
    if not (
        isinstance(model_node, ast.Constant)
        and isinstance(model_node.value, str)
        and isinstance(name_node, ast.Constant)
        and isinstance(name_node.value, str)
        and isinstance(field_node, ast.Call)
    ):
        return
    model_name: str = model_node.value
    column_name: str = name_node.value
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    new_snap = _field_snapshot(field_node, mig=mig, tables=tables)

    key = (mig.app_label, model_name, column_name)
    prev = fields.get(key)
    fields[key] = new_snap

    if prev is None:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                table_name=table_name,
                column_name=column_name,
                new_value=new_snap.django_type,
                detail={"django_type": new_snap.django_type},
            ),
        )
        return

    if new_snap.base_type != prev.base_type:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_TYPE_CHANGED,
                table_name=table_name,
                column_name=column_name,
                old_value=prev.django_type,
                new_value=new_snap.django_type,
                detail={"django_type": new_snap.django_type},
            ),
        )
    if new_snap.nullable != prev.nullable:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_NULLABILITY_CHANGED,
                table_name=table_name,
                column_name=column_name,
                old_value=str(prev.nullable),
                new_value=str(new_snap.nullable),
                detail={"django_type": new_snap.django_type},
            ),
        )
    if new_snap.default != prev.default:
        out.append(
            SchemaChange(
                change_type=SchemaChangeType.COLUMN_DEFAULT_CHANGED,
                table_name=table_name,
                column_name=column_name,
                old_value=prev.default,
                new_value=new_snap.default,
                detail={"django_type": new_snap.django_type},
            ),
        )


_HANDLERS["AlterField"] = _handle_alter_field


# ---------------------------------------------------------------------------
# Task 11: RenameField + RenameModel handlers
# ---------------------------------------------------------------------------


def _handle_rename_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    old_node = kw.get("old_name")
    new_node = kw.get("new_name")
    if not (
        isinstance(model_node, ast.Constant)
        and isinstance(model_node.value, str)
        and isinstance(old_node, ast.Constant)
        and isinstance(old_node.value, str)
        and isinstance(new_node, ast.Constant)
        and isinstance(new_node.value, str)
    ):
        return
    model_name: str = model_node.value
    old: str = old_node.value
    new: str = new_node.value
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    rename_detail: dict[str, object] = {"rename_of": {"old": old, "new": new}}
    snap = fields.pop((mig.app_label, model_name, old), None)
    if snap is not None:
        fields[(mig.app_label, model_name, new)] = snap
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_REMOVED,
            table_name=table_name,
            column_name=old,
            detail=rename_detail,
        ),
    )
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.COLUMN_ADDED,
            table_name=table_name,
            column_name=new,
            detail=rename_detail,
        ),
    )


def _handle_rename_model(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    old_node = kw.get("old_name")
    new_node = kw.get("new_name")
    if not (
        isinstance(old_node, ast.Constant)
        and isinstance(old_node.value, str)
        and isinstance(new_node, ast.Constant)
        and isinstance(new_node.value, str)
    ):
        return
    old: str = old_node.value
    new: str = new_node.value
    old_table = tables.pop((mig.app_label, old), _default_table_name(mig.app_label, old))
    new_table = _default_table_name(mig.app_label, new)
    tables[(mig.app_label, new)] = new_table
    # Move field-ledger entries to the new model name.
    for key in [k for k in fields if k[0] == mig.app_label and k[1] == old]:
        fields[(mig.app_label, new, key[2])] = fields.pop(key)
    rename_detail: dict[str, object] = {"rename_of": {"old": old_table, "new": new_table}}
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.TABLE_REMOVED,
            table_name=old_table,
            detail=rename_detail,
        )
    )
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.TABLE_ADDED,
            table_name=new_table,
            detail=rename_detail,
        )
    )


_HANDLERS["RenameField"] = _handle_rename_field
_HANDLERS["RenameModel"] = _handle_rename_model


# ---------------------------------------------------------------------------
# Task 12: AddIndex + RemoveIndex handlers
# ---------------------------------------------------------------------------


def _handle_add_index(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,  # noqa: ARG001
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    index_node = kw.get("index")
    if not isinstance(model_node, ast.Constant) or not isinstance(model_node.value, str):
        return
    if not isinstance(index_node, ast.Call):
        return
    model_name: str = model_node.value
    index_kw = _kwargs(index_node)
    cols_node = index_kw.get("fields")
    cols: list[str] = []
    if isinstance(cols_node, (ast.List, ast.Tuple)):
        cols = [
            e.value
            for e in cols_node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    name_kw = index_kw.get("name")
    index_name = (
        name_kw.value
        if isinstance(name_kw, ast.Constant) and isinstance(name_kw.value, str)
        else None
    )
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.INDEX_ADDED,
            table_name=table_name,
            detail={"cols": cols, "index_name": index_name},
        ),
    )


def _handle_remove_index(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,  # noqa: ARG001
    out: list[SchemaChange],
) -> None:
    from dbsprout.migrate.models import SchemaChangeType  # noqa: PLC0415

    kw = _kwargs(op)
    model_node = kw.get("model_name")
    name_node = kw.get("name")
    if not (
        isinstance(model_node, ast.Constant)
        and isinstance(model_node.value, str)
        and isinstance(name_node, ast.Constant)
        and isinstance(name_node.value, str)
    ):
        return
    model_name: str = model_node.value
    index_name: str = name_node.value
    table_name = tables.get(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    out.append(
        SchemaChange(
            change_type=SchemaChangeType.INDEX_REMOVED,
            table_name=table_name,
            detail={"index_name": index_name},
        ),
    )


_HANDLERS["AddIndex"] = _handle_add_index
_HANDLERS["RemoveIndex"] = _handle_remove_index
