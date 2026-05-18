"""Django migration op-handlers and field-snapshot helpers.

All ``_handle_*`` functions and the shared helper utilities live here to keep
``django.py`` within the 500-line cap (spec §12).

Importing this module as a side-effect registers every handler into
``_HANDLERS`` so ``_dispatch_op`` can find them.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

from dbsprout.migrate.models import SchemaChange, SchemaChangeType

if TYPE_CHECKING:
    from dbsprout.migrate.parsers.django import (
        _FieldLedger,
        _ParsedMigration,
        _TableNameLedger,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field snapshot dataclass (lives here; django.py imports it under TYPE_CHECKING)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FieldSnapshot:
    type_name: str
    django_type: str
    base_type: str  # django_type with null= and default= kwargs stripped
    nullable: bool
    default: str | None
    is_fk: bool
    ref_table: str | None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


def _is_str(node: ast.AST | None) -> TypeGuard[ast.Constant]:
    """Return ``True`` (narrowing to ``ast.Constant``) when *node* is a string constant."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _op_name_from_call(op: ast.Call) -> str:
    """Return the bare function name from an ``ast.Call`` node."""
    func = op.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return "<unknown>"


def _resolve_ref(raw: str, *, mig: _ParsedMigration, tables: _TableNameLedger) -> str:
    if "." in raw:
        app, model = raw.split(".", 1)
        return tables.get((app, model), _default_table_name(app, model))
    return tables.get((mig.app_label, raw), _default_table_name(mig.app_label, raw))


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
    type_name = _op_name_from_call(field_call)
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


# ---------------------------------------------------------------------------
# CreateModel + DeleteModel handlers
# ---------------------------------------------------------------------------


def _handle_create_model(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
    kw = _kwargs(op)
    name_node = kw.get("name")
    if not _is_str(name_node) or not isinstance(name_node.value, str):
        logger.debug("CreateModel with non-literal name in %s", mig.path)
        return
    model_name: str = name_node.value

    options = _kwargs_dict(kw.get("options"))
    db_table = options.get("db_table") if options else None
    table_name = db_table or _default_table_name(mig.app_label, model_name)
    tables[(mig.app_label, model_name.lower())] = table_name

    field_nodes = kw.get("fields")
    field_dicts, fk_dicts = _extract_fields(
        field_nodes, mig=mig, model=model_name.lower(), fields_ledger=fields, tables=tables
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
    kw = _kwargs(op)
    name_node = kw.get("name")
    if not _is_str(name_node) or not isinstance(name_node.value, str):
        return
    model_name: str = name_node.value.lower()
    table_name = tables.pop(
        (mig.app_label, model_name), _default_table_name(mig.app_label, model_name)
    )
    # Clean up field ledger entries under this model.
    for key in [k for k in fields if k[0] == mig.app_label and k[1] == model_name]:
        fields.pop(key)
    out.append(SchemaChange(change_type=SchemaChangeType.TABLE_REMOVED, table_name=table_name))


# ---------------------------------------------------------------------------
# AddField handler (plain, FK, M2M)
# ---------------------------------------------------------------------------


def _emit_m2m_through(
    field_node: ast.Call,
    *,
    mig: _ParsedMigration,
    model_col_pair: tuple[str, str],
    tables: _TableNameLedger,
    out: list[SchemaChange],
) -> None:
    """Emit implicit through-table TABLE_ADDED for a ManyToManyField."""
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


def _handle_add_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
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
    model_name: str = model_node.value.lower()
    column_name: str = name_node.value

    if _op_name_from_call(field_node) == "ManyToManyField":
        _emit_m2m_through(
            field_node,
            mig=mig,
            model_col_pair=(model_name, column_name),
            tables=tables,
            out=out,
        )
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


# ---------------------------------------------------------------------------
# RemoveField handler
# ---------------------------------------------------------------------------


def _handle_remove_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
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
    model_name: str = model_node.value.lower()
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


# ---------------------------------------------------------------------------
# AlterField handler with per-dimension ledger diffing
# ---------------------------------------------------------------------------


def _handle_alter_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
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
    model_name: str = model_node.value.lower()
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


# ---------------------------------------------------------------------------
# RenameField + RenameModel handlers
# ---------------------------------------------------------------------------


def _handle_rename_field(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,
    out: list[SchemaChange],
) -> None:
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
    model_name: str = model_node.value.lower()
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
    old: str = old_node.value.lower()
    new: str = new_node.value.lower()
    old_table = tables.pop((mig.app_label, old), _default_table_name(mig.app_label, old))
    # Preserve explicit db_table override; otherwise derive from new model name.
    if old_table != _default_table_name(mig.app_label, old):
        new_table = old_table
    else:
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


# ---------------------------------------------------------------------------
# AddIndex + RemoveIndex handlers
# ---------------------------------------------------------------------------


def _handle_add_index(
    op: ast.Call,
    *,
    mig: _ParsedMigration,
    tables: _TableNameLedger,
    fields: _FieldLedger,  # noqa: ARG001
    out: list[SchemaChange],
) -> None:
    kw = _kwargs(op)
    model_node = kw.get("model_name")
    index_node = kw.get("index")
    if not isinstance(model_node, ast.Constant) or not isinstance(model_node.value, str):
        return
    if not isinstance(index_node, ast.Call):
        return
    model_name: str = model_node.value.lower()
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
    model_name: str = model_node.value.lower()
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


# ---------------------------------------------------------------------------
# Register handlers — must remain at bottom after all handler definitions.
# ---------------------------------------------------------------------------

from dbsprout.migrate.parsers.django import _HANDLERS  # noqa: E402

_HANDLERS["CreateModel"] = _handle_create_model
_HANDLERS["DeleteModel"] = _handle_delete_model
_HANDLERS["AddField"] = _handle_add_field
_HANDLERS["RemoveField"] = _handle_remove_field
_HANDLERS["AlterField"] = _handle_alter_field
_HANDLERS["RenameField"] = _handle_rename_field
_HANDLERS["RenameModel"] = _handle_rename_model
_HANDLERS["AddIndex"] = _handle_add_index
_HANDLERS["RemoveIndex"] = _handle_remove_index
