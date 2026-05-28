"""Referential-integrity guard for in-place spec edits (S-119).

The web spec-edit endpoint (``PUT /api/spec/tables/{t}/columns/{c}``) lets the
user retune any column's generator config. Most edits are harmless — the
:class:`~dbsprout.spec.models.GeneratorConfig` is already validated by Pydantic
at the request boundary — but a small set of changes would break the contract
the downstream generator relies on:

* A primary-key column whose generator stops being ``unique`` would create
  PK violations the moment two rows collide.
* A column referenced by a foreign key (the *target* side) must remain unique
  for the same reason: the dependent table samples its FK values from these
  PKs at generation time, so duplicates corrupt the join.

The guard is a pure, schema-only function. The router calls it after Pydantic
validation but before reaching the workspace mutator, so the failure path
short-circuits with a ``409 CONSTRAINT_VIOLATION`` envelope.

Anything that isn't a PK or an FK-target column is freely editable; the
*source* side of an FK (e.g. ``orders.user_id``) is sampled from the parent
PK at run time, so its ``unique`` flag isn't load-bearing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import GeneratorConfig


def _fk_target_columns(schema: DatabaseSchema, table_name: str) -> set[str]:
    """Return the set of *target* column names referenced by some FK.

    A column is a FK-target when *any* table in the schema has a foreign key
    whose ``ref_table`` matches ``table_name`` and whose ``ref_columns``
    contain the column. Self-references count.
    """
    targets: set[str] = set()
    for table in schema.tables:
        for fk in table.foreign_keys:
            if fk.ref_table == table_name:
                targets.update(fk.ref_columns)
    return targets


def check_column_update(
    schema: DatabaseSchema,
    table_name: str,
    column_name: str,
    new_config: GeneratorConfig,
) -> str | None:
    """Return a rejection reason, or ``None`` when the change is safe.

    Args:
        schema: the loaded database schema.
        table_name: target table for the update.
        column_name: column being edited.
        new_config: the (already Pydantic-validated) replacement
            :class:`~dbsprout.spec.models.GeneratorConfig`.

    Returns:
        ``None`` when the change is allowed. Otherwise a single-line
        human-readable reason naming the violated invariant — suitable for
        echoing back to the user inside the ``CONSTRAINT_VIOLATION`` envelope.
    """
    table = schema.get_table(table_name)
    if table is None:
        return f"unknown table {table_name!r}"
    column = table.get_column(column_name)
    if column is None:
        return f"unknown column {column_name!r} on table {table_name!r}"

    # PK guard fires first — deterministic ordering means the message is stable
    # for PK columns that are *also* FK targets.
    if column_name in table.primary_key and not new_config.unique:
        return (
            f"column {table_name}.{column_name} is part of the primary key; "
            "the generator must produce unique values (set unique=true)."
        )

    fk_targets = _fk_target_columns(schema, table_name)
    if column_name in fk_targets and not new_config.unique:
        return (
            f"column {table_name}.{column_name} is referenced by a foreign key; "
            "the generator must produce unique values (set unique=true)."
        )

    return None


__all__ = ["check_column_update"]
