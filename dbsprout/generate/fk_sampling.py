"""FK sampling — fill FK columns with valid parent PK references.

Guarantees 100% FK integrity structurally by sampling only from
already-generated parent PK values. Deferred FKs (from cycle
breaking) are skipped — they stay as None until the second pass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from dbsprout.schema.models import ForeignKeySchema, TableSchema

logger = logging.getLogger(__name__)


def sample_fk_values(
    table: TableSchema,
    parent_data: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Fill FK columns in *rows* with valid parent PK references.

    Mutates ``rows`` in-place. Returns the same list.

    For each FK in ``table.foreign_keys``:
    - **Normal FK**: sample from ``parent_data[fk.ref_table]``
    - **Self-referencing FK**: first 20% rows get None, rest reference
      earlier rows from the same table
    - **Deferred FK** (parent not in ``parent_data``): skip (stay None)
    - **Empty parent**: all FK values set to None with a warning
    """
    rng = np.random.default_rng(seed)

    for fk in table.foreign_keys:
        is_self_ref = fk.ref_table == table.name

        if is_self_ref:
            _sample_self_ref_fk(fk, rows, rng)
            continue

        # Deferred FK — parent not yet generated (cycle breaking)
        if fk.ref_table not in parent_data:
            continue

        parent_rows = parent_data[fk.ref_table]

        # Empty parent → None + warning
        if len(parent_rows) == 0:
            logger.warning(
                "Parent table '%s' has 0 rows — FK columns %s in '%s' will be None",
                fk.ref_table,
                fk.columns,
                table.name,
            )
            continue

        if len(fk.columns) == 1:
            _sample_single_fk(fk, parent_rows, rows, rng)
        else:
            _sample_composite_fk(fk, parent_rows, rows, rng)

    return rows


def _sample_single_fk(
    fk: ForeignKeySchema,
    parent_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    rng: Generator,
) -> None:
    """Sample single-column FK values from parent PKs."""
    ref_col = fk.ref_columns[0]
    parent_pks = [r[ref_col] for r in parent_rows]
    pk_array = np.array(parent_pks)
    indices = rng.integers(0, len(pk_array), size=len(rows))

    fk_col = fk.columns[0]
    for i, row in enumerate(rows):
        row[fk_col] = parent_pks[indices[i]]


def _sample_composite_fk(
    fk: ForeignKeySchema,
    parent_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    rng: Generator,
) -> None:
    """Sample composite FK — pick parent row indices, extract all FK columns."""
    indices = rng.integers(0, len(parent_rows), size=len(rows))

    for i, row in enumerate(rows):
        parent_row = parent_rows[indices[i]]
        for fk_col, ref_col in zip(fk.columns, fk.ref_columns, strict=True):
            row[fk_col] = parent_row[ref_col]


def _sample_self_ref_fk(
    fk: ForeignKeySchema,
    rows: list[dict[str, Any]],
    rng: Generator,
) -> None:
    """Self-referencing FK: first 20% None, rest reference earlier rows."""
    num_rows = len(rows)
    null_count = max(num_rows // 5, 1) if num_rows > 0 else 0
    pk_cols = fk.ref_columns
    fk_cols = fk.columns

    for i in range(null_count):
        for col in fk_cols:
            rows[i][col] = None

    for i in range(null_count, num_rows):
        # Sample from rows 0..i-1
        ref_idx = int(rng.integers(0, i))
        ref_row = rows[ref_idx]
        for fk_col, pk_col in zip(fk_cols, pk_cols, strict=True):
            rows[i][fk_col] = ref_row[pk_col]
