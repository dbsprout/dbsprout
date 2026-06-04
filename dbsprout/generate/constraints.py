"""Constraint enforcement — UNIQUE dedup + NOT NULL + autoincrement PK.

Runs as a post-processing step after initial generation and FK sampling.
Ensures generated data never violates schema constraints.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from dbsprout.generate.check_parser import CheckConstraint
    from dbsprout.schema.models import ColumnSchema, TableSchema

from dbsprout.schema.models import ColumnType

_MAX_RETRIES = 10
_INTEGER_TYPES = frozenset({ColumnType.INTEGER, ColumnType.BIGINT, ColumnType.SMALLINT})
_ASCII_LOWER_CHARS = list(string.ascii_lowercase)


class ConstraintError(Exception):
    """Raised when a constraint cannot be satisfied after retries."""

    def __init__(self, table: str, column: str, constraint: str, attempts: int) -> None:
        self.table = table
        self.column = column
        self.constraint = constraint
        self.attempts = attempts
        super().__init__(
            f"Cannot satisfy {constraint} constraint on {table}.{column} after {attempts} attempts"
        )


def enforce_constraints(
    table: TableSchema,
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Enforce UNIQUE, NOT NULL, and PK constraints on generated rows.

    Returns a **new list** of row dicts. Does not mutate the input.
    """
    result = [{**row} for row in rows]
    rng = np.random.default_rng(seed)
    fk_cols = _fk_columns(table)

    _assign_autoincrement_pks(table, result)
    _enforce_unique(table, result, rng, fk_cols)
    _enforce_not_null(table, result, rng, fk_cols)
    _enforce_check(table, result, rng)

    return result


def _assign_autoincrement_pks(
    table: TableSchema,
    rows: list[dict[str, Any]],
) -> None:
    """Assign sequential 1, 2, 3, ... for auto-increment PK columns."""
    for col in table.columns:
        if col.primary_key and col.autoincrement:
            for i, row in enumerate(rows):
                row[col.name] = i + 1


def _enforce_unique(
    table: TableSchema,
    rows: list[dict[str, Any]],
    rng: Generator,
    fk_cols: set[str],
) -> None:
    """Enforce single-column and composite UNIQUE constraints."""
    # Single-column UNIQUE (from ColumnSchema.unique or single-column PK).
    # The PK case keys off ``table.primary_key`` — the same source the
    # integrity validator uses — rather than the per-column ``primary_key``
    # flag, so a schema where the two disagree (e.g. a stale snapshot with
    # ``table.primary_key=['id']`` but ``col.primary_key`` unset) is still
    # deduplicated instead of silently emitting duplicate PKs.
    single_col_pk = table.primary_key[0] if len(table.primary_key) == 1 else None
    for col in table.columns:
        if col.name in fk_cols:
            continue
        if col.autoincrement:
            continue
        if not (col.unique or col.name == single_col_pk):
            continue
        _dedup_single_column(table.name, col, rows, rng)

    # Composite UNIQUE indexes. Junction tables (all-FK composite keys) are
    # NOT skipped: FK sampling can draw the same parent pair for two rows, so
    # the dedup must run. Colliding FK columns are re-sampled FK-aware (from
    # the valid parent-PK pool) inside ``_dedup_composite``.
    for idx in table.indexes:
        if not idx.unique or len(idx.columns) <= 1:
            continue
        _dedup_composite(table, idx.columns, rows, rng, fk_cols)

    # Composite PK as implicit UNIQUE — runs for all-FK junction-table PKs too.
    if len(table.primary_key) > 1:
        _dedup_composite(table, table.primary_key, rows, rng, fk_cols)


def _dedup_single_column(
    table_name: str,
    col: ColumnSchema,
    rows: list[dict[str, Any]],
    rng: Generator,
) -> None:
    """Remove duplicates from a single UNIQUE column via regeneration."""
    seen: set[Any] = set()
    for row in rows:
        val = row[col.name]
        if val in seen:
            for _attempt in range(_MAX_RETRIES):
                new_val = _regenerate_value(col, rng)
                if new_val not in seen:
                    row[col.name] = new_val
                    val = new_val
                    break
            else:
                raise ConstraintError(
                    table=table_name,
                    column=col.name,
                    constraint="UNIQUE",
                    attempts=_MAX_RETRIES,
                )
        seen.add(val)


def _dedup_composite(
    table: TableSchema,
    col_names: list[str],
    rows: list[dict[str, Any]],
    rng: Generator,
    fk_cols: set[str],
) -> None:
    """Remove duplicate tuples for composite UNIQUE / PK constraints.

    FK-aware: a colliding FK column is re-sampled from its *valid parent-PK
    pool* — the distinct values already present in that column after FK
    sampling, which is exactly the set of parent PKs that were drawn. This
    keeps FK integrity intact while driving the tuple toward a new unique
    combination (junction tables whose entire key is FK columns). Non-FK
    columns keep the type-appropriate ``_regenerate_value`` fallback. When the
    combination space is genuinely exhausted (rows > distinct parent combos),
    ``_MAX_RETRIES`` is hit and a :class:`ConstraintError` is raised rather
    than emitting a silent duplicate.
    """
    seen: set[tuple[Any, ...]] = set()
    col_name_set = set(col_names)
    col_schemas = [c for c in table.columns if c.name in col_name_set]
    fk_pools = _fk_value_pools(col_names, fk_cols, rows)

    for row in rows:
        tup = tuple(row[c] for c in col_names)
        if tup in seen:
            for _attempt in range(_MAX_RETRIES):
                for col_schema in col_schemas:
                    if col_schema.autoincrement:
                        continue
                    pool = fk_pools.get(col_schema.name)
                    if pool is not None:
                        row[col_schema.name] = pool[int(rng.integers(0, len(pool)))]
                    else:
                        row[col_schema.name] = _regenerate_value(col_schema, rng)
                tup = tuple(row[c] for c in col_names)
                if tup not in seen:
                    break
            else:
                raise ConstraintError(
                    table=table.name,
                    column=", ".join(col_names),
                    constraint="UNIQUE (composite)",
                    attempts=_MAX_RETRIES,
                )
        seen.add(tup)


def _fk_value_pools(
    col_names: list[str],
    fk_cols: set[str],
    rows: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Map each FK column in *col_names* to its valid parent-PK pool.

    The pool is the sorted distinct non-None values present in that column
    after FK sampling — i.e. the parent PKs already drawn — so re-sampling
    from it can never invent a value that breaks FK integrity. Columns with an
    empty pool (parent had 0 rows → all None) are omitted, so the caller falls
    back to ``_regenerate_value`` instead of sampling an empty sequence.
    """
    pools: dict[str, list[Any]] = {}
    for name in col_names:
        if name not in fk_cols:
            continue
        values = sorted({row[name] for row in rows if row[name] is not None})
        if values:
            pools[name] = values
    return pools


def _enforce_not_null(
    table: TableSchema,
    rows: list[dict[str, Any]],
    rng: Generator,
    fk_cols: set[str],
) -> None:
    """Replace None values in non-nullable columns."""
    for col in table.columns:
        if col.nullable:
            continue
        if col.name in fk_cols:
            continue
        if col.autoincrement:
            continue
        for row in rows:
            if row[col.name] is None:
                row[col.name] = _regenerate_value(col, rng)


def _regenerate_value(col: ColumnSchema, rng: Generator) -> Any:
    """Generate a type-appropriate fallback value."""
    dt = col.data_type
    if dt in _INTEGER_TYPES:
        return int(rng.integers(0, 2**31))
    if dt in (ColumnType.FLOAT, ColumnType.DECIMAL):
        return _regen_numeric(col, rng)
    if dt == ColumnType.BOOLEAN:
        return bool(rng.integers(0, 2))
    if dt in (ColumnType.VARCHAR, ColumnType.TEXT):
        length = min(col.max_length or 20, 50)
        return "".join(rng.choice(_ASCII_LOWER_CHARS, size=length))
    return _regen_special_or_fallback(col, rng)


def _regen_special_or_fallback(col: ColumnSchema, rng: Generator) -> Any:
    """Handle ENUM, UUID, and fallback types."""
    if col.data_type == ColumnType.ENUM and col.enum_values:
        return str(rng.choice(col.enum_values))
    if col.data_type == ColumnType.UUID:
        return _regen_uuid(rng)
    # Fallback for DATE, JSON, ARRAY, etc.
    return int(rng.integers(0, 10000))


def _regen_uuid(rng: Generator) -> str:
    """Generate a deterministic UUID from RNG bytes."""
    import uuid  # noqa: PLC0415

    raw = bytes(rng.integers(0, 256, size=16, dtype=np.uint8))
    return str(uuid.UUID(bytes=raw))


def _regen_numeric(col: ColumnSchema, rng: Generator) -> float:
    """Regenerate a FLOAT or DECIMAL value."""
    if col.data_type == ColumnType.DECIMAL:
        precision = col.precision or 10
        scale = col.scale or 2
        max_val = max(10 ** (precision - scale) - 1, 1)
        return round(float(rng.uniform(0, max_val)), scale)
    return round(float(rng.uniform(0, 10000)), 2)


def _enforce_check(
    table: TableSchema,
    rows: list[dict[str, Any]],
    rng: Generator,
) -> None:
    """Enforce CHECK constraints by regenerating out-of-bounds values."""
    from dbsprout.generate.check_parser import parse_check  # noqa: PLC0415

    for col in table.columns:
        if col.check_constraint is None:
            continue

        cc = parse_check(col.name, col.check_constraint)
        if cc is None:
            continue

        for row in rows:
            val = row[col.name]
            if val is None:
                continue

            if cc.allowed_values is not None:
                if str(val) not in cc.allowed_values:
                    row[col.name] = str(rng.choice(cc.allowed_values))
            elif _is_numeric_out_of_bounds(val, cc):
                lo = cc.min_value if cc.min_value is not None else val
                hi = cc.max_value if cc.max_value is not None else val
                row[col.name] = round(float(rng.uniform(lo, hi)), 2)


def _is_numeric_out_of_bounds(val: Any, cc: CheckConstraint) -> bool:
    """Check if a numeric value is outside CHECK constraint bounds."""
    if not isinstance(val, (int, float)):
        return False
    if cc.min_value is None and cc.max_value is None:
        return False
    lo = cc.min_value if cc.min_value is not None else val
    hi = cc.max_value if cc.max_value is not None else val
    return bool(val < lo or val > hi)


def _fk_columns(table: TableSchema) -> set[str]:
    """Get column names that are FK columns."""
    return {col for fk in table.foreign_keys for col in fk.columns}
