"""Single-table and single-column re-roll for the Web-UI Studio.

Two public entry points live here:

* ``regenerate_table`` (S-128) — re-generates the non-key cells of one table
  while keeping its primary-key values byte-identical and re-sampling its
  foreign-key columns against the *current* parent rows in ``state``.
* ``regenerate_column`` (S-129) — re-rolls a single column on one table.
  Every other column (including FK columns) is byte-identical to the
  originals. PK columns and FK-referenced columns are restricted —
  :class:`RegenerateError` is raised so the route layer (S-131) can map to a
  ``409 Conflict``.

Design intent
-------------
The full :func:`dbsprout.generate.orchestrator.orchestrate` pipeline runs all
tables in dependency order; this module is the surgical equivalent for a
single table or column — used by the Web-UI Studio to iterate without
re-running the whole pipeline.

Both functions are **pure** at their API boundary: schema in, in-memory
``tables_data`` dict in, fresh row dicts out. The module deliberately does
NOT import from :mod:`dbsprout.web` and does NOT touch any progress /
cancel / job-manager surface — that wiring belongs to S-131 (HTTP) and
lives behind the API layer.

The implementation reuses the *same* code paths the initial pass uses:

* :func:`dbsprout.plugins.dispatch.resolve_engine` to build the engine.
* :func:`dbsprout.generate.fk_sampling.sample_fk_values` to re-sample FKs.
* :func:`dbsprout.generate.constraints.enforce_constraints` for
  UNIQUE / NOT NULL / CHECK / autoincrement-PK handling.
* :func:`dbsprout.generate.deterministic.column_seed` to derive a per-column
  RNG seed from ``(global_seed, table, column, nonce)``.

After constraint enforcement the *original* PK values are written back as
the last step — this is what guarantees PK stability (and dodges the
autoincrement assigner clobbering preserved PKs).

The shared :func:`_regen_columns` core is what lets both the whole-table
(S-128) and single-column (S-129) paths run without duplicating algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dbsprout.generate.constraints import enforce_constraints
from dbsprout.generate.deterministic import column_seed
from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.plugins.dispatch import resolve_engine
from dbsprout.spec.heuristics import map_columns

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema
    from dbsprout.spec.models import DataSpec


__all__ = ["RegenerateError", "regenerate_column", "regenerate_table"]


class RegenerateError(ValueError):
    """Domain error for an illegal regenerate call.

    Subclasses :class:`ValueError` so legacy callers using
    ``except ValueError`` continue to work, while the route layer (S-131)
    can catch ``RegenerateError`` directly and translate the structured
    payload into a ``409 Conflict`` response.

    Attributes
    ----------
    table:
        Target table the call referenced.
    column:
        Target column. ``None`` for table-scope errors.
    reason:
        Short machine-readable cause — one of:

        * ``"unknown_table"`` — table is not in the schema.
        * ``"unknown_column"`` — column does not exist on the table.
        * ``"no_rows"`` — ``state`` has no rows for the table.
        * ``"primary_key"`` — column is part of the table's primary key.
        * ``"fk_referenced"`` — column is referenced by another table's
          foreign key (re-rolling would orphan child rows).
    """

    __slots__ = ("column", "reason", "table")

    def __init__(self, message: str, *, table: str, column: str | None, reason: str) -> None:
        super().__init__(message)
        self.table = table
        self.column = column
        self.reason = reason


def regenerate_table(  # noqa: PLR0913 — public surface; flat kwargs mirror orchestrator
    schema: DatabaseSchema,
    state: dict[str, list[dict[str, Any]]],
    table_name: str,
    *,
    seed: int,
    spec: DataSpec | None = None,
    engine: str = "heuristic",
) -> list[dict[str, Any]]:
    """Re-generate one table; preserve PKs; re-sample FKs; enforce constraints.

    Parameters
    ----------
    schema:
        The full :class:`~dbsprout.schema.models.DatabaseSchema`.
    state:
        In-memory generated-rows dict (the same shape orchestrator's
        ``tables_data`` returns). Used to look up the rows being re-rolled and
        as the FK-sampling source for parent PKs. **Not mutated.**
    table_name:
        The table to re-roll.
    seed:
        Deterministic seed passed to the engine, FK sampler, and constraint
        enforcer. Same seed twice → identical output.
    spec:
        Optional :class:`~dbsprout.spec.models.DataSpec`. When supplied
        together with ``engine="spec"`` the spec-driven engine is used; when
        absent or the table has no spec entry, the heuristic engine is used.
    engine:
        ``"heuristic"`` (default) or ``"spec"``. Mirrors the orchestrator's
        engine selection; other values fall through to heuristic.

    Returns
    -------
    list[dict[str, Any]]
        A NEW list of row dicts with the original PK values pinned.

    Raises
    ------
    ValueError
        If ``table_name`` is not in the schema, or ``state`` has no rows for
        it (we need the existing row count + PK values to pin).
    """
    table_schema = schema.get_table(table_name)
    if table_schema is None:
        msg = f"regenerate_table: unknown table {table_name!r}"
        raise RegenerateError(msg, table=table_name, column=None, reason="unknown_table")

    original_rows = state.get(table_name)
    if not original_rows:
        msg = f"regenerate_table: no rows in state for table {table_name!r}"
        raise RegenerateError(msg, table=table_name, column=None, reason="no_rows")

    return _regen_columns(
        schema=schema,
        state=state,
        table_schema=table_schema,
        original_rows=original_rows,
        columns=None,
        seed=seed,
        spec=spec,
        engine=engine,
    )


def regenerate_column(  # noqa: PLR0913 — public surface mirrors regenerate_table
    schema: DatabaseSchema,
    state: dict[str, list[dict[str, Any]]],
    table_name: str,
    column_name: str,
    *,
    seed: int,
    nonce: int = 0,
    spec: DataSpec | None = None,
    engine: str = "heuristic",
) -> list[dict[str, Any]]:
    """Re-roll a single column on one table; preserve PK and every other column.

    Use this when the Studio user wants to fix one specific column without
    disturbing any other data on the row.

    The per-column RNG is derived from
    :func:`dbsprout.generate.deterministic.column_seed` so the same
    ``(seed, table, column, nonce)`` tuple is fully reproducible. Bumping
    ``nonce`` is the "re-roll" knob — same global seed, fresh draw.

    Parameters
    ----------
    schema:
        The full :class:`~dbsprout.schema.models.DatabaseSchema`.
    state:
        In-memory generated-rows dict (same shape as orchestrator
        ``tables_data``). Used to look up rows being re-rolled and as the
        FK-sampling source for parent PKs. **Not mutated.**
    table_name:
        The table containing the column.
    column_name:
        The single column to re-roll.
    seed:
        Global deterministic seed. Combined with ``table_name``,
        ``column_name`` and ``nonce`` to derive the per-column seed.
    nonce:
        Re-roll counter — bump to obtain a different value while keeping
        ``seed`` stable. Default ``0``.
    spec:
        Optional :class:`~dbsprout.spec.models.DataSpec`. When supplied
        with ``engine="spec"`` the spec-driven engine is used.
    engine:
        ``"heuristic"`` (default) or ``"spec"``.

    Returns
    -------
    list[dict[str, Any]]
        A NEW list of row dicts. Only ``column_name`` differs from the
        originals; every other column (including FKs) is byte-identical.

    Raises
    ------
    RegenerateError
        With ``reason="unknown_table"`` if the table is missing,
        ``reason="unknown_column"`` if the column is missing,
        ``reason="primary_key"`` if the column is part of the PK,
        ``reason="fk_referenced"`` if another table's FK references this
        column (re-rolling would orphan child rows), or
        ``reason="no_rows"`` if ``state`` has no rows for the table.
    """
    table_schema = schema.get_table(table_name)
    if table_schema is None:
        msg = f"regenerate_column: unknown table {table_name!r}"
        raise RegenerateError(msg, table=table_name, column=column_name, reason="unknown_table")

    if table_schema.get_column(column_name) is None:
        msg = f"regenerate_column: unknown column {table_name}.{column_name!r}"
        raise RegenerateError(msg, table=table_name, column=column_name, reason="unknown_column")

    if column_name in table_schema.primary_key:
        msg = (
            f"regenerate_column: {table_name}.{column_name!r} is part of the "
            "primary key and cannot be re-rolled"
        )
        raise RegenerateError(msg, table=table_name, column=column_name, reason="primary_key")

    if _is_referenced_by_child_fk(schema, table_name, column_name):
        msg = (
            f"regenerate_column: {table_name}.{column_name!r} is referenced "
            "by another table's foreign key and cannot be re-rolled"
        )
        raise RegenerateError(msg, table=table_name, column=column_name, reason="fk_referenced")

    original_rows = state.get(table_name)
    if not original_rows:
        msg = f"regenerate_column: no rows in state for table {table_name!r}"
        raise RegenerateError(msg, table=table_name, column=column_name, reason="no_rows")

    per_column_seed = column_seed(seed, table_name, column_name, nonce=nonce)
    return _regen_columns(
        schema=schema,
        state=state,
        table_schema=table_schema,
        original_rows=original_rows,
        columns={column_name},
        seed=per_column_seed,
        spec=spec,
        engine=engine,
    )


def _is_referenced_by_child_fk(schema: DatabaseSchema, table_name: str, column_name: str) -> bool:
    """Return True if any other table has a FK whose ``ref_columns`` include
    ``column_name`` on ``table_name``.

    Self-references are excluded — the column is referenced by *its own*
    foreign key, which is fine: re-rolling self-ref data is allowed and
    handled by the constraint pass downstream.
    """
    for child in schema.tables:
        if child.name == table_name:
            continue
        for fk in child.foreign_keys:
            if fk.ref_table == table_name and column_name in fk.ref_columns:
                return True
    return False


def _regen_columns(  # noqa: PLR0913 — keyword-only fan-out mirrors the public API
    *,
    schema: DatabaseSchema,
    state: dict[str, list[dict[str, Any]]],
    table_schema: TableSchema,
    original_rows: list[dict[str, Any]],
    columns: set[str] | None,
    seed: int,
    spec: DataSpec | None,
    engine: str,
) -> list[dict[str, Any]]:
    """Shared core for whole-table and (future, S-129) single-column re-rolls.

    ``columns`` semantics:
        * ``None``  → re-roll every non-PK column (the S-128 whole-table case).
        * ``set``   → re-roll only those columns; values for any other non-PK
          column are copied from ``original_rows`` (reserved for S-129).

    The PK pin always runs last so any constraint-pass autoincrement assignment
    is overwritten by the original PK values.
    """
    num_rows = len(original_rows)
    fresh_rows = _generate_fresh_rows(
        schema=schema,
        table_schema=table_schema,
        num_rows=num_rows,
        seed=seed,
        spec=spec,
        engine=engine,
    )

    # Re-sample FKs from current parent PKs (same path as the initial pass).
    sample_fk_values(table_schema, state, fresh_rows, seed)

    # Enforce UNIQUE / NOT NULL / CHECK / autoincrement-PK (returns a new list).
    enforced = enforce_constraints(table_schema, fresh_rows, seed)

    # If a partial-column re-roll was requested (S-129), splice the untouched
    # columns back from the originals. For S-128 (columns is None) this is a
    # no-op.
    if columns is not None:
        enforced = _splice_unchanged_columns(
            original_rows=original_rows,
            enforced=enforced,
            keep_columns=columns,
            table_schema=table_schema,
        )

    # PK pin — must be the very last step (clobbers any autoincrement assignment).
    _pin_primary_keys(table_schema, original_rows, enforced)

    return enforced


def _generate_fresh_rows(  # noqa: PLR0913 — local helper; kwargs mirror _regen_columns
    *,
    schema: DatabaseSchema,
    table_schema: TableSchema,
    num_rows: int,
    seed: int,
    spec: DataSpec | None,
    engine: str,
) -> list[dict[str, Any]]:
    """Build a fresh batch of rows using the same engine surface the
    orchestrator uses.

    Engine selection mirrors
    :func:`dbsprout.generate.orchestrator._select_engines` but is local: we
    instantiate at most one engine per call and do not memoise across calls
    (the Web UI typically re-rolls one table at a time).
    """
    if engine == "spec" and spec is not None:
        table_spec = spec.get_table_spec(table_schema.name)
        if table_spec is not None:
            spec_engine = resolve_engine("spec_driven", seed=seed)
            return list(spec_engine.generate_table(table_schema, table_spec, num_rows))

    heuristic = resolve_engine("heuristic", seed=seed)
    mappings = map_columns(schema).get(table_schema.name, {})
    return list(heuristic.generate_table(table_schema, mappings, num_rows))


def _pin_primary_keys(
    table_schema: TableSchema,
    original_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> None:
    """Overwrite PK cells in ``new_rows`` with the values from ``original_rows``.

    Handles both single-column and composite PKs. Mutates ``new_rows`` in
    place (it is already a local fresh list inside this module).
    """
    pk_cols = table_schema.primary_key
    if not pk_cols:
        return
    for orig, new in zip(original_rows, new_rows, strict=False):
        for col in pk_cols:
            new[col] = orig[col]


def _splice_unchanged_columns(
    *,
    original_rows: list[dict[str, Any]],
    enforced: list[dict[str, Any]],
    keep_columns: set[str],
    table_schema: TableSchema,
) -> list[dict[str, Any]]:
    """Used by S-129 — for any non-PK column NOT in ``keep_columns``,
    overwrite the freshly-generated value with the value from
    ``original_rows``.

    PK columns are handled by :func:`_pin_primary_keys` (runs after this
    helper). FK columns are spliced too: a single-column re-roll on a
    *non-FK* column must leave the FK byte-identical, otherwise the
    "other columns byte-identical" invariant breaks. When the caller
    explicitly targets a FK column they put it in ``keep_columns`` and the
    freshly-sampled value survives.
    """
    pk_cols = set(table_schema.primary_key)
    for orig, new in zip(original_rows, enforced, strict=False):
        for col_name in orig:
            if col_name in pk_cols:
                continue
            if col_name in keep_columns:
                continue
            new[col_name] = orig[col_name]
    return enforced
