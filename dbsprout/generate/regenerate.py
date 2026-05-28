"""Single-table re-roll for the Web-UI Studio (S-128).

``regenerate_table`` re-generates the non-key cells of one table while keeping
its primary-key values byte-identical and re-sampling its foreign-key columns
against the *current* parent rows in ``state``.

Design intent
-------------
The full :func:`dbsprout.generate.orchestrator.orchestrate` pipeline runs all
tables in dependency order; this module is the surgical equivalent for a
single table — used by the Web-UI Studio to iterate on one table without
re-running the whole pipeline.

The function is **pure** at its API boundary: it takes the schema, the
in-memory ``tables_data`` dict (the same shape
:class:`~dbsprout.generate.orchestrator.GenerateResult.tables_data`),
a table name and a seed, then returns a fresh list of row dicts. It deliberately
does NOT import from :mod:`dbsprout.web` and does NOT touch any progress /
cancel / job-manager surface — that wiring belongs to S-131 (HTTP) and lives
behind the API layer.

The implementation reuses the *same* code paths the initial pass uses:

* :func:`dbsprout.plugins.dispatch.resolve_engine` to build the engine.
* :func:`dbsprout.generate.fk_sampling.sample_fk_values` to re-sample FKs.
* :func:`dbsprout.generate.constraints.enforce_constraints` for
  UNIQUE / NOT NULL / CHECK / autoincrement-PK handling.

After constraint enforcement the *original* PK values are written back as the
last step — this is what guarantees PK stability (and dodges the autoincrement
assigner clobbering preserved PKs).

A private :func:`_regen_columns` is reserved for S-129 (single-column re-roll)
so that story can land alongside ``regenerate_table`` without duplicating the
algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dbsprout.generate.constraints import enforce_constraints
from dbsprout.generate.fk_sampling import sample_fk_values
from dbsprout.plugins.dispatch import resolve_engine
from dbsprout.spec.heuristics import map_columns

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema
    from dbsprout.spec.models import DataSpec


__all__ = ["regenerate_table"]


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
        raise ValueError(msg)

    original_rows = state.get(table_name)
    if not original_rows:
        msg = f"regenerate_table: no rows in state for table {table_name!r}"
        raise ValueError(msg)

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
    """Reserved for S-129 — for any non-PK column NOT in ``keep_columns``,
    overwrite the freshly-generated value with the value from ``original_rows``.

    PK columns are handled by :func:`_pin_primary_keys` (which runs after this
    helper). FK columns are *not* spliced — by design, single-column re-rolls
    on non-FK columns still benefit from the orchestrator's FK invariant.
    """
    pk_cols = set(table_schema.primary_key)
    fk_cols = {c for fk in table_schema.foreign_keys for c in fk.columns}
    for orig, new in zip(original_rows, enforced, strict=False):
        for col_name in orig:
            if col_name in pk_cols or col_name in fk_cols:
                continue
            if col_name in keep_columns:
                continue
            new[col_name] = orig[col_name]
    return enforced
