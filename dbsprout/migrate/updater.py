"""Incremental seed updater — per-change-type update rules.

Consumes ``list[SchemaChange]`` from ``SchemaDiffer`` and applies
deterministic update rules to transform existing seed data, touching
only the affected columns, tables, and constraints.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.update_models import (
    PlannedAction,
    UpdateAction,
    UpdatePlan,
    UpdateResult,
)
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.config.models import DBSproutConfig
    from dbsprout.migrate.models import SchemaChange

logger = logging.getLogger(__name__)

_DEDUP_MAX_CANDIDATES = 10


class _ApplyContext:
    """Mutable tracking state for an ``apply()`` execution."""

    __slots__ = ("modified", "rows_added", "tables_added", "tables_removed")

    def __init__(self) -> None:
        self.modified: set[tuple[str, int]] = set()
        self.rows_added: int = 0
        self.tables_added: list[str] = []
        self.tables_removed: list[str] = []


# ── Classifier functions ────────────────────────────────────────────
#
# Each returns ``(UpdateAction, str)`` — the action enum plus a
# human-readable description for the update plan.


def _classify_table_added(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.GENERATE_TABLE,
        f"Generate seed data for new table '{change.table_name}'",
    )


def _classify_table_removed(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.DROP_TABLE,
        f"Drop seed data for removed table '{change.table_name}'",
    )


def _classify_column_added(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.GENERATE_COLUMN,
        f"Generate values for new column '{change.column_name}' in '{change.table_name}'",
    )


def _classify_column_removed(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.DROP_COLUMN,
        f"Drop column '{change.column_name}' from '{change.table_name}' seed data",
    )


def _classify_column_type_changed(change: SchemaChange) -> tuple[UpdateAction, str]:
    detail = change.detail or {}
    old_t = detail.get("old_type", "?")
    new_t = detail.get("new_type", "?")
    return (
        UpdateAction.REGENERATE_COLUMN,
        (
            f"Regenerate column '{change.column_name}' in '{change.table_name}' "
            f"(type changed from {old_t} to {new_t})"
        ),
    )


def _classify_nullability_changed(change: SchemaChange) -> tuple[UpdateAction, str]:
    detail = change.detail or {}
    new_nullable: bool = detail.get("new_nullable", True)
    if not new_nullable:
        return (
            UpdateAction.REGENERATE_COLUMN,
            (
                f"Regenerate column '{change.column_name}' in '{change.table_name}' "
                f"(now NOT NULL — fill existing NULLs)"
            ),
        )
    return (
        UpdateAction.NO_ACTION,
        (
            f"Column '{change.column_name}' in '{change.table_name}' "
            f"became nullable — no seed data update needed"
        ),
    )


def _classify_default_changed(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.NO_ACTION,
        (
            f"Default value changed for '{change.column_name}' in "
            f"'{change.table_name}' — no seed data update needed"
        ),
    )


def _classify_fk_added(change: SchemaChange) -> tuple[UpdateAction, str]:
    detail = change.detail or {}
    fk = detail.get("fk", {})
    ref_table = fk.get("ref_table", "?")
    return (
        UpdateAction.VALIDATE_FK,
        (
            f"Validate FK on '{change.table_name}' referencing "
            f"'{ref_table}' — ensure referential integrity"
        ),
    )


def _classify_fk_removed(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.NO_ACTION,
        f"FK removed from '{change.table_name}' — no seed data update needed",
    )


def _classify_index_added(change: SchemaChange) -> tuple[UpdateAction, str]:
    detail = change.detail or {}
    index = detail.get("index", {})
    is_unique: bool = index.get("unique", False)
    if is_unique:
        cols = index.get("columns", [])
        return (
            UpdateAction.DEDUPLICATE,
            (f"Deduplicate '{change.table_name}' seed data for new unique index on {cols}"),
        )
    return (
        UpdateAction.NO_ACTION,
        f"Non-unique index added on '{change.table_name}' — no seed data update needed",
    )


def _classify_index_removed(change: SchemaChange) -> tuple[UpdateAction, str]:
    return (
        UpdateAction.NO_ACTION,
        f"Index removed from '{change.table_name}' — no seed data update needed",
    )


def _classify_enum_changed(change: SchemaChange) -> tuple[UpdateAction, str]:
    desc = (
        f"Replace enum values for '{change.column_name}' "  # noqa: S608 -- not SQL
        f"in '{change.table_name}' with updated set"
    )
    return (UpdateAction.REPLACE_ENUM, desc)


_CLASSIFY_DISPATCH: dict[
    SchemaChangeType,
    Callable[[SchemaChange], tuple[UpdateAction, str]],
] = {
    SchemaChangeType.TABLE_ADDED: _classify_table_added,
    SchemaChangeType.TABLE_REMOVED: _classify_table_removed,
    SchemaChangeType.COLUMN_ADDED: _classify_column_added,
    SchemaChangeType.COLUMN_REMOVED: _classify_column_removed,
    SchemaChangeType.COLUMN_TYPE_CHANGED: _classify_column_type_changed,
    SchemaChangeType.COLUMN_NULLABILITY_CHANGED: _classify_nullability_changed,
    SchemaChangeType.COLUMN_DEFAULT_CHANGED: _classify_default_changed,
    SchemaChangeType.FOREIGN_KEY_ADDED: _classify_fk_added,
    SchemaChangeType.FOREIGN_KEY_REMOVED: _classify_fk_removed,
    SchemaChangeType.INDEX_ADDED: _classify_index_added,
    SchemaChangeType.INDEX_REMOVED: _classify_index_removed,
    SchemaChangeType.ENUM_CHANGED: _classify_enum_changed,
}


# ── Action ordering ────────────────────────────────────────────────

_ACTION_ORDER: dict[UpdateAction, int] = {
    UpdateAction.GENERATE_TABLE: 0,
    UpdateAction.GENERATE_COLUMN: 1,
    UpdateAction.REGENERATE_COLUMN: 1,
    UpdateAction.DROP_COLUMN: 1,
    UpdateAction.VALIDATE_FK: 2,
    UpdateAction.DEDUPLICATE: 2,
    UpdateAction.REPLACE_ENUM: 2,
    UpdateAction.DROP_TABLE: 3,
    UpdateAction.NO_ACTION: 4,
}


def _order_actions(actions: list[PlannedAction]) -> list[PlannedAction]:
    """Sort actions for safe execution order.

    0. DROP_TABLE for tables that are also being re-added (same name)
    1. GENERATE_TABLE (topologically ordered by FK deps)
    2. Column-level: GENERATE_COLUMN, REGENERATE_COLUMN, DROP_COLUMN
    3. Constraint: VALIDATE_FK, DEDUPLICATE, REPLACE_ENUM
    4. DROP_TABLE for tables that are only removed
    5. NO_ACTION (skip)
    """
    # Identify tables being both removed and re-added (AC-29)
    added_tables = {a.change.table_name for a in actions if a.action == UpdateAction.GENERATE_TABLE}
    removed_tables = {a.change.table_name for a in actions if a.action == UpdateAction.DROP_TABLE}
    recreated = added_tables & removed_tables

    # Build topological order for GENERATE_TABLE actions (AC-28)
    gen_table_order = _topological_order_for_new_tables(actions)

    def sort_key(a: PlannedAction) -> tuple[int, int]:
        if a.action == UpdateAction.DROP_TABLE and a.change.table_name in recreated:
            return (-1, 0)  # Drop-before-recreate runs first
        if a.action == UpdateAction.GENERATE_TABLE:
            return (0, gen_table_order.get(a.change.table_name, 0))
        return (_ACTION_ORDER[a.action], 0)

    return sorted(actions, key=sort_key)


def _topological_order_for_new_tables(actions: list[PlannedAction]) -> dict[str, int]:
    """Build topological ordering for GENERATE_TABLE actions by FK deps."""
    from graphlib import TopologicalSorter  # noqa: PLC0415

    gen_actions = [a for a in actions if a.action == UpdateAction.GENERATE_TABLE]
    if len(gen_actions) <= 1:
        return {a.change.table_name: 0 for a in gen_actions}

    new_table_names = {a.change.table_name for a in gen_actions}
    deps: dict[str, set[str]] = {name: set() for name in new_table_names}

    for a in gen_actions:
        detail = a.change.detail or {}
        table_dict = detail.get("table", {})
        fks = table_dict.get("foreign_keys", [])
        for fk in fks:
            ref_table = fk.get("ref_table", "") if isinstance(fk, dict) else fk.ref_table
            if ref_table in new_table_names:
                deps[a.change.table_name].add(ref_table)

    sorter = TopologicalSorter(deps)
    order: dict[str, int] = {}
    for idx, table_name in enumerate(sorter.static_order()):
        order[table_name] = idx

    return order


class IncrementalUpdater:
    """Applies per-change-type update rules to existing seed data."""

    def __init__(
        self,
        schema: DatabaseSchema,
        config: DBSproutConfig,
        seed: int,
    ) -> None:
        self._schema = schema
        self._config = config
        self._seed = seed

    # ── Planning ───────────────────────────────────────────────────

    def plan(
        self,
        changes: list[SchemaChange],
        existing_data: dict[str, list[dict[str, Any]]],  # noqa: ARG002
    ) -> UpdatePlan:
        """Classify each change into an action without modifying data."""
        actions = [self._classify_change(c) for c in changes]
        return UpdatePlan(actions=actions)

    def _classify_change(self, change: SchemaChange) -> PlannedAction:
        """Map a SchemaChange to a PlannedAction."""
        classifier = _CLASSIFY_DISPATCH[change.change_type]
        action, description = classifier(change)
        return PlannedAction(change=change, action=action, description=description)

    # ── Execution ──────────────────────────────────────────────────

    def apply(
        self,
        plan: UpdatePlan,
        existing_data: dict[str, list[dict[str, Any]]],
    ) -> UpdateResult:
        """Execute the plan and return updated data."""
        data = copy.deepcopy(existing_data)
        ctx = _ApplyContext()
        applied: list[PlannedAction] = []

        ordered = _order_actions(plan.actions)
        for action in ordered:
            if action.action == UpdateAction.NO_ACTION:
                applied.append(action)
                continue

            self._dispatch_action(action, data, ctx)
            applied.append(action)

        rows_removed = sum(len(existing_data.get(t, [])) for t in ctx.tables_removed)

        return UpdateResult(
            tables_data=data,
            actions_applied=applied,
            rows_modified=len(ctx.modified),
            rows_added=ctx.rows_added,
            rows_removed=rows_removed,
            tables_added=ctx.tables_added,
            tables_removed=ctx.tables_removed,
        )

    def update(
        self,
        changes: list[SchemaChange],
        existing_data: dict[str, list[dict[str, Any]]],
    ) -> UpdateResult:
        """Convenience: plan + apply in one call."""
        return self.apply(self.plan(changes, existing_data), existing_data)

    # ── Dispatch ───────────────────────────────────────────────────

    def _dispatch_action(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        ctx: _ApplyContext,
    ) -> None:
        """Route an action to its handler."""
        act = action.action
        if act == UpdateAction.DROP_COLUMN:
            self._apply_drop_column(action, data, ctx.modified)
        elif act == UpdateAction.DROP_TABLE:
            self._apply_drop_table(action, data, ctx.tables_removed)
        elif act == UpdateAction.GENERATE_COLUMN:
            self._apply_generate_column(action, data, ctx.modified)
        elif act == UpdateAction.REGENERATE_COLUMN:
            self._apply_regenerate_column(action, data, ctx.modified)
        elif act == UpdateAction.GENERATE_TABLE:
            self._apply_generate_table(action, data, ctx)
        elif act == UpdateAction.VALIDATE_FK:
            self._apply_validate_fk(action, data, ctx.modified)
        elif act == UpdateAction.DEDUPLICATE:
            self._apply_deduplicate(action, data, ctx.modified)
        elif act == UpdateAction.REPLACE_ENUM:
            self._apply_replace_enum(action, data, ctx.modified)

    # ── Handlers ───────────────────────────────────────────────────

    def _apply_drop_column(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Remove a column from every row in a table."""
        table_name = action.change.table_name
        col_name = action.change.column_name
        if table_name not in data or col_name is None:
            return
        for i, row in enumerate(data[table_name]):
            if col_name in row:
                del row[col_name]
                modified.add((table_name, i))

    def _apply_drop_table(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        tables_removed: list[str],
    ) -> None:
        """Remove an entire table from the data."""
        table_name = action.change.table_name
        if table_name in data:
            del data[table_name]
            tables_removed.append(table_name)

    def _apply_generate_column(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Add a new column to existing rows."""
        table_name = action.change.table_name
        col_name = action.change.column_name
        if table_name not in data or col_name is None:
            return

        rows = data[table_name]
        if not rows:
            return

        detail = action.change.detail or {}

        # COLUMN_NULLABILITY_CHANGED → NOT NULL: fill only None values
        if action.change.change_type == SchemaChangeType.COLUMN_NULLABILITY_CHANGED:
            self._fill_null_values(table_name, col_name, rows, modified)
            return

        col_dict = detail.get("column", {})
        is_nullable = col_dict.get("nullable", True)

        if is_nullable:
            for i, row in enumerate(rows):
                row[col_name] = None
                modified.add((table_name, i))
        else:
            values = self._generate_column_values(
                table_name,
                col_dict,
                len(rows),
            )
            for i, row in enumerate(rows):
                row[col_name] = values[i]
                modified.add((table_name, i))

    def _apply_regenerate_column(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Regenerate column values after a type or nullability change."""
        table_name = action.change.table_name
        col_name = action.change.column_name
        if table_name not in data or col_name is None:
            return

        rows = data[table_name]
        if not rows:
            return

        # COLUMN_NULLABILITY_CHANGED → NOT NULL: delegate to generate_column
        if action.change.change_type == SchemaChangeType.COLUMN_NULLABILITY_CHANGED:
            self._apply_generate_column(action, data, modified)
            return

        # COLUMN_TYPE_CHANGED: regenerate all values with new type
        detail = action.change.detail or {}
        new_type = detail.get("new_type", "varchar")

        # Preserve nullability from the schema (don't hardcode)
        table = self._schema.get_table(table_name)
        col = table.get_column(col_name) if table else None
        is_nullable = col.nullable if col else True

        col_dict = {"name": col_name, "data_type": new_type, "nullable": is_nullable}
        values = self._generate_column_values(table_name, col_dict, len(rows))
        for i, row in enumerate(rows):
            row[col_name] = values[i]
            modified.add((table_name, i))

    def _apply_generate_table(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        ctx: _ApplyContext,
    ) -> None:
        """Generate seed data for a brand-new table."""
        from dbsprout.generate.constraints import enforce_constraints  # noqa: PLC0415
        from dbsprout.generate.engines.heuristic import HeuristicEngine  # noqa: PLC0415
        from dbsprout.generate.fk_sampling import sample_fk_values  # noqa: PLC0415
        from dbsprout.spec.heuristics import map_columns as map_all_columns  # noqa: PLC0415

        table_name = action.change.table_name
        detail = action.change.detail or {}
        table_dict = detail.get("table", {})

        table_schema = TableSchema.model_validate(table_dict)

        num_rows = self._get_row_count(table_name)

        mini_schema = DatabaseSchema(tables=[table_schema])
        mappings = map_all_columns(mini_schema)
        table_mappings = mappings.get(table_name, {})

        engine = HeuristicEngine(seed=self._seed)
        rows = engine.generate_table(table_schema, table_mappings, num_rows)

        rows = sample_fk_values(table_schema, data, rows, self._seed)
        rows = enforce_constraints(table_schema, rows, self._seed)

        data[table_name] = rows
        ctx.tables_added.append(table_name)
        ctx.rows_added += len(rows)

    def _apply_validate_fk(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Validate FK references and replace invalid ones."""
        table_name = action.change.table_name
        if table_name not in data:
            return

        detail = action.change.detail or {}
        fk_dict = detail.get("fk", {})
        ref_table = fk_dict.get("ref_table", "")
        fk_columns: list[str] = fk_dict.get("columns", [])
        ref_columns: list[str] = fk_dict.get("ref_columns", [])

        if ref_table not in data or not fk_columns or not ref_columns:
            return

        parent_rows = data[ref_table]
        if not parent_rows:
            return

        rows = data[table_name]
        rng = np.random.default_rng(self._seed)

        if len(fk_columns) == 1:
            fk_col = fk_columns[0]
            ref_col = ref_columns[0]
            pk_list = [row[ref_col] for row in parent_rows]
            valid_pks = set(pk_list)

            for i, row in enumerate(rows):
                val = row.get(fk_col)
                if val is None:
                    continue  # preserve None for nullable FK columns
                if val not in valid_pks:
                    row[fk_col] = pk_list[int(rng.integers(0, len(pk_list)))]
                    modified.add((table_name, i))
        else:
            valid_tuples = {tuple(row[rc] for rc in ref_columns) for row in parent_rows}
            for i, row in enumerate(rows):
                current = tuple(row.get(fc) for fc in fk_columns)
                if any(v is None for v in current):
                    continue  # preserve None for nullable FK columns
                if current not in valid_tuples:
                    parent_idx = int(rng.integers(0, len(parent_rows)))
                    parent_row = parent_rows[parent_idx]
                    for fc, rc in zip(fk_columns, ref_columns, strict=True):
                        row[fc] = parent_row[rc]
                    modified.add((table_name, i))

    def _apply_deduplicate(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Remove duplicate values for a new unique index."""
        table_name = action.change.table_name
        if table_name not in data:
            return

        detail = action.change.detail or {}
        index_dict = detail.get("index", {})
        columns: list[str] = index_dict.get("columns", [])

        if not columns:
            return

        rows = data[table_name]
        if not rows:
            return

        if len(columns) == 1:
            col_name = columns[0]
            table = self._schema.get_table(table_name)
            col = table.get_column(col_name) if table else None

            seen: set[Any] = set()
            for i, row in enumerate(rows):
                val = row.get(col_name)
                if val in seen and col is not None:
                    new_values = self._generate_column_values(
                        table_name,
                        col.model_dump(),
                        _DEDUP_MAX_CANDIDATES,
                    )
                    for nv in new_values:
                        if nv not in seen:
                            row[col_name] = nv
                            val = nv
                            modified.add((table_name, i))
                            break
                seen.add(val)
        else:
            logger.warning(
                "Composite unique index deduplication not yet supported for table '%s'",
                table_name,
            )

    def _apply_replace_enum(
        self,
        action: PlannedAction,
        data: dict[str, list[dict[str, Any]]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Replace invalid enum values with values from the new enum set."""
        table_name = action.change.table_name
        col_name = action.change.column_name
        if table_name not in data or col_name is None:
            return

        detail = action.change.detail or {}
        new_values: list[str] = detail.get("new_values", [])
        if not new_values:
            return

        rows = data[table_name]
        new_set = set(new_values)
        rng = np.random.default_rng(self._seed)

        for i, row in enumerate(rows):
            if row.get(col_name) not in new_set:
                row[col_name] = new_values[int(rng.integers(0, len(new_values)))]
                modified.add((table_name, i))

    # ── Helpers ────────────────────────────────────────────────────

    def _fill_null_values(
        self,
        table_name: str,
        col_name: str,
        rows: list[dict[str, Any]],
        modified: set[tuple[str, int]],
    ) -> None:
        """Replace None values in an existing column with generated values."""
        null_indices = [i for i, row in enumerate(rows) if row.get(col_name) is None]
        if not null_indices:
            return

        table = self._schema.get_table(table_name)
        if table is None:
            return
        col = table.get_column(col_name)
        if col is None:
            return

        col_dict = col.model_dump()
        values = self._generate_column_values(
            table_name,
            col_dict,
            len(null_indices),
        )
        for j, i in enumerate(null_indices):
            rows[i][col_name] = values[j]
            modified.add((table_name, i))

    def _generate_column_values(
        self,
        table_name: str,
        col_dict: dict[str, Any],
        num_rows: int,
    ) -> list[Any]:
        """Generate values for a single column using HeuristicEngine."""
        from dbsprout.generate.engines.heuristic import HeuristicEngine  # noqa: PLC0415
        from dbsprout.spec.heuristics import map_columns as map_all_columns  # noqa: PLC0415

        data_type_str = col_dict.get("data_type", "varchar")
        try:
            data_type = ColumnType(data_type_str)
        except ValueError:
            data_type = ColumnType.VARCHAR

        col_schema = ColumnSchema(
            name=col_dict.get("name", "col"),
            data_type=data_type,
            nullable=col_dict.get("nullable", True),
            max_length=col_dict.get("max_length"),
            precision=col_dict.get("precision"),
            scale=col_dict.get("scale"),
            enum_values=col_dict.get("enum_values"),
        )

        mini_table = TableSchema(
            name=table_name,
            columns=[col_schema],
            primary_key=[],
        )
        mini_schema = DatabaseSchema(tables=[mini_table])

        mappings = map_all_columns(mini_schema)
        table_mappings = mappings.get(table_name, {})

        engine = HeuristicEngine(seed=self._seed)
        generated_rows = engine.generate_table(
            mini_table,
            table_mappings,
            num_rows,
        )

        return [row[col_schema.name] for row in generated_rows]

    def _get_row_count(self, table_name: str) -> int:
        """Get the row count for a table from config or default."""
        override = self._config.tables.get(table_name)
        if override and override.rows is not None:
            return override.rows
        return self._config.generation.default_rows
