"""Schema browser widget for the TUI Schema tab (S-088).

Renders a loaded :class:`~dbsprout.schema.models.DatabaseSchema` as a Textual
:class:`~textual.widgets.Tree` (database -> tables -> columns -> constraints)
beside a detail panel, with a real-time name filter.

Textual is the heavy optional ``[tui]`` extra; it is imported at module top
level here because this module is only imported lazily (by the running app and
by the ``dbsprout tui`` command), never on the hot CLI startup path.

The structural and text-rendering logic lives in *pure* module-level helpers
(``_filter_schema``, ``_detail_text``, label/summary builders) so it can be unit
tested without a running app and reused by the widget's event handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static, Tree

from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.widgets.tree import TreeNode


NodeKind = Literal["database", "table", "column", "constraint"]

# Glyphs for node types (plain ASCII-safe markers; terminals render reliably).
_DB_ICON = "▣"
_TABLE_ICON = "▤"
_COLUMN_ICON = "·"
_PK_ICON = "🔑"
_FK_ICON = "🔗"

_EMPTY_MESSAGE = (
    "No snapshot found.\n\n"
    "Run [b]dbsprout init --db <url>[/b] (or generate from a schema file) to "
    "capture a schema snapshot, then reopen the TUI."
)


@dataclass(frozen=True)
class SchemaNodeData:
    """Immutable payload attached to each tree node.

    Identifies what a node represents so selection handling and detail rendering
    are pure functions of the payload rather than of the rendered label text.
    """

    kind: NodeKind
    table: TableSchema | None = None
    column: ColumnSchema | None = None
    constraint_label: str | None = None


# ── Pure helpers: type / constraint / summary rendering ─────────────────────


def _column_type_str(column: ColumnSchema) -> str:
    """Human-readable column type, including length / precision when present."""
    base = column.data_type.value.upper()
    if column.data_type == ColumnType.VARCHAR and column.max_length is not None:
        return f"VARCHAR({column.max_length})"
    if column.data_type == ColumnType.DECIMAL and column.precision is not None:
        if column.scale is not None:
            return f"DECIMAL({column.precision},{column.scale})"
        return f"DECIMAL({column.precision})"
    return base


def _fk_reference(table: TableSchema, column: ColumnSchema) -> str | None:
    """Return ``ref_table.ref_column`` if *column* participates in an FK."""
    for fk in table.foreign_keys:
        if column.name in fk.columns:
            idx = fk.columns.index(column.name)
            ref_col = fk.ref_columns[idx] if idx < len(fk.ref_columns) else fk.ref_columns[0]
            return f"{fk.ref_table}.{ref_col}"
    return None


def _column_constraint_labels(table: TableSchema, column: ColumnSchema) -> list[str]:
    """Constraint labels for a column, in display order."""
    labels: list[str] = []
    if column.primary_key:
        labels.append("PK")
    fk_ref = _fk_reference(table, column)
    if fk_ref is not None:
        labels.append(f"FK -> {fk_ref}")
    if column.unique:
        labels.append("UNIQUE")
    if column.autoincrement:
        labels.append("AUTO_INCREMENT")
    labels.append("NOT NULL" if not column.nullable else "NULLABLE")
    if column.check_constraint is not None:
        labels.append(f"CHECK ({column.check_constraint})")
    if column.default is not None:
        labels.append(f"DEFAULT {column.default}")
    return labels


def _column_summary(table: TableSchema, column: ColumnSchema) -> str:
    """One-line column summary, e.g. ``email: VARCHAR(255) [UNIQUE, NOT NULL]``."""
    labels = _column_constraint_labels(table, column)
    suffix = f" [{', '.join(labels)}]" if labels else ""
    return f"{column.name}: {_column_type_str(column)}{suffix}"


def _table_constraint_count(table: TableSchema) -> int:
    """Count of table-level constraints (PK presence + each FK + each index)."""
    count = len(table.foreign_keys) + len(table.indexes)
    if table.primary_key:
        count += 1
    return count


def _table_summary(table: TableSchema) -> str:
    """One-line table summary, e.g. ``users (4 columns, 2 constraints)``."""
    n_cols = len(table.columns)
    n_constraints = _table_constraint_count(table)
    col_word = "column" if n_cols == 1 else "columns"
    con_word = "constraint" if n_constraints == 1 else "constraints"
    return f"{table.name} ({n_cols} {col_word}, {n_constraints} {con_word})"


# ── Pure helper: filtering ──────────────────────────────────────────────────


def _filter_schema(schema: DatabaseSchema, query: str) -> DatabaseSchema:
    """Return a new schema containing only tables/columns matching *query*.

    Empty/blank query returns the schema unchanged. Matching is a
    case-insensitive substring test: a table-name match keeps the whole table;
    otherwise only the columns whose names match are kept (and the table is
    surfaced). Non-matching tables are dropped entirely.
    """
    needle = query.strip().lower()
    if not needle:
        return schema

    kept: list[TableSchema] = []
    for table in schema.tables:
        if needle in table.name.lower():
            kept.append(table)
            continue
        matching_cols = [c for c in table.columns if needle in c.name.lower()]
        if matching_cols:
            kept.append(table.model_copy(update={"columns": matching_cols}))
    return schema.model_copy(update={"tables": kept})


# ── Pure helper: detail-panel text ──────────────────────────────────────────


def _detail_text(data: SchemaNodeData, schema: DatabaseSchema | None) -> str:
    """Build rich-markup detail text for the highlighted node."""
    if schema is None or not schema.tables:
        return _EMPTY_MESSAGE
    if data.kind == "database":
        return _database_detail(schema)
    if data.kind == "table" and data.table is not None:
        return _table_detail(data.table)
    if data.kind == "column" and data.table is not None and data.column is not None:
        return _column_detail(data.table, data.column)
    if data.kind == "constraint" and data.constraint_label is not None:
        col = data.column.name if data.column is not None else ""
        return f"[b]Constraint[/b]\n\n[cyan]{data.constraint_label}[/cyan]\n\nColumn: {col}"
    return ""


def _database_detail(schema: DatabaseSchema) -> str:
    name = schema.source or schema.dialect or "database"
    lines = [
        f"[b]{_DB_ICON} Database: {name}[/b]",
        "",
        f"Dialect: {schema.dialect or 'unknown'}",
        f"Tables: {len(schema.tables)}",
        "",
        "[dim]" + ", ".join(schema.table_names()) + "[/dim]",
    ]
    return "\n".join(lines)


def _table_detail(table: TableSchema) -> str:
    pk = ", ".join(table.primary_key) if table.primary_key else "(none)"
    lines = [
        f"[b]{_TABLE_ICON} Table: {table.name}[/b]",
        "",
        f"Columns: {len(table.columns)}",
        f"Primary key: {pk}",
        f"Constraints: {_table_constraint_count(table)}",
    ]
    if table.foreign_keys:
        lines.append("")
        lines.append("[b]Foreign keys[/b]")
        for fk in table.foreign_keys:
            cols = ", ".join(fk.columns)
            refs = ", ".join(fk.ref_columns)
            lines.append(f"  {_FK_ICON} {cols} -> {fk.ref_table}.{refs}")
    if table.comment:
        lines.extend(("", f"[dim]{table.comment}[/dim]"))
    return "\n".join(lines)


def _column_detail(table: TableSchema, column: ColumnSchema) -> str:
    labels = _column_constraint_labels(table, column)
    lines = [
        f"[b]{_COLUMN_ICON} Column: {table.name}.{column.name}[/b]",
        "",
        f"Type: {_column_type_str(column)}",
        f"Nullable: {'yes' if column.nullable else 'no'}",
        f"Default: {column.default if column.default is not None else '(none)'}",
        "",
        "[b]Constraints[/b]",
    ]
    lines.extend(f"  [cyan]{label}[/cyan]" for label in labels)
    if column.comment:
        lines.extend(("", f"[dim]{column.comment}[/dim]"))
    return "\n".join(lines)


# ── Tree population (pure structural builder, app-independent) ───────────────


def _populate_tree(tree: Tree[SchemaNodeData], schema: DatabaseSchema | None) -> None:
    """Clear and rebuild *tree* from *schema* (no-op safe when ``None``/empty)."""
    tree.clear()
    if schema is None or not schema.tables:
        tree.root.label = f"{_DB_ICON} (no schema)"
        tree.root.data = SchemaNodeData(kind="database")
        return

    db_name = schema.source or schema.dialect or "database"
    tree.root.label = f"{_DB_ICON} Database: {db_name}"
    tree.root.data = SchemaNodeData(kind="database")
    tree.root.expand()

    for table in schema.tables:
        table_node = tree.root.add(
            f"{_TABLE_ICON} {_table_summary(table)}",
            data=SchemaNodeData(kind="table", table=table),
        )
        for column in table.columns:
            icon = _PK_ICON if column.primary_key else _COLUMN_ICON
            column_node = table_node.add(
                f"{icon} {_column_summary(table, column)}",
                data=SchemaNodeData(kind="column", table=table, column=column),
            )
            for label in _column_constraint_labels(table, column):
                column_node.add_leaf(
                    label,
                    data=SchemaNodeData(
                        kind="constraint",
                        table=table,
                        column=column,
                        constraint_label=label,
                    ),
                )


# ── Widget ──────────────────────────────────────────────────────────────────


class SchemaBrowser(Vertical):
    """Interactive schema tree + filter + detail panel for the Schema tab."""

    DEFAULT_CSS: ClassVar[str] = ""

    def __init__(self, schema: DatabaseSchema | None = None) -> None:
        super().__init__()
        self._schema = schema

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Filter tables / columns…", id="schema-filter")
        with Horizontal(id="schema-body"):
            yield Tree("Database", id="schema-tree")
            yield Static(id="schema-detail")

    def on_mount(self) -> None:
        """Populate the tree and seed the detail panel once mounted."""
        tree = self.query_one("#schema-tree", Tree)
        _populate_tree(tree, self._schema)
        detail = self.query_one("#schema-detail", Static)
        detail.update(_detail_text(SchemaNodeData(kind="database"), self._schema))

    # ── Event handlers ──────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        """Rebuild the tree to show only nodes matching the filter."""
        if event.input.id != "schema-filter":
            return
        filtered = _filter_schema(self._schema, event.value) if self._schema is not None else None
        tree = self.query_one("#schema-tree", Tree)
        _populate_tree(tree, filtered)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[SchemaNodeData]) -> None:
        """Update the detail panel for the highlighted node."""
        self._update_detail(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected[SchemaNodeData]) -> None:
        """Update the detail panel for the selected node."""
        self._update_detail(event.node)

    def _update_detail(self, node: TreeNode[SchemaNodeData]) -> None:
        data = node.data if node.data is not None else SchemaNodeData(kind="database")
        detail = self.query_one("#schema-detail", Static)
        detail.update(_detail_text(data, self._schema))
