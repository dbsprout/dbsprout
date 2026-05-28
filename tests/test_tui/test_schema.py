"""TUI schema browser tests (S-088).

Textual is an optional ``[tui]`` extra; the whole module is skipped when it is
absent (mirrors ``tests/test_tui/test_app.py`` and the pg/mongo guards). Async
widget scenarios wrap :meth:`textual.app.App.run_test` in ``asyncio.run(...)``
inside synchronous test functions — the repo has no async pytest plugin.

Pure helpers (``_filter_schema``, ``_detail_text``, label/summary builders) are
exercised directly without a running app for fast, deterministic coverage.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("textual", reason="textual absent (pip install dbsprout[tui])")

from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Tree

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.tui.app import DBSproutApp
from dbsprout.tui.screens.schema import (
    SchemaBrowser,
    SchemaNodeData,
    _column_constraint_labels,
    _column_summary,
    _detail_text,
    _filter_schema,
    _table_summary,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from typing import TypeVar

    _T = TypeVar("_T")


# ── Fixtures ──────────────────────────────────────────────────────────────


def _sample_schema() -> DatabaseSchema:
    """A small two-table schema covering PK, FK (incl. self-ref), UNIQUE, CHECK."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
                autoincrement=True,
            ),
            ColumnSchema(
                name="email",
                data_type=ColumnType.VARCHAR,
                max_length=255,
                nullable=False,
                unique=True,
            ),
            ColumnSchema(
                name="age",
                data_type=ColumnType.INTEGER,
                nullable=True,
                check_constraint="age >= 0",
            ),
            ColumnSchema(
                name="manager_id",
                data_type=ColumnType.INTEGER,
                nullable=True,
            ),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["manager_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(
                name="id",
                data_type=ColumnType.INTEGER,
                nullable=False,
                primary_key=True,
            ),
            ColumnSchema(
                name="user_id",
                data_type=ColumnType.INTEGER,
                nullable=False,
            ),
            ColumnSchema(
                name="total",
                data_type=ColumnType.DECIMAL,
                precision=10,
                scale=2,
                nullable=False,
                default="0.00",
            ),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, orders], dialect="sqlite", source="test")


def _run(coro: Awaitable[_T]) -> _T:
    """Run an async pilot coroutine in a fresh event loop."""
    return asyncio.run(coro)  # type: ignore[arg-type]


# ── Pure helper: constraint labels ──────────────────────────────────────────


def test_constraint_labels_for_pk_column() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    col = users.get_column("id")
    assert col is not None
    labels = _column_constraint_labels(users, col)
    assert "PK" in labels
    assert "NOT NULL" in labels


def test_constraint_labels_for_unique_column() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    col = users.get_column("email")
    assert col is not None
    labels = _column_constraint_labels(users, col)
    assert "UNIQUE" in labels
    assert "NOT NULL" in labels


def test_constraint_labels_for_check_and_nullable_column() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    col = users.get_column("age")
    assert col is not None
    labels = _column_constraint_labels(users, col)
    assert any(label.startswith("CHECK") for label in labels)
    assert "NULLABLE" in labels


def test_constraint_labels_for_fk_column_show_reference() -> None:
    schema = _sample_schema()
    orders = schema.get_table("orders")
    assert orders is not None
    col = orders.get_column("user_id")
    assert col is not None
    labels = _column_constraint_labels(orders, col)
    assert "FK -> users.id" in labels


# ── Pure helper: summaries ──────────────────────────────────────────────────


def test_column_summary_includes_type_length_and_constraints() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    col = users.get_column("email")
    assert col is not None
    summary = _column_summary(users, col)
    assert "email" in summary
    assert "VARCHAR(255)" in summary
    assert "UNIQUE" in summary


def test_column_summary_decimal_precision_scale() -> None:
    schema = _sample_schema()
    orders = schema.get_table("orders")
    assert orders is not None
    col = orders.get_column("total")
    assert col is not None
    summary = _column_summary(orders, col)
    assert "DECIMAL(10,2)" in summary


def test_table_summary_counts_columns_and_constraints() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    summary = _table_summary(users)
    assert "users" in summary
    assert "4 column" in summary


def test_table_summary_singular_words_for_one_column_one_constraint() -> None:
    table = TableSchema(
        name="solo",
        columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)],
        primary_key=["id"],
    )
    summary = _table_summary(table)
    assert "1 column," in summary
    assert "1 constraint)" in summary


def test_column_summary_decimal_without_scale() -> None:
    table = TableSchema(
        name="t",
        columns=[ColumnSchema(name="amt", data_type=ColumnType.DECIMAL, precision=8)],
    )
    col = table.get_column("amt")
    assert col is not None
    assert "DECIMAL(8)" in _column_summary(table, col)


# ── Pure helper: filter ─────────────────────────────────────────────────────


def test_filter_empty_query_returns_all_tables() -> None:
    schema = _sample_schema()
    filtered = _filter_schema(schema, "")
    assert filtered.table_names() == schema.table_names()


def test_filter_by_table_name_keeps_only_matching_table() -> None:
    schema = _sample_schema()
    filtered = _filter_schema(schema, "order")
    assert filtered.table_names() == ["orders"]
    orders = filtered.get_table("orders")
    assert orders is not None
    assert len(orders.columns) == 3  # full table kept on table-name match


def test_filter_by_column_name_surfaces_table_with_only_matching_columns() -> None:
    schema = _sample_schema()
    filtered = _filter_schema(schema, "email")
    assert filtered.table_names() == ["users"]
    users = filtered.get_table("users")
    assert users is not None
    assert [c.name for c in users.columns] == ["email"]


def test_filter_no_match_returns_empty_tables() -> None:
    schema = _sample_schema()
    filtered = _filter_schema(schema, "zzz_nonexistent")
    assert filtered.tables == []


def test_filter_is_case_insensitive() -> None:
    schema = _sample_schema()
    filtered = _filter_schema(schema, "USERS")
    assert filtered.table_names() == ["users"]


# ── Pure helper: detail text ────────────────────────────────────────────────


def test_detail_text_for_table() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    data = SchemaNodeData(kind="table", table=users)
    text = _detail_text(data, schema)
    assert "users" in text
    assert "4" in text  # column count


def test_detail_text_for_column_shows_type_and_constraints() -> None:
    schema = _sample_schema()
    users = schema.get_table("users")
    assert users is not None
    col = users.get_column("email")
    assert col is not None
    data = SchemaNodeData(kind="column", table=users, column=col)
    text = _detail_text(data, schema)
    assert "VARCHAR(255)" in text
    assert "UNIQUE" in text
    assert "NOT NULL" in text


def test_detail_text_for_fk_constraint_shows_reference() -> None:
    schema = _sample_schema()
    orders = schema.get_table("orders")
    assert orders is not None
    col = orders.get_column("user_id")
    assert col is not None
    data = SchemaNodeData(
        kind="constraint",
        table=orders,
        column=col,
        constraint_label="FK -> users.id",
    )
    text = _detail_text(data, schema)
    assert "users.id" in text


def test_detail_text_database_kind_lists_tables() -> None:
    schema = _sample_schema()
    data = SchemaNodeData(kind="database")
    text = _detail_text(data, schema)
    assert "2" in text  # table count


def test_detail_text_empty_schema_message() -> None:
    data = SchemaNodeData(kind="database")
    text = _detail_text(data, None)
    assert "no snapshot" in text.lower()


def test_detail_text_includes_table_and_column_comments() -> None:
    table = TableSchema(
        name="notes",
        columns=[
            ColumnSchema(
                name="body",
                data_type=ColumnType.TEXT,
                comment="free-form note text",
            ),
        ],
        comment="user-authored notes",
    )
    schema = DatabaseSchema(tables=[table])
    table_text = _detail_text(SchemaNodeData(kind="table", table=table), schema)
    assert "user-authored notes" in table_text
    col = table.get_column("body")
    assert col is not None
    column_text = _detail_text(SchemaNodeData(kind="column", table=table, column=col), schema)
    assert "free-form note text" in column_text


# ── Widget: tree structure ──────────────────────────────────────────────────


def test_tree_builds_database_table_column_constraint_hierarchy() -> None:
    async def _scenario() -> None:
        schema = _sample_schema()
        app = DBSproutApp(schema=schema)
        async with app.run_test() as pilot:
            await pilot.press("s")  # switch to Schema tab
            await pilot.pause()
            browser = pilot.app.query_one(SchemaBrowser)
            tree = browser.query_one(Tree)
            root = tree.root
            # root → tables
            table_labels = [str(node.label) for node in root.children]
            assert any("users" in label for label in table_labels)
            assert any("orders" in label for label in table_labels)
            # table → columns
            users_node = next(n for n in root.children if "users" in str(n.label))
            users_node.expand()
            await pilot.pause()
            column_labels = [str(c.label) for c in users_node.children]
            assert any("email" in label for label in column_labels)
            # column → constraints
            email_node = next(c for c in users_node.children if "email" in str(c.label))
            email_node.expand()
            await pilot.pause()
            constraint_labels = [str(x.label) for x in email_node.children]
            assert any("UNIQUE" in label for label in constraint_labels)

    _run(_scenario())


def test_tree_renders_fk_reference() -> None:
    async def _scenario() -> None:
        schema = _sample_schema()
        app = DBSproutApp(schema=schema)
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            browser = pilot.app.query_one(SchemaBrowser)
            tree = browser.query_one(Tree)
            orders_node = next(n for n in tree.root.children if "orders" in str(n.label))
            orders_node.expand()
            await pilot.pause()
            user_id_node = next(c for c in orders_node.children if "user_id" in str(c.label))
            user_id_node.expand()
            await pilot.pause()
            labels = [str(x.label) for x in user_id_node.children]
            assert any("FK -> users.id" in label for label in labels)

    _run(_scenario())


# ── Widget: filter ──────────────────────────────────────────────────────────


def test_input_filter_narrows_visible_tables() -> None:
    async def _scenario() -> None:
        schema = _sample_schema()
        app = DBSproutApp(schema=schema)
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            browser = pilot.app.query_one(SchemaBrowser)
            filter_input = browser.query_one(Input)
            filter_input.value = "order"
            await pilot.pause()
            tree = browser.query_one(Tree)
            table_labels = [str(n.label) for n in tree.root.children]
            assert any("orders" in label for label in table_labels)
            assert not any("users" in label for label in table_labels)

    _run(_scenario())


# ── Widget: detail panel ────────────────────────────────────────────────────


def test_highlighting_node_updates_detail_panel() -> None:
    async def _scenario() -> None:
        schema = _sample_schema()
        app = DBSproutApp(schema=schema)
        async with app.run_test() as pilot:
            await pilot.press("s")
            await pilot.pause()
            browser = pilot.app.query_one(SchemaBrowser)
            tree = browser.query_one(Tree)
            users_node = next(n for n in tree.root.children if "users" in str(n.label))
            tree.select_node(users_node)
            await pilot.pause()
            detail = browser.query_one("#schema-detail", Static)
            rendered = str(detail.render())
            assert "users" in rendered

    _run(_scenario())


# ── Widget: empty / missing snapshot ────────────────────────────────────────


class _BrowserHarness(App[None]):
    """Minimal host that mounts a ``SchemaBrowser`` with an injected schema.

    Mounting the widget directly (rather than through ``DBSproutApp``) keeps the
    empty-state test deterministic: it never touches the on-disk snapshot store.
    """

    def __init__(self, schema: DatabaseSchema | None) -> None:
        super().__init__()
        self._schema = schema

    def compose(self) -> ComposeResult:
        yield SchemaBrowser(self._schema)


def test_empty_schema_renders_friendly_message() -> None:
    async def _scenario() -> None:
        app = _BrowserHarness(schema=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            browser = pilot.app.query_one(SchemaBrowser)
            detail = browser.query_one("#schema-detail", Static)
            rendered = str(detail.render())
            assert "no snapshot" in rendered.lower()
            tree = browser.query_one(Tree)
            assert not tree.root.children  # empty schema → no table nodes

    _run(_scenario())


# ── App wiring ──────────────────────────────────────────────────────────────


def test_schema_tab_hosts_schema_browser() -> None:
    async def _scenario() -> None:
        app = DBSproutApp(schema=_sample_schema())
        async with app.run_test() as pilot:
            assert pilot.app.query(SchemaBrowser)

    _run(_scenario())


def test_app_loads_latest_snapshot_when_no_schema_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DBSproutApp(schema=None)`` falls back to the snapshot loader on compose."""
    import dbsprout.tui.app as app_module  # noqa: PLC0415

    sample = _sample_schema()
    monkeypatch.setattr(app_module, "_load_latest_schema", lambda: sample)

    async def _scenario() -> None:
        app = DBSproutApp(schema=None)
        async with app.run_test() as pilot:
            browser = pilot.app.query_one(SchemaBrowser)
            tree = browser.query_one(Tree)
            await pilot.pause()
            labels = [str(n.label) for n in tree.root.children]
            assert any("users" in label for label in labels)

    _run(_scenario())


def test_load_latest_schema_returns_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_load_latest_schema`` delegates to ``SnapshotStore.load_latest``."""
    import dbsprout.migrate.snapshot as snap_module  # noqa: PLC0415
    from dbsprout.tui.app import _load_latest_schema  # noqa: PLC0415

    sample = _sample_schema()
    monkeypatch.setattr(snap_module.SnapshotStore, "load_latest", lambda _self: sample)
    assert _load_latest_schema() is sample
