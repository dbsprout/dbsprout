"""Tests for S-107 progress hooks + cooperative cancel on ``orchestrate``.

Covers the new ``dbsprout.generate.progress`` module (``ProgressEvent`` model,
``GenerationCancelled`` exception, cancel-token normalization) and the two new
optional ``orchestrate(...)`` hooks. The #1 acceptance criterion is **parity**:
with ``progress_callback=None`` and ``cancel_token=None`` the output is
byte-identical to the pre-S-107 pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.generate.progress import GenerationCancelled, ProgressEvent
from dbsprout.plugins.dispatch import resolve_writer
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 42
_ROWS = 25


def _users_orders_schema() -> DatabaseSchema:
    """Two tables: users (parent) → orders (child with FK)."""
    return DatabaseSchema(
        tables=[
            TableSchema(
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
                        nullable=False,
                        unique=True,
                    ),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False),
                    ColumnSchema(name="amount", data_type=ColumnType.FLOAT, nullable=False),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    ),
                ],
            ),
        ],
    )


def _read_sql_bytes(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.sql"))}


class TestProgressEventModel:
    def test_progress_event_is_frozen(self) -> None:
        """ProgressEvent is an immutable (frozen) Pydantic model."""
        event = ProgressEvent(
            phase="table_start",
            table="users",
            tables_done=0,
            tables_total=2,
            rows_in_table=0,
            total_rows=0,
        )
        with pytest.raises(pydantic_validation_error()):
            event.tables_done = 5  # type: ignore[misc]

    def test_progress_event_defaults(self) -> None:
        """Optional fields default sensibly (no table, no message, zero counts)."""
        event = ProgressEvent(phase="table_start")
        assert event.table is None
        assert event.message is None
        assert event.tables_done == 0
        assert event.tables_total == 0
        assert event.rows_in_table == 0
        assert event.total_rows == 0


class TestParity:
    """AC #1: a no-op callback must not perturb output (byte-identical)."""

    def test_noop_callback_byte_identical_tables_data(self) -> None:
        schema = _users_orders_schema()
        config = DBSproutConfig()

        baseline = orchestrate(schema, config, seed=_SEED, default_rows=_ROWS)
        with_cb = orchestrate(
            schema,
            config,
            seed=_SEED,
            default_rows=_ROWS,
            progress_callback=lambda _e: None,
        )

        assert with_cb.tables_data == baseline.tables_data
        assert with_cb.insertion_order == baseline.insertion_order
        assert with_cb.total_rows == baseline.total_rows

    def test_noop_callback_byte_identical_sql(self, tmp_path: Path) -> None:
        schema = _users_orders_schema()
        config = DBSproutConfig()
        writer = resolve_writer("sql")

        base = orchestrate(schema, config, seed=_SEED, default_rows=_ROWS)
        base_dir = tmp_path / "baseline"
        writer.write(base.tables_data, schema, base.insertion_order, base_dir, dialect="postgresql")

        cb = orchestrate(
            schema,
            config,
            seed=_SEED,
            default_rows=_ROWS,
            progress_callback=lambda _e: None,
        )
        cb_dir = tmp_path / "with_cb"
        writer.write(cb.tables_data, schema, cb.insertion_order, cb_dir, dialect="postgresql")

        assert _read_sql_bytes(cb_dir) == _read_sql_bytes(base_dir)
        assert _read_sql_bytes(base_dir), "golden capture produced no SQL files"


class TestCallbackEmission:
    def test_callback_emits_start_and_done_per_table(self) -> None:
        schema = _users_orders_schema()
        config = DBSproutConfig()
        events: list[ProgressEvent] = []

        orchestrate(
            schema,
            config,
            seed=_SEED,
            default_rows=5,
            progress_callback=events.append,
        )

        # Two tables → 4 events, strictly ordered start/done per table.
        phases = [e.phase for e in events]
        assert phases == ["table_start", "table_done", "table_start", "table_done"]

        # Every event knows the right denominator.
        assert all(e.tables_total == 2 for e in events)

        # tables_done: 0 (start1), 1 (done1), 1 (start2), 2 (done2).
        assert [e.tables_done for e in events] == [0, 1, 1, 2]

        # rows_in_table: 0 on start, len(rows) on done.
        assert events[0].rows_in_table == 0
        assert events[1].rows_in_table == 5
        assert events[2].rows_in_table == 0
        assert events[3].rows_in_table == 5

        # total_rows is monotonic non-decreasing; ends at the grand total.
        totals = [e.total_rows for e in events]
        assert totals == sorted(totals)
        assert totals[-1] == 10

        # Tables are named in insertion order (parent before child).
        assert events[0].table == "users"
        assert events[2].table == "orders"

    def test_callback_skips_excluded_tables(self) -> None:
        from dbsprout.config.models import TableOverride  # noqa: PLC0415

        schema = _users_orders_schema()
        config = DBSproutConfig(tables={"orders": TableOverride(exclude=True)})
        events: list[ProgressEvent] = []

        orchestrate(
            schema,
            config,
            seed=_SEED,
            default_rows=5,
            progress_callback=events.append,
        )

        # Only users is generated → 2 events, denominator is 1.
        assert [e.phase for e in events] == ["table_start", "table_done"]
        assert all(e.table == "users" for e in events)
        assert all(e.tables_total == 1 for e in events)


class TestCancel:
    def test_cancel_callable_stops_before_first_table(self) -> None:
        schema = _users_orders_schema()
        config = DBSproutConfig()
        events: list[ProgressEvent] = []

        with pytest.raises(GenerationCancelled) as exc:
            orchestrate(
                schema,
                config,
                seed=_SEED,
                default_rows=5,
                progress_callback=events.append,
                cancel_token=lambda: True,
            )

        # Cancelled at the very top → no table emitted anything.
        assert events == []
        assert exc.value.tables_done == 0
        assert exc.value.tables_total == 2

    def test_cancel_after_first_table(self) -> None:
        schema = _users_orders_schema()
        config = DBSproutConfig()
        events: list[ProgressEvent] = []
        checks = {"n": 0}

        def token() -> bool:
            # First check (before users) → False; second (before orders) → True.
            checks["n"] += 1
            return checks["n"] >= 2

        with pytest.raises(GenerationCancelled) as exc:
            orchestrate(
                schema,
                config,
                seed=_SEED,
                default_rows=5,
                progress_callback=events.append,
                cancel_token=token,
            )

        # Exactly one table fully generated before the cancel.
        assert [e.phase for e in events] == ["table_start", "table_done"]
        assert events[1].table == "users"
        assert exc.value.tables_done == 1

    def test_cancel_token_object_form(self) -> None:
        class Token:
            def is_cancelled(self) -> bool:
                return True

        schema = _users_orders_schema()
        config = DBSproutConfig()

        with pytest.raises(GenerationCancelled):
            orchestrate(schema, config, seed=_SEED, default_rows=5, cancel_token=Token())

    def test_no_cancel_token_completes(self) -> None:
        """``cancel_token=None`` never cancels (default path)."""
        schema = _users_orders_schema()
        config = DBSproutConfig()
        result = orchestrate(schema, config, seed=_SEED, default_rows=5, cancel_token=None)
        assert result.total_tables == 2


def pydantic_validation_error() -> type[Exception]:
    """Return the Pydantic ValidationError class (frozen-assignment guard)."""
    from pydantic import ValidationError  # noqa: PLC0415

    return ValidationError
