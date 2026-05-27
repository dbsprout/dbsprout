"""S-107: ``core.service.generate`` forwards progress + cancel hooks.

The facade must pass ``progress_callback`` / ``cancel_token`` straight through
to ``orchestrate`` (defaults ``None``), and stay byte-identical when neither is
supplied — the same parity contract S-106 established.
"""

from __future__ import annotations

import pytest

from dbsprout.config.models import DBSproutConfig
from dbsprout.core import service
from dbsprout.generate.progress import GenerationCancelled, ProgressEvent
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

_SEED = 42


def _single_table_schema() -> DatabaseSchema:
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
        ],
    )


def test_generate_forwards_progress_callback() -> None:
    """A callback handed to the facade receives the orchestrator's events."""
    events: list[ProgressEvent] = []
    service.generate(
        _single_table_schema(),
        DBSproutConfig(),
        seed=_SEED,
        default_rows=5,
        progress_callback=events.append,
    )
    assert [e.phase for e in events] == ["table_start", "table_done"]
    assert events[1].table == "users"
    assert events[1].rows_in_table == 5
    assert events[1].total_rows == 5


def test_generate_forwards_cancel_token() -> None:
    """A cancel token handed to the facade aborts generation."""
    with pytest.raises(GenerationCancelled):
        service.generate(
            _single_table_schema(),
            DBSproutConfig(),
            seed=_SEED,
            default_rows=5,
            cancel_token=lambda: True,
        )


def test_generate_parity_noop_callback() -> None:
    """A no-op callback through the facade does not change output (parity)."""
    schema = _single_table_schema()
    config = DBSproutConfig()
    base = service.generate(schema, config, seed=_SEED, default_rows=10)
    with_cb = service.generate(
        schema,
        config,
        seed=_SEED,
        default_rows=10,
        progress_callback=lambda _e: None,
    )
    assert with_cb.tables_data == base.tables_data
