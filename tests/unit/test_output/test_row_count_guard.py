"""Orchestrator max_rows_per_table guard (S-094)."""

from __future__ import annotations

import pytest

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)


def _schema() -> DatabaseSchema:
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
                ],
                primary_key=["id"],
            ),
        ],
    )


def test_orchestrate_row_count_guard_rejects_excess() -> None:
    cfg = DBSproutConfig.model_validate({"generation": {"max_rows_per_table": 10}})
    with pytest.raises(ValueError, match=r"max_rows_per_table"):
        orchestrate(_schema(), cfg, seed=1, default_rows=50)


def test_orchestrate_row_count_guard_allows_within_limit() -> None:
    cfg = DBSproutConfig.model_validate({"generation": {"max_rows_per_table": 100}})
    result = orchestrate(_schema(), cfg, seed=1, default_rows=50)
    assert result.total_rows == 50


def test_orchestrate_no_guard_when_unset() -> None:
    cfg = DBSproutConfig()
    result = orchestrate(_schema(), cfg, seed=1, default_rows=50)
    assert result.total_rows == 50
