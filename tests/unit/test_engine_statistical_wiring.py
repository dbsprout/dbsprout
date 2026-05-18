"""Wiring tests for the statistical engine (S-071).

Covers registry dispatch and orchestrator integration, including the
reference-data fitting path and the no-reference fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.engines.statistical import StatisticalEngine
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.plugins.dispatch import resolve_engine
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    import pytest


def _col(
    name: str,
    *,
    nullable: bool = True,
    pk: bool = False,
    autoincrement: bool = False,
    data_type: ColumnType = ColumnType.INTEGER,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=data_type,
        nullable=nullable,
        primary_key=pk,
        autoincrement=autoincrement,
    )


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    _col("id", nullable=False, pk=True, autoincrement=True),
                    _col("age", nullable=False, data_type=ColumnType.INTEGER),
                    _col("tier", data_type=ColumnType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            TableSchema(
                name="orders",
                columns=[
                    _col("id", nullable=False, pk=True, autoincrement=True),
                    _col("user_id", nullable=False, data_type=ColumnType.INTEGER),
                    _col("total", nullable=False, data_type=ColumnType.FLOAT),
                ],
                primary_key=["id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    )
                ],
            ),
        ]
    )


def _reference() -> dict[str, list[dict[str, object]]]:
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(0)
    users = [
        {
            "id": None,
            "age": int(rng.integers(18, 80)),
            "tier": rng.choice(["free", "pro"], p=[0.7, 0.3]),
        }
        for _ in range(300)
    ]
    orders = [
        {"id": None, "user_id": None, "total": float(rng.normal(120, 30))} for _ in range(300)
    ]
    return {"users": users, "orders": orders}


class TestDispatch:
    def test_resolve_engine_returns_statistical(self) -> None:
        engine = resolve_engine("statistical", seed=7)
        assert isinstance(engine, StatisticalEngine)


class TestOrchestratorIntegration:
    def test_statistical_with_reference_keeps_fk_integrity(self) -> None:
        from dbsprout.quality.integrity import validate_integrity  # noqa: PLC0415

        schema = _schema()
        config = DBSproutConfig()
        result = orchestrate(
            schema,
            config,
            seed=42,
            default_rows=50,
            engine="statistical",
            reference_data=_reference(),
        )
        assert result.total_tables == 2
        assert result.total_rows == 100
        report = validate_integrity(result.tables_data, schema)
        assert report.passed

    def test_statistical_without_reference_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        schema = _schema()
        config = DBSproutConfig()
        with caplog.at_level(logging.WARNING):
            result = orchestrate(
                schema,
                config,
                seed=42,
                default_rows=20,
                engine="statistical",
            )
        assert result.total_rows == 40
        assert any("insufficient" in m.lower() for m in caplog.messages)
