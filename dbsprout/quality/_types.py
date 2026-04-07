"""Shared column type classifications for quality metrics."""

from __future__ import annotations

from dbsprout.schema.models import ColumnType

NUMERIC_TYPES: frozenset[ColumnType] = frozenset(
    {
        ColumnType.INTEGER,
        ColumnType.BIGINT,
        ColumnType.SMALLINT,
        ColumnType.FLOAT,
        ColumnType.DECIMAL,
    }
)

CATEGORICAL_TYPES: frozenset[ColumnType] = frozenset(
    {
        ColumnType.VARCHAR,
        ColumnType.TEXT,
        ColumnType.ENUM,
        ColumnType.BOOLEAN,
    }
)
