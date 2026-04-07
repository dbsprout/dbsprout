"""Tests for shared quality type classifications."""

from __future__ import annotations

from dbsprout.quality._types import CATEGORICAL_TYPES, NUMERIC_TYPES
from dbsprout.schema.models import ColumnType


class TestNumericTypes:
    """Tests for NUMERIC_TYPES constant."""

    def test_numeric_types_is_frozenset(self) -> None:
        assert isinstance(NUMERIC_TYPES, frozenset)

    def test_numeric_types_contains_exactly(self) -> None:
        expected = frozenset(
            {
                ColumnType.INTEGER,
                ColumnType.BIGINT,
                ColumnType.SMALLINT,
                ColumnType.FLOAT,
                ColumnType.DECIMAL,
            }
        )
        assert expected == NUMERIC_TYPES


class TestCategoricalTypes:
    """Tests for CATEGORICAL_TYPES constant."""

    def test_categorical_types_is_frozenset(self) -> None:
        assert isinstance(CATEGORICAL_TYPES, frozenset)

    def test_categorical_types_contains_exactly(self) -> None:
        expected = frozenset(
            {
                ColumnType.VARCHAR,
                ColumnType.TEXT,
                ColumnType.ENUM,
                ColumnType.BOOLEAN,
            }
        )
        assert expected == CATEGORICAL_TYPES


class TestTypeDisjointness:
    """Ensure type sets don't overlap."""

    def test_types_are_disjoint(self) -> None:
        assert NUMERIC_TYPES.isdisjoint(CATEGORICAL_TYPES)
