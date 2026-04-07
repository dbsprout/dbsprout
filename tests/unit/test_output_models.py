"""Tests for shared output models."""

from __future__ import annotations

import pytest

from dbsprout.output.models import InsertResult


class TestInsertResult:
    def test_fields(self) -> None:
        result = InsertResult(tables_inserted=3, total_rows=500, duration_seconds=1.23)
        assert result.tables_inserted == 3
        assert result.total_rows == 500
        assert result.duration_seconds == 1.23

    def test_frozen(self) -> None:
        result = InsertResult(tables_inserted=1, total_rows=10, duration_seconds=0.5)
        with pytest.raises(AttributeError):
            result.total_rows = 99  # type: ignore[misc]
