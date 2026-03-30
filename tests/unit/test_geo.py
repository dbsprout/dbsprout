"""Tests for dbsprout.generate.geo — geo coherence lookup tables."""

from __future__ import annotations

from dbsprout.generate.geo import GeoLookup, apply_geo_coherence, detect_geo_columns
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    TableSchema,
)


def _col(name: str, data_type: ColumnType = ColumnType.VARCHAR) -> ColumnSchema:
    return ColumnSchema(name=name, data_type=data_type)


class TestGeoLookupSample:
    def test_returns_correct_count(self) -> None:
        """Requested num_rows returned."""
        lookup = GeoLookup()
        result = lookup.sample(10, seed=42)
        assert len(result) == 10

    def test_has_city_state_zip_keys(self) -> None:
        """Each dict has city, state, zip keys."""
        lookup = GeoLookup()
        result = lookup.sample(5, seed=42)
        for row in result:
            assert "city" in row
            assert "state" in row
            assert "zip" in row

    def test_coherence(self) -> None:
        """City/state/zip are real combinations from the data."""
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        data_path = (
            Path(__file__).resolve().parents[2] / "dbsprout" / "_vendor" / "geo" / "us_cities.json"
        )
        raw = json.loads(data_path.read_text())
        valid_tuples = {(e[0], e[1], e[2]) for e in raw}

        lookup = GeoLookup()
        result = lookup.sample(50, seed=42)

        for row in result:
            assert (row["city"], row["state"], row["zip"]) in valid_tuples

    def test_deterministic(self) -> None:
        """Same seed → same tuples."""
        lookup = GeoLookup()
        r1 = lookup.sample(20, seed=99)
        r2 = lookup.sample(20, seed=99)
        assert r1 == r2


class TestDetectGeoColumns:
    def test_detects_city_state_zip(self) -> None:
        """Detects standard geo column names."""
        table = TableSchema(
            name="addresses",
            columns=[
                _col("id", ColumnType.INTEGER),
                _col("city"),
                _col("state"),
                _col("zip_code"),
            ],
            primary_key=["id"],
        )
        detected = detect_geo_columns(table)
        assert "city" in detected
        assert "state" in detected
        assert "zip" in detected

    def test_detects_postal_code(self) -> None:
        """postal_code maps to zip."""
        table = TableSchema(
            name="t",
            columns=[_col("city"), _col("postal_code")],
            primary_key=[],
        )
        detected = detect_geo_columns(table)
        assert "zip" in detected

    def test_no_geo_columns(self) -> None:
        """Table without geo columns returns empty dict."""
        table = TableSchema(
            name="orders",
            columns=[_col("id", ColumnType.INTEGER), _col("total", ColumnType.FLOAT)],
            primary_key=[],
        )
        detected = detect_geo_columns(table)
        assert detected == {}

    def test_single_geo_column_not_enough(self) -> None:
        """Single geo column (no pair) returns empty dict."""
        table = TableSchema(
            name="t",
            columns=[_col("city")],
            primary_key=[],
        )
        detected = detect_geo_columns(table)
        assert detected == {}


class TestApplyGeoCoherence:
    def test_patches_geo_columns(self) -> None:
        """Patches city/state columns with coherent values."""
        table = TableSchema(
            name="customers",
            columns=[
                _col("id", ColumnType.INTEGER),
                _col("city"),
                _col("state"),
            ],
            primary_key=["id"],
        )
        rows = [
            {"id": 1, "city": "random1", "state": "XX"},
            {"id": 2, "city": "random2", "state": "YY"},
            {"id": 3, "city": "random3", "state": "ZZ"},
        ]

        result = apply_geo_coherence(table, rows, seed=42)

        # Values should be overwritten with coherent data
        for row in result:
            assert row["city"] != ""
            assert len(row["state"]) == 2  # state code

    def test_no_geo_columns_unchanged(self) -> None:
        """Table without geo columns returns rows unchanged."""
        table = TableSchema(
            name="orders",
            columns=[_col("id", ColumnType.INTEGER), _col("total", ColumnType.FLOAT)],
            primary_key=[],
        )
        rows = [{"id": 1, "total": 9.99}]

        result = apply_geo_coherence(table, rows, seed=42)

        assert result == rows
