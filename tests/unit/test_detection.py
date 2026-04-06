"""Tests for dbsprout.quality.detection — C2ST detection metrics."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="sklearn not installed (optional [stats] extra)")

from dbsprout.quality.detection import (  # noqa: E402
    DetectionReport,
    _build_feature_matrix,
    c2st_accuracy,
    validate_detection,
)
from dbsprout.schema.models import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)


class TestC2stAccuracy:
    """Tests for the c2st_accuracy function."""

    def test_identical_data(self) -> None:
        """Same data for real and synthetic should yield accuracy near 0.5."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (200, 3))
        score = c2st_accuracy(data, data)
        assert abs(score - 0.5) < 0.15, f"Expected ~0.5 for identical data, got {score}"

    def test_distinguishable_data(self) -> None:
        """Clearly different distributions should yield accuracy > 0.8."""
        rng = np.random.default_rng(42)
        real = rng.normal(0, 1, (200, 3))
        synthetic = rng.normal(10, 1, (200, 3))
        score = c2st_accuracy(real, synthetic)
        assert score > 0.8, f"Expected >0.8 for distinguishable data, got {score}"

    def test_empty_features(self) -> None:
        """Empty feature matrix should return 0.5."""
        # Zero columns
        empty_cols = np.empty((10, 0))
        assert c2st_accuracy(empty_cols, empty_cols) == 0.5

        # Zero rows
        empty_rows = np.empty((0, 5))
        assert c2st_accuracy(empty_rows, empty_rows) == 0.5

    def test_deterministic(self) -> None:
        """Same seed should produce identical results across two calls."""
        rng = np.random.default_rng(99)
        real = rng.normal(0, 1, (100, 4))
        synthetic = rng.normal(2, 1, (100, 4))

        score_a = c2st_accuracy(real, synthetic, seed=123)
        score_b = c2st_accuracy(real, synthetic, seed=123)
        assert score_a == score_b

    def test_subsampling(self) -> None:
        """Large datasets (>10K rows) should still work with reasonable results."""
        rng = np.random.default_rng(42)
        real = rng.normal(0, 1, (15000, 3))
        synthetic = rng.normal(0, 1, (15000, 3))
        score = c2st_accuracy(real, synthetic)
        # With same distribution, accuracy should be near 0.5
        assert 0.35 <= score <= 0.65, f"Expected ~0.5 for same distribution, got {score}"


# ── Helpers for validate_detection tests ─────────────────────────────


def _detection_schema() -> DatabaseSchema:
    """Schema with numeric + categorical columns for detection tests."""
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
                    ),
                    ColumnSchema(
                        name="age",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="salary",
                        data_type=ColumnType.FLOAT,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="city",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            )
        ],
        dialect="postgresql",
    )


def _identical_rows(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate rows from the same distribution for both real and synthetic."""
    rng = np.random.default_rng(seed)
    cities = ["NYC", "LA", "CHI", "HOU"]
    return [
        {
            "id": i,
            "age": int(rng.integers(18, 65)),
            "salary": float(rng.normal(50000, 10000)),
            "city": cities[int(rng.integers(0, len(cities)))],
        }
        for i in range(n)
    ]


def _distinct_rows_real(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate clearly separable 'real' data."""
    rng = np.random.default_rng(seed)
    return [
        {
            "id": i,
            "age": int(rng.integers(18, 30)),
            "salary": float(rng.normal(30000, 1000)),
            "city": "NYC",
        }
        for i in range(n)
    ]


def _distinct_rows_synthetic(n: int, seed: int = 99) -> list[dict[str, Any]]:
    """Generate clearly separable 'synthetic' data."""
    rng = np.random.default_rng(seed)
    return [
        {
            "id": i,
            "age": int(rng.integers(50, 80)),
            "salary": float(rng.normal(100000, 1000)),
            "city": "LA",
        }
        for i in range(n)
    ]


class TestValidateDetection:
    """Tests for the validate_detection orchestrator."""

    def test_end_to_end(self) -> None:
        """Full report with schema + data: metrics present, score in [0, 1]."""
        schema = _detection_schema()
        real = _identical_rows(100, seed=1)
        syn = _identical_rows(100, seed=2)
        report = validate_detection(
            synthetic_data={"users": syn},
            reference_data={"users": real},
            schema=schema,
        )
        assert isinstance(report, DetectionReport)
        assert len(report.metrics) > 0
        assert 0.0 <= report.overall_score <= 1.0

    def test_no_reference_data(self) -> None:
        """Empty reference dict → empty report with default 0.5 score."""
        schema = _detection_schema()
        syn = _identical_rows(50)
        report = validate_detection(
            synthetic_data={"users": syn},
            reference_data={},
            schema=schema,
        )
        assert len(report.metrics) == 0
        assert report.overall_score == 0.5
        assert report.passed is True

    def test_few_rows_skipped(self) -> None:
        """<10 rows per class → table skipped, no metrics produced."""
        schema = _detection_schema()
        real = _identical_rows(4)
        syn = _identical_rows(4)
        report = validate_detection(
            synthetic_data={"users": syn},
            reference_data={"users": real},
            schema=schema,
        )
        assert len(report.metrics) == 0
        assert report.overall_score == 0.5

    def test_threshold_pass(self) -> None:
        """Identical data → accuracy ~0.5 → below threshold 0.6 → passed=True."""
        schema = _detection_schema()
        rows = _identical_rows(100, seed=42)
        report = validate_detection(
            synthetic_data={"users": rows},
            reference_data={"users": rows},
            schema=schema,
            threshold=0.6,
        )
        assert report.passed is True
        assert report.overall_score <= 0.6

    def test_threshold_fail(self) -> None:
        """Distinguishable data → accuracy > threshold → passed=False."""
        schema = _detection_schema()
        real = _distinct_rows_real(200)
        syn = _distinct_rows_synthetic(200)
        report = validate_detection(
            synthetic_data={"users": syn},
            reference_data={"users": real},
            schema=schema,
            threshold=0.4,
        )
        assert report.passed is False
        assert report.overall_score > 0.4

    def test_no_shared_columns(self) -> None:
        """Data columns don't overlap with schema → table skipped."""
        schema = _detection_schema()
        # Rows have columns that don't match the schema
        rows = [{"x": 1, "y": 2} for _ in range(20)]
        report = validate_detection(
            synthetic_data={"users": rows},
            reference_data={"users": rows},
            schema=schema,
        )
        assert len(report.metrics) == 0
        assert report.overall_score == 0.5

    def test_all_constant_columns(self) -> None:
        """All columns have zero variance → features dropped → table skipped."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[
                        ColumnSchema(
                            name="v",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                    ],
                    primary_key=["v"],
                )
            ],
            dialect="postgresql",
        )
        # Every row has the exact same value → zero variance → dropped
        rows = [{"v": 42} for _ in range(20)]
        report = validate_detection(
            synthetic_data={"t": rows},
            reference_data={"t": rows},
            schema=schema,
        )
        assert len(report.metrics) == 0

    def test_non_numeric_value_fallback(self) -> None:
        """Non-convertible numeric values should fall back to 0.0."""
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="t",
                    columns=[
                        ColumnSchema(
                            name="val",
                            data_type=ColumnType.FLOAT,
                            nullable=False,
                            primary_key=True,
                        ),
                    ],
                    primary_key=["val"],
                )
            ],
            dialect="postgresql",
        )
        rng = np.random.default_rng(42)
        # Mix in non-numeric values among normal floats
        real = [{"val": float(rng.normal(0, 1))} for _ in range(18)]
        real.extend([{"val": "not_a_number"}, {"val": None}])
        syn = [{"val": float(rng.normal(0, 1))} for _ in range(20)]
        report = validate_detection(
            synthetic_data={"t": syn},
            reference_data={"t": real},
            schema=schema,
        )
        # Should still produce a result (non-numeric falls back to 0.0)
        assert isinstance(report, DetectionReport)


class TestBuildFeatureMatrix:
    """Tests for _build_feature_matrix edge cases."""

    def test_empty_columns(self) -> None:
        """No columns should return empty feature matrix."""
        rows = [{"a": 1}, {"a": 2}]
        result = _build_feature_matrix(rows, [], {})
        assert result.shape == (2, 0)

    def test_empty_rows(self) -> None:
        """No rows should return empty feature matrix."""
        result = _build_feature_matrix([], ["a"], {})
        assert result.shape == (0, 0)

    def test_categorical_only(self) -> None:
        """Only categorical columns should produce encoded features."""
        rows = [{"c": "a"}, {"c": "b"}, {"c": "a"}]
        result = _build_feature_matrix(rows, ["c"], {"c": ColumnType.VARCHAR})
        assert result.shape[0] == 3
        assert result.shape[1] >= 1

    def test_numeric_only(self) -> None:
        """Only numeric columns should produce scaled features."""
        rows = [{"n": 1.0}, {"n": 2.0}, {"n": 3.0}]
        result = _build_feature_matrix(rows, ["n"], {"n": ColumnType.FLOAT})
        assert result.shape == (3, 1)


class TestImportGuard:
    """Test that missing sklearn raises a helpful error."""

    def test_error_without_sklearn(self) -> None:
        """Should raise ImportError with helpful message when sklearn is missing."""
        import importlib  # noqa: PLC0415

        import dbsprout.quality.detection as mod  # noqa: PLC0415

        try:
            with patch.dict(
                "sys.modules",
                {
                    "sklearn": None,
                    "sklearn.linear_model": None,
                    "sklearn.model_selection": None,
                    "sklearn.preprocessing": None,
                },
            ):
                importlib.reload(mod)
                with pytest.raises(ImportError, match="scikit-learn"):
                    mod.validate_detection({}, {}, _detection_schema())
        finally:
            importlib.reload(mod)

    def test_c2st_accuracy_import_guard(self) -> None:
        """c2st_accuracy should also raise ImportError when sklearn is missing."""
        import importlib  # noqa: PLC0415

        import dbsprout.quality.detection as mod  # noqa: PLC0415

        try:
            with patch.dict(
                "sys.modules",
                {
                    "sklearn": None,
                    "sklearn.linear_model": None,
                    "sklearn.model_selection": None,
                    "sklearn.preprocessing": None,
                },
            ):
                importlib.reload(mod)
                with pytest.raises(ImportError, match="scikit-learn"):
                    mod.c2st_accuracy(np.ones((10, 2)), np.ones((10, 2)))
        finally:
            importlib.reload(mod)
