"""Tests for dbsprout.quality.fidelity — fidelity metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

scipy = pytest.importorskip("scipy", reason="scipy not installed (optional [stats] extra)")

from dbsprout.quality.fidelity import (  # noqa: E402
    FidelityReport,
    cardinality_similarity,
    correlation_similarity,
    ks_complement,
    load_reference_csv,
    tv_complement,
    validate_fidelity,
)
from dbsprout.schema.models import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path


def _numeric_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="metrics",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="score",
                        data_type=ColumnType.FLOAT,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="rating",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
        dialect="postgresql",
    )


def _mixed_schema() -> DatabaseSchema:
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
                        name="city",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="active",
                        data_type=ColumnType.BOOLEAN,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
        dialect="postgresql",
    )


# ── KS Complement ──────────────────────────────────────────────────


class TestKsComplement:
    def test_identical_distributions(self) -> None:
        """Same data should give score close to 1.0."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        score = ks_complement(data, data)
        assert score >= 0.99

    def test_different_distributions(self) -> None:
        """Very different data should give low score."""
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        synthetic = [100.0, 200.0, 300.0, 400.0, 500.0]
        score = ks_complement(real, synthetic)
        assert score < 0.5

    def test_empty_input(self) -> None:
        """Empty lists should return 1.0 (no data to compare)."""
        assert ks_complement([], [1.0, 2.0]) == 1.0
        assert ks_complement([1.0], []) == 1.0
        assert ks_complement([], []) == 1.0

    def test_score_range(self) -> None:
        """Score must be in [0, 1]."""
        real = [1.0, 2.0, 3.0]
        synthetic = [4.0, 5.0, 6.0]
        score = ks_complement(real, synthetic)
        assert 0.0 <= score <= 1.0


# ── TV Complement ──────────────────────────────────────────────────


class TestTvComplement:
    def test_identical_categories(self) -> None:
        """Same distribution should give 1.0."""
        data = ["a", "b", "a", "b"]
        score = tv_complement(data, data)
        assert score >= 0.99

    def test_disjoint_categories(self) -> None:
        """Completely different categories should give score close to 0."""
        real = ["a", "a", "a"]
        synthetic = ["x", "x", "x"]
        score = tv_complement(real, synthetic)
        assert score < 0.1

    def test_partial_overlap(self) -> None:
        """Partial overlap should be between 0 and 1."""
        real = ["a", "a", "b", "b"]
        synthetic = ["a", "a", "a", "c"]
        score = tv_complement(real, synthetic)
        assert 0.0 < score < 1.0

    def test_empty_input(self) -> None:
        """Empty lists should return 1.0."""
        assert tv_complement([], ["a"]) == 1.0
        assert tv_complement(["a"], []) == 1.0

    def test_score_range(self) -> None:
        """Score must be in [0, 1]."""
        score = tv_complement(["a", "b"], ["c", "d"])
        assert 0.0 <= score <= 1.0


# ── Correlation Similarity ─────────────────────────────────────────


class TestCorrelationSimilarity:
    def test_identical_matrices(self) -> None:
        """Same column data should give score close to 1.0."""
        cols_real = {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]}
        score = correlation_similarity(cols_real, cols_real)
        assert score >= 0.99

    def test_single_column(self) -> None:
        """Single column — no correlation to compute, return 1.0."""
        cols = {"a": [1.0, 2.0, 3.0]}
        score = correlation_similarity(cols, cols)
        assert score == 1.0

    def test_different_correlations(self) -> None:
        """Different correlation structures should give lower score."""
        real = {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]}
        synthetic = {"a": [1.0, 2.0, 3.0, 4.0], "b": [8.0, 6.0, 4.0, 2.0]}
        score = correlation_similarity(real, synthetic)
        assert score < 0.8

    def test_empty_columns(self) -> None:
        """Empty column dict should return 1.0."""
        assert correlation_similarity({}, {}) == 1.0


# ── Cardinality Similarity ─────────────────────────────────────────


class TestCardinalitySimilarity:
    def test_same_cardinality(self) -> None:
        """Same number of unique values and similar distribution → high score."""
        data = ["a", "b", "c", "a", "b", "c"]
        score = cardinality_similarity(data, data)
        assert score >= 0.99

    def test_different_cardinality(self) -> None:
        """Very different unique value counts → lower score."""
        real = ["a", "a", "a", "a"]
        synthetic = ["a", "b", "c", "d"]
        score = cardinality_similarity(real, synthetic)
        assert score < 1.0

    def test_empty_input(self) -> None:
        """Empty lists should return 1.0."""
        assert cardinality_similarity([], []) == 1.0
        assert cardinality_similarity([], ["a"]) == 1.0


# ── validate_fidelity (end-to-end) ─────────────────────────────────


class TestValidateFidelity:
    def test_end_to_end_mixed_columns(self) -> None:
        """Full report with mixed column types."""
        schema = _mixed_schema()
        synthetic: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "age": 25, "city": "NYC", "active": True},
                {"id": 2, "age": 30, "city": "LA", "active": False},
                {"id": 3, "age": 35, "city": "NYC", "active": True},
            ],
        }
        reference: dict[str, list[dict[str, Any]]] = {
            "users": [
                {"id": 1, "age": 26, "city": "NYC", "active": True},
                {"id": 2, "age": 31, "city": "LA", "active": False},
                {"id": 3, "age": 36, "city": "NYC", "active": True},
            ],
        }
        report = validate_fidelity(synthetic, reference, schema)

        assert isinstance(report, FidelityReport)
        assert len(report.metrics) > 0
        assert 0.0 <= report.overall_score <= 1.0

    def test_no_reference_data(self) -> None:
        """Empty reference data should produce empty report."""
        schema = _mixed_schema()
        synthetic: dict[str, list[dict[str, Any]]] = {
            "users": [{"id": 1, "age": 25, "city": "NYC", "active": True}],
        }
        report = validate_fidelity(synthetic, {}, schema)

        assert len(report.metrics) == 0
        assert report.overall_score == 0.0

    def test_identical_data_high_score(self) -> None:
        """Identical synthetic and reference data should give high overall score."""
        schema = _numeric_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": i, "score": float(i) * 1.5, "rating": i % 5} for i in range(20)],
        }
        report = validate_fidelity(data, data, schema)

        assert report.overall_score >= 0.9

    def test_threshold_pass_fail(self) -> None:
        """Report should pass/fail based on threshold."""
        schema = _numeric_schema()
        data: dict[str, list[dict[str, Any]]] = {
            "metrics": [{"id": i, "score": float(i), "rating": i} for i in range(10)],
        }
        report_high = validate_fidelity(data, data, schema, threshold=0.5)
        assert report_high.passed is True

        report_impossible = validate_fidelity(data, data, schema, threshold=1.1)
        assert report_impossible.passed is False


# ── Reference CSV Loading ──────────────────────────────────────────


class TestLoadReferenceCsv:
    def test_load_csv(self, tmp_path: Path) -> None:
        """Load a CSV file into list[dict]."""
        csv_path = tmp_path / "users.csv"
        csv_path.write_text("id,age,city\n1,25,NYC\n2,30,LA\n")
        rows = load_reference_csv(csv_path, "users")

        assert len(rows) == 2
        assert rows[0]["city"] == "NYC"

    def test_numeric_coercion(self, tmp_path: Path) -> None:
        """Numeric strings should be coerced to int/float."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("id,score\n1,3.14\n2,2.71\n")
        rows = load_reference_csv(csv_path, "data")

        assert isinstance(rows[0]["score"], float)
        assert isinstance(rows[0]["id"], int)

    def test_missing_file(self, tmp_path: Path) -> None:
        """Missing CSV file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_reference_csv(tmp_path / "nope.csv", "t")


# ── Import Guard ───────────────────────────────────────────────────


class TestImportGuard:
    def test_error_without_scipy(self, tmp_path: Path) -> None:
        """Should raise ImportError with helpful message when scipy is missing."""
        import importlib  # noqa: PLC0415

        import dbsprout.quality.fidelity as mod  # noqa: PLC0415

        try:
            with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
                importlib.reload(mod)
                with pytest.raises(ImportError, match="scipy"):
                    mod.validate_fidelity({}, {}, _numeric_schema())
        finally:
            importlib.reload(mod)
