"""Tests for dbsprout.generate.engines.statistical — GaussianCopula engine.

S-071: GaussianCopula statistical engine. Implemented from scratch with
scipy + numpy (no SDV / sdmetrics in the dependency set).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any

import numpy as np
import pytest
from scipy import stats

from dbsprout.generate.engines.statistical import (
    MIN_SAMPLE,
    StatisticalEngine,
    _coerce_datetime_ordinal,
    _coerce_numeric,
    _column,
    _empirical_marginal,
    _fit_copula,
    _inverse_marginal,
    _nearest_psd,
    _ordinal_to_value,
    _sample_categorical,
    _to_normal_scores,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.spec.heuristics import map_columns


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


def _mappings(table: TableSchema) -> dict[str, Any]:
    schema = DatabaseSchema(tables=[table])
    return map_columns(schema)[table.name]


# ── Task 1: math core ────────────────────────────────────────────────


class TestEmpiricalMarginal:
    def test_inverse_round_trip_preserves_quantiles(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(50, 10, size=500).tolist()
        marg = _empirical_marginal(sample)
        # Uniforms 0..1 -> values should be monotonic and within sample range.
        u = np.linspace(0.01, 0.99, 50)
        vals = np.array([_inverse_marginal(marg, x) for x in u])
        assert np.all(np.diff(vals) >= -1e-9)
        assert vals.min() >= min(sample) - 1e-6
        assert vals.max() <= max(sample) + 1e-6

    def test_inverse_recovers_distribution(self) -> None:
        rng = np.random.default_rng(1)
        sample = rng.normal(0, 1, size=1000).tolist()
        marg = _empirical_marginal(sample)
        u = rng.uniform(size=2000)
        drawn = np.array([_inverse_marginal(marg, x) for x in u])
        # KS test: reconstructed sample matches source distribution.
        _, p = stats.ks_2samp(drawn, sample)
        assert p > 0.05


class TestNormalScores:
    def test_normal_scores_are_standard_normal(self) -> None:
        rng = np.random.default_rng(2)
        sample = rng.exponential(3.0, size=800)
        scores = _to_normal_scores(sample)
        assert abs(float(np.mean(scores))) < 0.15
        assert 0.7 < float(np.std(scores)) < 1.3


class TestNearestPSD:
    def test_repairs_non_psd_matrix(self) -> None:
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        fixed = _nearest_psd(bad)
        eigvals = np.linalg.eigvalsh(fixed)
        assert float(eigvals.min()) >= -1e-8

    def test_psd_matrix_unchanged_within_tolerance(self) -> None:
        good = np.eye(3)
        fixed = _nearest_psd(good)
        assert np.allclose(fixed, good, atol=1e-6)


class TestFitCopula:
    def test_recovers_known_correlation(self) -> None:
        rng = np.random.default_rng(3)
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        data = rng.multivariate_normal([0, 0], cov, size=2000)
        corr = _fit_copula(data)
        assert corr.shape == (2, 2)
        assert abs(corr[0, 1] - 0.8) < 0.15


class TestSampleCategorical:
    def test_reproduces_input_frequencies(self) -> None:
        sample = ["a"] * 700 + ["b"] * 200 + ["c"] * 100
        rng = np.random.default_rng(4)
        draws = _sample_categorical(sample, 5000, rng)
        counts = {v: draws.count(v) for v in ("a", "b", "c")}
        assert abs(counts["a"] / 5000 - 0.7) < 0.05
        assert abs(counts["b"] / 5000 - 0.2) < 0.05
        assert abs(counts["c"] / 5000 - 0.1) < 0.05

    def test_deterministic_with_same_seed(self) -> None:
        sample = ["x", "y", "z"] * 50
        d1 = _sample_categorical(sample, 100, np.random.default_rng(7))
        d2 = _sample_categorical(sample, 100, np.random.default_rng(7))
        assert d1 == d2


# ── Task 2: generate_table happy path ────────────────────────────────


def _stats_table() -> TableSchema:
    return TableSchema(
        name="metrics",
        columns=[
            _col("id", pk=True, autoincrement=True),
            _col("score", data_type=ColumnType.FLOAT, nullable=False),
            _col("count", data_type=ColumnType.INTEGER, nullable=False),
            _col("category", data_type=ColumnType.VARCHAR),
            _col("active", data_type=ColumnType.BOOLEAN),
            _col("created_at", data_type=ColumnType.DATETIME),
        ],
        primary_key=["id"],
    )


def _reference_rows(n: int = 400) -> list[dict[str, Any]]:
    rng = np.random.default_rng(99)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        s = float(rng.normal(75, 12))
        rows.append(
            {
                "id": None,
                "score": s,
                "count": int(max(0, s * 2 + rng.normal(0, 5))),
                "category": rng.choice(["alpha", "beta", "gamma"], p=[0.6, 0.3, 0.1]),
                "active": bool(rng.random() > 0.3),
                "created_at": base.isoformat(),
            }
        )
    return rows


class TestGenerateTableHappyPath:
    def test_returns_requested_rows_with_all_columns(self) -> None:
        table = _stats_table()
        engine = StatisticalEngine(seed=42)
        rows = engine.generate_table(table, _reference_rows(), _mappings(table), 200)
        assert len(rows) == 200
        for row in rows:
            assert set(row) == {
                "id",
                "score",
                "count",
                "category",
                "active",
                "created_at",
            }
            assert row["id"] is None  # autoincrement PK -> None

    def test_distribution_similar_ks_test(self) -> None:
        table = _stats_table()
        ref = _reference_rows(500)
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 500)
        syn = [r["score"] for r in rows]
        real = [r["score"] for r in ref]
        _, p = stats.ks_2samp(syn, real)
        assert p > 0.05

    def test_preserves_pairwise_correlation(self) -> None:
        table = _stats_table()
        ref = _reference_rows(600)
        engine = StatisticalEngine(seed=2)
        rows = engine.generate_table(table, ref, _mappings(table), 600)
        syn_corr = np.corrcoef([r["score"] for r in rows], [r["count"] for r in rows])[0, 1]
        real_corr = np.corrcoef([r["score"] for r in ref], [r["count"] for r in ref])[0, 1]
        assert abs(syn_corr - real_corr) < 0.15

    def test_handles_categorical_and_boolean(self) -> None:
        table = _stats_table()
        engine = StatisticalEngine(seed=3)
        rows = engine.generate_table(table, _reference_rows(), _mappings(table), 300)
        cats = {r["category"] for r in rows}
        assert cats.issubset({"alpha", "beta", "gamma"})
        assert all(isinstance(r["active"], bool) for r in rows)

    def test_deterministic_same_seed(self) -> None:
        table = _stats_table()
        ref = _reference_rows()
        r1 = StatisticalEngine(seed=5).generate_table(table, ref, _mappings(table), 50)
        r2 = StatisticalEngine(seed=5).generate_table(table, ref, _mappings(table), 50)
        assert r1 == r2

    def test_fk_columns_set_to_none(self) -> None:
        table = TableSchema(
            name="orders",
            columns=[
                _col("id", pk=True, autoincrement=True),
                _col("user_id", data_type=ColumnType.INTEGER),
                _col("amount", data_type=ColumnType.FLOAT, nullable=False),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        ref = [{"id": None, "user_id": None, "amount": float(i)} for i in range(150)]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 100)
        assert all(r["user_id"] is None for r in rows)
        assert all(r["id"] is None for r in rows)
        assert all(isinstance(r["amount"], float) for r in rows)


# ── Task 3: fallback ─────────────────────────────────────────────────


class TestFallback:
    def test_empty_reference_falls_back_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        table = _stats_table()
        engine = StatisticalEngine(seed=1)
        with caplog.at_level(logging.WARNING):
            rows = engine.generate_table(table, [], _mappings(table), 30)
        assert len(rows) == 30
        assert any("insufficient" in m.lower() for m in caplog.messages)
        # Heuristic fallback still populates non-FK columns.
        assert all(r["score"] is not None for r in rows)

    def test_too_few_rows_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        table = _stats_table()
        engine = StatisticalEngine(seed=1)
        with caplog.at_level(logging.WARNING):
            rows = engine.generate_table(
                table, _reference_rows(MIN_SAMPLE - 1), _mappings(table), 20
            )
        assert len(rows) == 20
        assert any("insufficient" in m.lower() for m in caplog.messages)

    def test_min_sample_constant(self) -> None:
        assert MIN_SAMPLE == 100


# ── Task 6: edge cases / coverage hardening ──────────────────────────


class TestCoerceHelpers:
    def test_coerce_numeric_handles_bad_values(self) -> None:
        assert _coerce_numeric("3.5") == 3.5
        assert _coerce_numeric("not-a-number") is None
        assert _coerce_numeric(None) is None

    def test_coerce_datetime_ordinal_variants(self) -> None:
        dt = datetime(2021, 6, 1, 12, 0, tzinfo=timezone.utc)
        assert _coerce_datetime_ordinal(dt) == dt.timestamp()
        assert _coerce_datetime_ordinal(date(2021, 6, 1)) == float(date(2021, 6, 1).toordinal())
        assert _coerce_datetime_ordinal(time(1, 2, 3)) == 3723.0
        assert _coerce_datetime_ordinal("2021-06-01T00:00:00") is not None
        assert _coerce_datetime_ordinal("not-a-date") is None
        assert _coerce_datetime_ordinal(123) == 123.0
        assert _coerce_datetime_ordinal(object()) is None


class TestInverseMarginalEdges:
    def test_single_value_marginal(self) -> None:
        marg = _empirical_marginal([7.0])
        assert _inverse_marginal(marg, 0.0) == 7.0
        assert _inverse_marginal(marg, 1.0) == 7.0

    def test_empty_marginal_returns_zero(self) -> None:
        marg = _empirical_marginal([])
        assert _inverse_marginal(marg, 0.5) == 0.0


class TestColumnLookup:
    def test_column_found_and_missing(self) -> None:
        table = TableSchema(
            name="t",
            columns=[_col("a", data_type=ColumnType.INTEGER)],
            primary_key=[],
        )
        assert _column(table, "a").name == "a"
        with pytest.raises(KeyError):
            _column(table, "missing")


class TestOrdinalToValue:
    def test_date_round_trip(self) -> None:
        col = _col("d", data_type=ColumnType.DATE)
        ordinal = float(date(2022, 3, 4).toordinal())
        assert _ordinal_to_value(col, ordinal) == date(2022, 3, 4)

    def test_time_round_trip(self) -> None:
        col = _col("t", data_type=ColumnType.TIME)
        result = _ordinal_to_value(col, 3661.0)
        assert isinstance(result, time)
        assert (result.hour, result.minute, result.second) == (1, 1, 1)

    def test_datetime_round_trip(self) -> None:
        col = _col("ts", data_type=ColumnType.DATETIME)
        result = _ordinal_to_value(col, 1_600_000_000.0)
        assert isinstance(result, datetime)


class TestDegenerateColumns:
    def test_constant_numeric_column(self) -> None:
        """A constant numeric column must still produce that constant."""
        table = TableSchema(
            name="c",
            columns=[
                _col("k", data_type=ColumnType.INTEGER, nullable=False),
                _col("flat", data_type=ColumnType.FLOAT, nullable=False),
            ],
            primary_key=[],
        )
        ref = [{"k": i, "flat": 5.0} for i in range(150)]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 50)
        assert all(abs(r["flat"] - 5.0) < 1e-6 for r in rows)

    def test_all_null_column_emits_none(self) -> None:
        table = TableSchema(
            name="n",
            columns=[
                _col("v", data_type=ColumnType.INTEGER, nullable=False),
                _col("empty", data_type=ColumnType.VARCHAR),
            ],
            primary_key=[],
        )
        ref = [{"v": i, "empty": None} for i in range(150)]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 30)
        assert all(r["empty"] is None for r in rows)

    def test_single_numeric_column_no_correlation(self) -> None:
        table = TableSchema(
            name="s",
            columns=[_col("x", data_type=ColumnType.FLOAT, nullable=False)],
            primary_key=[],
        )
        rng = np.random.default_rng(0)
        ref = [{"x": float(rng.normal(0, 1))} for _ in range(200)]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 100)
        assert len(rows) == 100
        assert all(isinstance(r["x"], float) for r in rows)


class TestNearestPSDDegenerate:
    def test_single_element_matrix(self) -> None:
        scores = np.arange(150, dtype=float).reshape(-1, 1)
        corr = _fit_copula(scores)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0


class TestNoNumericBlock:
    def test_all_categorical_table(self) -> None:
        """A table with no numeric/datetime columns skips the copula."""
        table = TableSchema(
            name="cats",
            columns=[
                _col("tier", data_type=ColumnType.VARCHAR),
                _col("flag", data_type=ColumnType.BOOLEAN),
            ],
            primary_key=[],
        )
        rng = np.random.default_rng(0)
        ref = [
            {
                "tier": rng.choice(["a", "b"]),
                "flag": bool(rng.random() > 0.5),
            }
            for _ in range(150)
        ]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 40)
        assert len(rows) == 40
        assert all(r["tier"] in ("a", "b") for r in rows)
        assert all(isinstance(r["flag"], bool) for r in rows)

    def test_numeric_column_mostly_null_is_skipped(self) -> None:
        """Numeric column with < MIN_SAMPLE non-null values is dropped."""
        table = TableSchema(
            name="sparse",
            columns=[
                _col("label", data_type=ColumnType.VARCHAR, nullable=False),
                _col("rare", data_type=ColumnType.FLOAT),
            ],
            primary_key=[],
        )
        ref = [{"label": f"L{i % 3}", "rare": (1.0 if i < 10 else None)} for i in range(150)]
        engine = StatisticalEngine(seed=1)
        rows = engine.generate_table(table, ref, _mappings(table), 25)
        assert len(rows) == 25
        assert all(r["label"].startswith("L") for r in rows)
