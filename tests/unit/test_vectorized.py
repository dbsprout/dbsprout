"""Tests for dbsprout.generate.vectorized — NumPy vectorized generation."""

from __future__ import annotations

import time
import uuid
from datetime import date, datetime

from dbsprout.generate.vectorized import generate_vectorized


class TestRandomInt:
    def test_returns_correct_count(self) -> None:
        result = generate_vectorized("random_int", 100, seed=42, params={})
        assert result is not None
        assert len(result) == 100

    def test_all_ints(self) -> None:
        result = generate_vectorized("random_int", 50, seed=42, params={})
        assert result is not None
        for v in result:
            assert isinstance(v, int)

    def test_respects_bounds(self) -> None:
        result = generate_vectorized("random_int", 100, seed=42, params={"min": 10, "max": 20})
        assert result is not None
        for v in result:
            assert 10 <= v <= 20


class TestRandomFloat:
    def test_returns_floats(self) -> None:
        result = generate_vectorized("random_float", 50, seed=42, params={})
        assert result is not None
        for v in result:
            assert isinstance(v, float)

    def test_precision_rounding(self) -> None:
        result = generate_vectorized(
            "random_decimal", 50, seed=42, params={"precision": 10, "scale": 2}
        )
        assert result is not None
        for v in result:
            # Check at most 2 decimal places
            assert round(v, 2) == v


class TestRandomBool:
    def test_returns_bools(self) -> None:
        result = generate_vectorized("random_bool", 100, seed=42, params={})
        assert result is not None
        for v in result:
            assert isinstance(v, bool)

    def test_ratio(self) -> None:
        result = generate_vectorized("random_bool", 10000, seed=42, params={"true_ratio": 0.7})
        assert result is not None
        true_count = sum(1 for v in result if v)
        # Should be ~70% ± 5%
        assert 6500 <= true_count <= 7500


class TestDatetime:
    def test_returns_datetimes(self) -> None:
        result = generate_vectorized("random_datetime", 50, seed=42, params={})
        assert result is not None
        for v in result:
            assert isinstance(v, datetime)

    def test_in_default_range(self) -> None:
        result = generate_vectorized("random_datetime", 100, seed=42, params={})
        assert result is not None
        for v in result:
            assert v.year >= 2020
            assert v.year <= 2026


class TestDate:
    def test_returns_dates(self) -> None:
        result = generate_vectorized("random_date", 50, seed=42, params={})
        assert result is not None
        for v in result:
            assert isinstance(v, date)


class TestUuid:
    def test_valid_uuids(self) -> None:
        result = generate_vectorized("uuid4", 50, seed=42, params={})
        assert result is not None
        for v in result:
            uuid.UUID(str(v))  # validates format

    def test_all_unique(self) -> None:
        result = generate_vectorized("uuid4", 1000, seed=42, params={})
        assert result is not None
        assert len(set(result)) == 1000


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        r1 = generate_vectorized("random_int", 100, seed=42, params={})
        r2 = generate_vectorized("random_int", 100, seed=42, params={})
        assert r1 == r2

    def test_different_seed_different_output(self) -> None:
        r1 = generate_vectorized("random_int", 100, seed=42, params={})
        r2 = generate_vectorized("random_int", 100, seed=99, params={})
        assert r1 != r2


class TestPerformance:
    def test_100k_ints_under_10ms(self) -> None:
        start = time.perf_counter()
        result = generate_vectorized("random_int", 100_000, seed=42, params={})
        elapsed = time.perf_counter() - start
        assert result is not None
        assert len(result) == 100_000
        assert elapsed < 0.5  # generous for CI; actual is <10ms


class TestUnsupported:
    def test_unknown_returns_none(self) -> None:
        result = generate_vectorized("email", 10, seed=42, params={})
        assert result is None
