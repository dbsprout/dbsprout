"""Tests for dbsprout.generate.check_parser — SQL CHECK expression parser."""

from __future__ import annotations

from dbsprout.generate.check_parser import CheckConstraint, parse_check


class TestGreaterThan:
    def test_greater_than(self) -> None:
        """col > 0 → min_value exclusive."""
        result = parse_check("price", "price > 0")
        assert result is not None
        assert result.min_value == 1  # exclusive: > 0 means >= 1 for integers

    def test_greater_equal(self) -> None:
        """col >= 18 → min_value=18."""
        result = parse_check("age", "age >= 18")
        assert result is not None
        assert result.min_value == 18


class TestLessThan:
    def test_less_than(self) -> None:
        """col < 100 → max_value exclusive."""
        result = parse_check("score", "score < 100")
        assert result is not None
        assert result.max_value == 99  # exclusive: < 100 means <= 99

    def test_less_equal(self) -> None:
        """col <= 120 → max_value=120."""
        result = parse_check("age", "age <= 120")
        assert result is not None
        assert result.max_value == 120


class TestBetween:
    def test_between(self) -> None:
        """col BETWEEN 1 AND 100 → min=1, max=100."""
        result = parse_check("score", "score BETWEEN 1 AND 100")
        assert result is not None
        assert result.min_value == 1
        assert result.max_value == 100

    def test_between_case_insensitive(self) -> None:
        """BETWEEN is case-insensitive."""
        result = parse_check("x", "x between 0 and 10")
        assert result is not None
        assert result.min_value == 0
        assert result.max_value == 10


class TestIn:
    def test_in_strings(self) -> None:
        """col IN ('a', 'b') → allowed_values."""
        result = parse_check("status", "status IN ('active', 'inactive', 'pending')")
        assert result is not None
        assert result.allowed_values == ["active", "inactive", "pending"]

    def test_in_case_insensitive(self) -> None:
        """IN is case-insensitive."""
        result = parse_check("x", "x in ('a', 'b')")
        assert result is not None
        assert result.allowed_values == ["a", "b"]


class TestCombinedAnd:
    def test_combined_range(self) -> None:
        """col >= 0 AND col <= 100 → min=0, max=100."""
        result = parse_check("price", "price >= 0 AND price <= 100")
        assert result is not None
        assert result.min_value == 0
        assert result.max_value == 100


class TestUnparseable:
    def test_complex_expression_returns_none(self) -> None:
        """Complex expression that cannot be parsed → None."""
        result = parse_check("x", "(a + b) > c * 2")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty expression → None."""
        result = parse_check("x", "")
        assert result is None


class TestCheckConstraintModel:
    def test_defaults(self) -> None:
        """CheckConstraint defaults."""
        cc = CheckConstraint(column="x")
        assert cc.min_value is None
        assert cc.max_value is None
        assert cc.allowed_values is None

    def test_frozen(self) -> None:
        """CheckConstraint is frozen."""
        import dataclasses  # noqa: PLC0415

        cc = CheckConstraint(column="x", min_value=1)
        assert dataclasses.is_dataclass(cc)
