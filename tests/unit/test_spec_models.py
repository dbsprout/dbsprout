"""Tests for dbsprout.spec.models — DataSpec Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dbsprout.spec.models import (
    CorrelationRule,
    DataSpec,
    DerivedColumn,
    GeneratorConfig,
    TableSpec,
)

# ── GeneratorConfig tests ───────────────────────────────────────────


class TestGeneratorConfig:
    def test_defaults(self) -> None:
        """Default values are correct."""
        cfg = GeneratorConfig(provider="mimesis.Person.email")
        assert cfg.provider == "mimesis.Person.email"
        assert cfg.method is None
        assert cfg.params == {}
        assert cfg.distribution is None
        assert cfg.distribution_params == {}
        assert cfg.min_value is None
        assert cfg.max_value is None
        assert cfg.enum_values is None
        assert cfg.format_pattern is None
        assert cfg.unique is False
        assert cfg.nullable_rate == 0.0
        assert cfg.vectorized is False

    def test_frozen(self) -> None:
        """Mutation must raise error."""
        cfg = GeneratorConfig(provider="test")
        with pytest.raises(ValidationError):
            cfg.provider = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Unknown fields must be rejected."""
        with pytest.raises(ValidationError, match="extra_forbidden"):
            GeneratorConfig(provider="test", unknown_field="bad")  # type: ignore[call-arg]

    def test_full_config(self) -> None:
        """All optional fields populated."""
        cfg = GeneratorConfig(
            provider="numpy.integers",
            method="randint",
            params={"low": 0, "high": 100},
            distribution="normal",
            distribution_params={"mean": 50.0, "std": 10.0},
            min_value=0.0,
            max_value=100.0,
            enum_values=["a", "b", "c"],
            format_pattern=r"\d{3}-\d{4}",
            unique=True,
            nullable_rate=0.1,
            vectorized=True,
        )
        assert cfg.distribution == "normal"
        assert cfg.nullable_rate == 0.1
        assert cfg.unique is True


# ── DerivedColumn tests ─────────────────────────────────────────────


class TestDerivedColumn:
    def test_creation(self) -> None:
        """Expression + depends_on fields work."""
        dc = DerivedColumn(
            column="full_name",
            expression="row['first_name'] + ' ' + row['last_name']",
            depends_on=["first_name", "last_name"],
        )
        assert dc.column == "full_name"
        assert "first_name" in dc.expression
        assert dc.depends_on == ["first_name", "last_name"]

    def test_frozen(self) -> None:
        dc = DerivedColumn(column="x", expression="1", depends_on=[])
        with pytest.raises(ValidationError):
            dc.column = "y"  # type: ignore[misc]


# ── CorrelationRule tests ───────────────────────────────────────────


class TestCorrelationRule:
    def test_defaults(self) -> None:
        """Strategy defaults to 'lookup'."""
        cr = CorrelationRule(columns=["city", "state", "zip_code"])
        assert cr.strategy == "lookup"
        assert cr.lookup_table is None

    def test_with_lookup_table(self) -> None:
        cr = CorrelationRule(
            columns=["city", "state"],
            lookup_table="geo_us",
            strategy="lookup",
        )
        assert cr.lookup_table == "geo_us"


# ── TableSpec tests ─────────────────────────────────────────────────


class TestTableSpec:
    def test_with_columns(self) -> None:
        """Dict of GeneratorConfig keyed by column name."""
        ts = TableSpec(
            table_name="users",
            columns={
                "email": GeneratorConfig(provider="mimesis.Person.email"),
                "age": GeneratorConfig(
                    provider="numpy.integers",
                    min_value=18.0,
                    max_value=80.0,
                ),
            },
        )
        assert ts.table_name == "users"
        assert len(ts.columns) == 2
        assert ts.columns["email"].provider == "mimesis.Person.email"
        assert ts.row_count == 100  # default

    def test_with_derived_and_correlations(self) -> None:
        ts = TableSpec(
            table_name="orders",
            columns={"qty": GeneratorConfig(provider="numpy.integers")},
            derived=[
                DerivedColumn(
                    column="total",
                    expression="row['qty'] * row['price']",
                    depends_on=["qty", "price"],
                ),
            ],
            correlations=[
                CorrelationRule(columns=["city", "state"]),
            ],
        )
        assert len(ts.derived) == 1
        assert len(ts.correlations) == 1


# ── DataSpec tests ──────────────────────────────────────────────────


class TestDataSpec:
    def test_round_trip_json(self) -> None:
        """model → JSON → model must produce identical result."""
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="users",
                    row_count=50,
                    columns={
                        "email": GeneratorConfig(provider="mimesis.Person.email"),
                        "age": GeneratorConfig(
                            provider="numpy.integers",
                            distribution="uniform",
                            min_value=18.0,
                            max_value=80.0,
                            vectorized=True,
                        ),
                    },
                    derived=[
                        DerivedColumn(
                            column="display",
                            expression="row['email'].split('@')[0]",
                            depends_on=["email"],
                        ),
                    ],
                ),
            ],
            global_seed=99,
            schema_hash="abc123def456",
            model_used="qwen2.5-1.5b",
            created_at="2026-03-30T12:00:00Z",
        )

        json_str = spec.model_dump_json()
        restored = DataSpec.model_validate_json(json_str)
        assert restored == spec
        assert restored.schema_hash == "abc123def456"
        assert restored.model_used == "qwen2.5-1.5b"

    def test_get_table_spec_found(self) -> None:
        """Lookup by name returns matching TableSpec."""
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="users",
                    columns={"id": GeneratorConfig(provider="builtin.autoincrement")},
                ),
                TableSpec(
                    table_name="orders",
                    columns={"id": GeneratorConfig(provider="builtin.autoincrement")},
                ),
            ],
        )
        result = spec.get_table_spec("orders")
        assert result is not None
        assert result.table_name == "orders"

    def test_get_table_spec_not_found(self) -> None:
        """Lookup for missing table returns None."""
        spec = DataSpec(tables=[])
        assert spec.get_table_spec("nonexistent") is None

    def test_json_schema_valid(self) -> None:
        """model_json_schema() returns a valid dict with expected structure."""
        schema = DataSpec.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "tables" in schema["properties"]

    def test_with_all_fields(self) -> None:
        """All optional fields populated."""
        spec = DataSpec(
            version="2.0",
            tables=[
                TableSpec(
                    table_name="t",
                    row_count=10,
                    columns={"c": GeneratorConfig(provider="test")},
                ),
            ],
            global_seed=123,
            schema_hash="deadbeef",
            model_used="gpt-4o",
            created_at="2026-01-01T00:00:00Z",
        )
        assert spec.version == "2.0"
        assert spec.global_seed == 123
        assert spec.created_at == "2026-01-01T00:00:00Z"


# ── Boundary validation tests ───────────────────────────────────────


class TestBoundaryValidation:
    def test_nullable_rate_too_high(self) -> None:
        """nullable_rate > 1.0 must be rejected."""
        with pytest.raises(ValidationError):
            GeneratorConfig(provider="test", nullable_rate=1.5)

    def test_nullable_rate_negative(self) -> None:
        """nullable_rate < 0.0 must be rejected."""
        with pytest.raises(ValidationError):
            GeneratorConfig(provider="test", nullable_rate=-0.1)

    def test_row_count_zero(self) -> None:
        """row_count=0 must be rejected."""
        with pytest.raises(ValidationError):
            TableSpec(
                table_name="t",
                row_count=0,
                columns={"c": GeneratorConfig(provider="test")},
            )

    def test_global_seed_negative(self) -> None:
        """Negative global_seed must be rejected."""
        with pytest.raises(ValidationError):
            DataSpec(tables=[], global_seed=-1)


class TestTableSpecCardinality:
    def test_cardinality_default_none(self) -> None:
        """Cardinality defaults to None."""
        ts = TableSpec(
            table_name="t",
            columns={"c": GeneratorConfig(provider="test")},
        )
        assert ts.cardinality is None

    def test_cardinality_with_value(self) -> None:
        """Cardinality can be set."""
        ts = TableSpec(
            table_name="t",
            columns={"c": GeneratorConfig(provider="test")},
            cardinality={"fk_ratio": 0.5},
        )
        assert ts.cardinality == {"fk_ratio": 0.5}


# ── Field-description tests (S-123) ─────────────────────────────────


class TestGeneratorConfigDescriptions:
    """S-123: every ``GeneratorConfig`` field carries a user-facing description.

    Studio surfaces these as field tooltips (``title=`` / ``aria-describedby``)
    so users can understand a setting without reading source. Pydantic v2's
    ``Field(description=…)`` is the single source of truth — no parallel
    hand-maintained list.
    """

    def test_every_field_has_description(self) -> None:
        """All public fields advertise a non-empty ``description``."""
        for name, info in GeneratorConfig.model_fields.items():
            assert info.description, f"GeneratorConfig.{name} missing description"
            assert isinstance(info.description, str)
            assert info.description.strip(), f"GeneratorConfig.{name} description is blank-ish"

    def test_field_descriptions_helper_returns_map(self) -> None:
        """``field_descriptions()`` exposes the descriptions as ``dict[str,str]``."""
        from dbsprout.spec.models import field_descriptions  # noqa: PLC0415

        mapping = field_descriptions()
        assert isinstance(mapping, dict)
        # Every model field must appear in the map with a non-empty string.
        for name in GeneratorConfig.model_fields:
            assert name in mapping, f"missing description for {name}"
            assert isinstance(mapping[name], str)
            assert mapping[name].strip(), f"blank description for {name}"

    def test_field_descriptions_is_immutable_copy(self) -> None:
        """Callers cannot mutate the shared module-level mapping."""
        from dbsprout.spec.models import field_descriptions  # noqa: PLC0415

        first = field_descriptions()
        first["provider"] = "TAMPERED"
        # Second call must not show the tampered value.
        second = field_descriptions()
        assert second["provider"] != "TAMPERED"
