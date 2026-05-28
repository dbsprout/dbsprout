"""Unit tests for the core service facade's validate + write paths (S-106)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dbsprout.config.models import DBSproutConfig
from dbsprout.core import service
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.quality.integrity import validate_integrity
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

_SEED = 42
_ROWS = 15


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="widgets",
                columns=[
                    ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
                    ColumnSchema(name="label", data_type=ColumnType.VARCHAR, nullable=False),
                ],
                primary_key=["id"],
            )
        ]
    )


def test_run_validation_integrity_matches_direct_pipeline() -> None:
    """Facade integrity report equals a direct orchestrate + validate call."""
    schema = _schema()
    config = DBSproutConfig.from_toml(None)

    outcome = service.run_validation(
        schema, config, seed=_SEED, default_rows=_ROWS, engine="heuristic"
    )

    direct = orchestrate(schema, config, seed=_SEED, default_rows=_ROWS, engine="heuristic")
    expected = validate_integrity(direct.tables_data, schema)

    assert outcome.integrity.passed == expected.passed
    assert len(outcome.integrity.checks) == len(expected.checks)
    assert outcome.fidelity is None
    assert outcome.detection is None
    assert set(outcome.tables_data) == {"widgets"}


def test_run_validation_without_reference_skips_fidelity_and_detection() -> None:
    """No reference data → fidelity/detection stay None even if detection=True."""
    schema = _schema()
    config = DBSproutConfig.from_toml(None)

    outcome = service.run_validation(
        schema,
        config,
        seed=_SEED,
        default_rows=_ROWS,
        engine="heuristic",
        reference_data=None,
        detection=True,
    )

    assert outcome.fidelity is None
    assert outcome.detection is None


def test_run_validation_with_reference_computes_fidelity() -> None:
    """Reference rows present → fidelity report is populated (needs [stats])."""
    pytest.importorskip("scipy", reason="fidelity needs the [stats] extra")
    schema = _schema()
    config = DBSproutConfig.from_toml(None)
    reference = {"widgets": [{"id": i, "label": f"ref-{i}"} for i in range(_ROWS)]}

    outcome = service.run_validation(
        schema,
        config,
        seed=_SEED,
        default_rows=_ROWS,
        engine="heuristic",
        reference_data=reference,
    )

    assert outcome.fidelity is not None


def test_run_validation_with_reference_and_detection_computes_both() -> None:
    """detection=True with reference rows → detection report is populated."""
    pytest.importorskip("sklearn", reason="detection needs the [stats] extra")
    pytest.importorskip("scipy", reason="fidelity needs the [stats] extra")
    schema = _schema()
    config = DBSproutConfig.from_toml(None)
    reference = {"widgets": [{"id": i, "label": f"ref-{i}"} for i in range(_ROWS)]}

    outcome = service.run_validation(
        schema,
        config,
        seed=_SEED,
        default_rows=_ROWS,
        engine="heuristic",
        reference_data=reference,
        detection=True,
    )

    assert outcome.fidelity is not None
    assert outcome.detection is not None


def test_write_output_unknown_format_raises(tmp_path: Path) -> None:
    """An unknown format (or the CLI-only ``direct``) is a ValueError."""
    schema = _schema()
    config = DBSproutConfig.from_toml(None)
    result = service.generate(schema, config, seed=_SEED, default_rows=_ROWS)

    with pytest.raises(ValueError, match="Unknown or unsupported output format"):
        service.write_output(
            result,
            schema,
            result.insertion_order,
            tmp_path,
            output_format="direct",
        )


def test_write_output_upsert_warns_for_non_sql_format(tmp_path: Path) -> None:
    """--upsert is ignored for csv; the warning is returned, not printed."""
    schema = _schema()
    config = DBSproutConfig.from_toml(None)
    result = service.generate(schema, config, seed=_SEED, default_rows=_ROWS)

    outcome = service.write_output(
        result,
        schema,
        result.insertion_order,
        tmp_path,
        output_format="csv",
        upsert=True,
    )

    assert len(outcome.warnings) == 1
    assert "ignored for 'csv'" in outcome.warnings[0]


def test_write_output_sql_no_warnings(tmp_path: Path) -> None:
    """SQL writes (with or without upsert) produce no warnings."""
    schema = _schema()
    config = DBSproutConfig.from_toml(None)
    result = service.generate(schema, config, seed=_SEED, default_rows=_ROWS)

    outcome = service.write_output(
        result,
        schema,
        result.insertion_order,
        tmp_path,
        output_format="sql",
        upsert=True,
    )

    assert outcome.warnings == ()
    assert list(tmp_path.glob("*.sql"))


def test_load_schema_parses_file(tmp_path: Path) -> None:
    """load_schema parses a schema file source via the parser dispatch."""
    from dbsprout.cli.sources import SchemaSource  # noqa: PLC0415

    ddl = tmp_path / "schema.sql"
    ddl.write_text(
        "CREATE TABLE widgets (id INTEGER PRIMARY KEY, label VARCHAR NOT NULL);",
        encoding="utf-8",
    )
    source = SchemaSource(kind="file", raw_value=str(ddl), display_value=str(ddl))

    schema = service.load_schema(source)

    assert [t.name for t in schema.tables] == ["widgets"]
