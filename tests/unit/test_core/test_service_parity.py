"""Byte-parity tests for the core service facade (S-106).

The #1 acceptance criterion for S-106 is that routing ``dbsprout generate``
through the new ``dbsprout.core.service`` facade produces **byte-identical**
output for a fixed seed. These tests freeze the *current* (pre-refactor)
pipeline output as the golden reference, then assert the facade reproduces it
exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.config.models import DBSproutConfig
from dbsprout.generate.orchestrator import orchestrate
from dbsprout.plugins.dispatch import resolve_writer
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_SEED = 42
_ROWS = 25
_DIALECT = "postgresql"


def _parity_schema() -> DatabaseSchema:
    """A fixed two-table schema with one FK (deterministic across runs)."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(
                name="email",
                data_type=ColumnType.VARCHAR,
                nullable=False,
                unique=True,
            ),
            ColumnSchema(name="full_name", data_type=ColumnType.VARCHAR, nullable=False),
        ],
        primary_key=["id"],
    )
    posts = TableSchema(
        name="posts",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=False),
            ColumnSchema(name="title", data_type=ColumnType.VARCHAR, nullable=False),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(tables=[users, posts])


def _read_sql_bytes(directory: Path) -> dict[str, bytes]:
    """Return a {filename: bytes} map of every ``*.sql`` file in *directory*."""
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.sql"))}


def _golden_sql_bytes(out_dir: Path) -> dict[str, bytes]:
    """Generate + write SQL the *current* (pre-facade) way → golden bytes."""
    schema = _parity_schema()
    config = DBSproutConfig.from_toml(None)
    result = orchestrate(schema, config, seed=_SEED, default_rows=_ROWS, engine="heuristic")
    writer = resolve_writer("sql")
    writer.write(
        result.tables_data,
        schema,
        result.insertion_order,
        out_dir,
        dialect=_DIALECT,
    )
    return _read_sql_bytes(out_dir)


def test_facade_generate_sql_is_byte_identical_to_golden(tmp_path: Path) -> None:
    """``service.generate`` + ``service.write_output`` == current pipeline bytes."""
    from dbsprout.core import service  # noqa: PLC0415

    golden = _golden_sql_bytes(tmp_path / "golden")

    schema = _parity_schema()
    config = DBSproutConfig.from_toml(None)
    facade_dir = tmp_path / "facade"
    result = service.generate(
        schema,
        config,
        seed=_SEED,
        default_rows=_ROWS,
        engine="heuristic",
    )
    service.write_output(
        result,
        schema,
        result.insertion_order,
        facade_dir,
        output_format="sql",
        dialect=_DIALECT,
    )
    facade_bytes = _read_sql_bytes(facade_dir)

    assert facade_bytes == golden, "facade SQL output diverged from the golden pipeline output"
    assert golden, "golden capture produced no SQL files — fixture is broken"


def test_cli_generate_output_unchanged_through_facade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the ``dbsprout generate`` command output is byte-identical."""
    from typer.testing import CliRunner  # noqa: PLC0415

    from dbsprout.cli.app import app  # noqa: PLC0415

    golden = _golden_sql_bytes(tmp_path / "golden")

    # Run the real CLI command against the same fixed schema/seed.
    monkeypatch.chdir(tmp_path)
    snapshot = tmp_path / "schema.json"
    snapshot.write_text(_parity_schema().model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "cli_seeds"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "generate",
            "--schema-snapshot",
            str(snapshot),
            "--rows",
            str(_ROWS),
            "--seed",
            str(_SEED),
            "--output-format",
            "sql",
            "--output-dir",
            str(out_dir),
            "--dialect",
            _DIALECT,
        ],
        env={"COLUMNS": "200", "NO_COLOR": "1"},
    )
    assert res.exit_code == 0, res.output
    cli_bytes = _read_sql_bytes(out_dir)
    assert cli_bytes == golden, "CLI generate output diverged from the golden pipeline output"
