"""Tests for dbsprout generate --incremental CLI flag."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.migrate.snapshot import SnapshotStore
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _minimal_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="items",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="name",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            )
        ],
    )


def _write_schema_sql(tmp_path: Path, with_created_at: bool = False) -> Path:
    f = tmp_path / "schema.sql"
    if with_created_at:
        f.write_text(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
            "created_at VARCHAR);",
            encoding="utf-8",
        )
    else:
        f.write_text(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL);",
            encoding="utf-8",
        )
    return f


class TestIncrementalFallback:
    def test_no_prior_snapshot_falls_back_to_full_gen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--incremental with no base snapshot logs warning and runs full gen."""
        schema_path = _write_schema_sql(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--incremental",
                "--file",
                str(schema_path),
                "--output-dir",
                str(tmp_path / "seeds"),
            ],
        )
        assert result.exit_code == 0
        out = _strip_ansi(result.output).lower()
        assert "falling back to full generation" in out


class TestIncrementalNoChanges:
    def test_identical_schema_returns_existing_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old snapshot equal to new schema → no-op passthrough, snapshot saved."""

        monkeypatch.chdir(tmp_path)
        store = SnapshotStore()
        store.save(_minimal_schema())

        schema_path = _write_schema_sql(tmp_path)

        result = runner.invoke(
            app,
            [
                "generate",
                "--incremental",
                "--file",
                str(schema_path),
                "--output-dir",
                str(tmp_path / "seeds"),
                "--output-format",
                "sql",
            ],
        )
        assert result.exit_code == 0
        out = _strip_ansi(result.output).lower()
        assert "no schema changes" in out or "no changes detected" in out
        assert any((tmp_path / "seeds").glob("*items.sql"))


class TestIncrementalColumnAdded:
    def test_column_added_triggers_updater(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NEW schema adds a column → IncrementalUpdater populates it, output present."""

        monkeypatch.chdir(tmp_path)
        store = SnapshotStore()
        store.save(_minimal_schema())

        schema_path = _write_schema_sql(tmp_path, with_created_at=True)

        result = runner.invoke(
            app,
            [
                "generate",
                "--incremental",
                "--file",
                str(schema_path),
                "--output-dir",
                str(tmp_path / "seeds"),
                "--output-format",
                "json",
                "--rows",
                "5",
            ],
        )
        assert result.exit_code == 0
        json_files = list((tmp_path / "seeds").glob("*items.json"))
        assert json_files, f"no json output produced; seeds={list((tmp_path / 'seeds').iterdir())}"
        rows = json_files[0].read_text(encoding="utf-8")
        assert '"created_at"' in rows
        out = _strip_ansi(result.output).lower()
        assert "column_added" in out or "generate_column" in out or "+cols" in out
