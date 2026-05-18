"""CLI tests for `dbsprout generate --engine statistical` (S-071)."""

from __future__ import annotations

import csv
import re
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from dbsprout.cli.app import app
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


def _write_project(tmp_path: Path) -> Path:
    schema = DatabaseSchema(
        tables=[
            TableSchema(
                name="metrics",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(
                        name="score",
                        data_type=ColumnType.FLOAT,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
    )
    snap_dir = tmp_path / ".dbsprout"
    snap_dir.mkdir()
    (snap_dir / "schema.json").write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    return tmp_path


def _write_reference_csv(tmp_path: Path) -> Path:
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(0)
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score"])
        for _ in range(200):
            writer.writerow([round(float(rng.normal(70, 12)), 4)])
    return path


class TestStatisticalCliFlag:
    def test_reference_data_flag_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--reference-data" in _strip_ansi(result.output)

    def test_statistical_with_reference_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # generate_command auto-saves a cwd-relative snapshot; isolate it.
        monkeypatch.chdir(tmp_path)
        project = _write_project(tmp_path)
        ref = _write_reference_csv(tmp_path)
        seeds = project / "seeds"
        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds),
                "--engine",
                "statistical",
                "--reference-data",
                str(ref),
                "--rows",
                "50",
            ],
        )
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert list(seeds.glob("*.sql"))

    def test_statistical_without_reference_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        project = _write_project(tmp_path)
        seeds = project / "seeds"
        result = runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                str(project / ".dbsprout" / "schema.json"),
                "--output-dir",
                str(seeds),
                "--engine",
                "statistical",
                "--rows",
                "10",
            ],
        )
        assert result.exit_code == 0, _strip_ansi(result.output)
        assert list(seeds.glob("*.sql"))
