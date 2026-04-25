"""End-to-end CliRunner test for `dbsprout train extract`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_cli_extract_against_sqlite_fixture(sqlite_db: str, tmp_path: Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    with patch("dbsprout.cli.commands.train.load_config", return_value=cfg):
        result = runner.invoke(
            app,
            [
                "train",
                "extract",
                "--db",
                sqlite_db,
                "--sample-rows",
                "30",
                "--output",
                str(tmp_path / "run"),
                "--seed",
                "5",
                "--max-per-table",
                "20",
            ],
        )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert (tmp_path / "run" / "manifest.json").exists()
    assert (tmp_path / "run" / "samples" / "users.parquet").exists()
    assert "Extracted" in result.stdout
