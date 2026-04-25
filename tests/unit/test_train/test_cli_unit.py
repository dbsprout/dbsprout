"""Unit tests for `dbsprout train extract` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_privacy_gate_blocks_non_local_tier(tmp_path: Path) -> None:
    """AC-7: exit 2 with remediation message; no extractor constructed."""
    cfg = MagicMock()
    cfg.privacy.tier = "cloud"
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor") as fake,
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "extract",
                "--db",
                "postgresql://u:p@h/d",
                "--sample-rows",
                "10",
                "--output",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 2
    assert "privacy tier 'local'" in result.stderr
    fake.assert_not_called()


def test_extract_invokes_sample_extractor(tmp_path: Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    fake_extractor = MagicMock()
    fake_extractor.extract.return_value = MagicMock(tables=(), duration_seconds=0.0)
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor", return_value=fake_extractor),
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "extract",
                "--db",
                "sqlite:///:memory:",
                "--sample-rows",
                "5",
                "--output",
                str(tmp_path),
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.stderr
    fake_extractor.extract.assert_called_once()
