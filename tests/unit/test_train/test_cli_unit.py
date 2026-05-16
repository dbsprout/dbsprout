"""Unit tests for `dbsprout train extract` CLI."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.cli.commands import train as train_cmd

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
    assert "privacy tier 'local'" in result.stdout
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


def test_extract_scrubs_password_from_error_output(tmp_path: Path) -> None:
    """An exception bubbling out of extract() must not leak the DSN password."""
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    dsn = "postgres://user:secret123@host/db"
    fake_extractor = MagicMock()
    # Simulate a connection failure whose message embeds the raw DSN + password
    # (typical of SQLAlchemy / DBAPI error chains).
    fake_extractor.extract.side_effect = RuntimeError(
        f"connection failed for {dsn} (password=secret123)"
    )
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
                dsn,
                "--sample-rows",
                "5",
                "--output",
                str(tmp_path),
                "--quiet",
            ],
        )
    assert result.exit_code == 1
    assert "secret123" not in result.stdout
    assert "secret123" not in (result.stderr or "")
    assert "Error" in result.stdout


# --- S-097: DP (epsilon, delta) line in the pipeline summary ---------------


def _capture_summary(adapter: object) -> str:
    """Render ``_print_summary`` into a string via a patched Rich console."""
    buf = StringIO()
    test_console = Console(file=buf, width=120, no_color=True)
    with patch.object(train_cmd, "console", test_console):
        train_cmd._print_summary(
            samples=3,
            serialize_result=SimpleNamespace(total_rows=3),
            adapter=adapter,
            export_result=SimpleNamespace(gguf_path="model.gguf", size_bytes=1024),
        )
    return buf.getvalue()


def test_print_summary_includes_dp_line_when_epsilon_set() -> None:
    out = _capture_summary(
        SimpleNamespace(
            duration_seconds=1.0,
            adapter_path="adapters/default",
            achieved_epsilon=7.5,
            dp_delta=1e-5,
        )
    )
    assert "DP guarantee" in out
    assert "7.5" in out


def test_print_summary_omits_dp_line_without_epsilon() -> None:
    out = _capture_summary(
        SimpleNamespace(
            duration_seconds=1.0,
            adapter_path="adapters/default",
            achieved_epsilon=None,
            dp_delta=None,
        )
    )
    assert "DP guarantee" not in out
