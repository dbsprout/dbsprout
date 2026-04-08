"""Tests for --db envvar support (DBSPROUT_TARGET_DB)."""

from __future__ import annotations

import os
from unittest.mock import patch

from typer.testing import CliRunner

from dbsprout.cli.app import app

runner = CliRunner()


def test_generate_db_reads_envvar() -> None:
    """AC: --db flag reads from DBSPROUT_TARGET_DB env var as fallback."""
    env = os.environ.copy()
    env["DBSPROUT_TARGET_DB"] = "postgresql://envuser@localhost/envdb"

    with patch("dbsprout.cli.commands.generate.generate_command") as mock_gen:
        result = runner.invoke(
            app,
            ["generate", "--schema-snapshot", "nonexistent.json", "--output-format", "direct"],
            env=env,
        )
        assert mock_gen.called, f"generate_command not called; output: {result.output}"
        _, kwargs = mock_gen.call_args
        assert kwargs["target_db"] == "postgresql://envuser@localhost/envdb"


def test_generate_db_cli_overrides_envvar() -> None:
    """CLI --db flag takes precedence over DBSPROUT_TARGET_DB env var."""
    env = os.environ.copy()
    env["DBSPROUT_TARGET_DB"] = "postgresql://envuser@localhost/envdb"

    with patch("dbsprout.cli.commands.generate.generate_command") as mock_gen:
        runner.invoke(
            app,
            [
                "generate",
                "--schema-snapshot",
                "nonexistent.json",
                "--output-format",
                "direct",
                "--db",
                "postgresql://cliuser@localhost/clidb",
            ],
            env=env,
        )
        assert mock_gen.called
        _, kwargs = mock_gen.call_args
        assert kwargs["target_db"] == "postgresql://cliuser@localhost/clidb"


def test_init_db_reads_envvar() -> None:
    """init --db also reads from DBSPROUT_TARGET_DB env var."""
    env = os.environ.copy()
    env["DBSPROUT_TARGET_DB"] = "sqlite:///from_env.db"

    with patch("dbsprout.cli.commands.init.init_command") as mock_init:
        result = runner.invoke(app, ["init"], env=env)
        assert mock_init.called, f"init_command not called; output: {result.output}"
        _, kwargs = mock_init.call_args
        assert kwargs["db"] == "sqlite:///from_env.db"
