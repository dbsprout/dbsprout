"""End-to-end CliRunner tests for `dbsprout train serialize`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import polars as pl
from typer.testing import CliRunner

from dbsprout.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _local_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    return cfg


def _seed_samples(input_dir: Path) -> None:
    samples = input_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": [1, 2], "name": ["Ann", "Bob"]}).write_parquet(samples / "users.parquet")
    pl.DataFrame({"id": [9], "user_id": [1]}).write_parquet(samples / "orders.parquet")


def test_cli_serialize_happy_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "run"
    _seed_samples(input_dir)
    out = tmp_path / "data.jsonl"
    with patch("dbsprout.cli.commands.train.load_config", return_value=_local_cfg()):
        result = runner.invoke(
            app,
            [
                "train",
                "serialize",
                "--input",
                str(input_dir),
                "--output",
                str(out),
                "--seed",
                "7",
            ],
        )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"text", "table"}
    assert "Serialized" in result.stdout


def test_cli_serialize_quiet_suppresses_progress_keeps_summary(tmp_path: Path) -> None:
    input_dir = tmp_path / "run"
    _seed_samples(input_dir)
    out = tmp_path / "data.jsonl"
    with patch("dbsprout.cli.commands.train.load_config", return_value=_local_cfg()):
        result = runner.invoke(
            app,
            [
                "train",
                "serialize",
                "--input",
                str(input_dir),
                "--output",
                str(out),
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "Serialized" in result.stdout


def test_cli_serialize_literal_null_policy(tmp_path: Path) -> None:
    input_dir = tmp_path / "run"
    samples = input_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": [1], "name": [None]}).write_parquet(samples / "users.parquet")
    out = tmp_path / "data.jsonl"
    with patch("dbsprout.cli.commands.train.load_config", return_value=_local_cfg()):
        result = runner.invoke(
            app,
            [
                "train",
                "serialize",
                "--input",
                str(input_dir),
                "--output",
                str(out),
                "--null-policy",
                "literal",
            ],
        )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "name is NULL" in obj["text"]


def test_cli_serialize_wrong_privacy_tier_exits_2_no_io(tmp_path: Path) -> None:
    input_dir = tmp_path / "run"
    _seed_samples(input_dir)
    cfg = MagicMock()
    cfg.privacy.tier = "cloud"
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch(
            "dbsprout.train.serializer.DataPreparer.prepare",
            side_effect=AssertionError("must not run"),
        ),
    ):
        result = runner.invoke(
            app,
            ["train", "serialize", "--input", str(input_dir), "--output", str(tmp_path / "x")],
        )
    assert result.exit_code == 2
    assert "requires privacy tier 'local'" in result.stdout
    assert "current: cloud" in result.stdout


def test_cli_serialize_missing_input_exits_1(tmp_path: Path) -> None:
    with patch("dbsprout.cli.commands.train.load_config", return_value=_local_cfg()):
        result = runner.invoke(
            app,
            [
                "train",
                "serialize",
                "--input",
                str(tmp_path / "nope"),
                "--output",
                str(tmp_path / "data.jsonl"),
            ],
        )
    assert result.exit_code == 1
    assert "no Parquet sample files found" in result.stdout
