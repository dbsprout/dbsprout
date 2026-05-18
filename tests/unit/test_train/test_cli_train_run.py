"""Unit tests for `dbsprout train run` CLI subcommand."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.train.config import LoRAAdapter, TrainConfig

if TYPE_CHECKING:
    from pathlib import Path as _Path

runner = CliRunner()


def _adapter(out: _Path) -> LoRAAdapter:
    return LoRAAdapter(
        adapter_path=out / "default",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        epochs=3,
        train_samples=1000,
        final_loss=0.12,
        duration_seconds=42.0,
    )


def test_privacy_gate_blocks_non_local_tier(tmp_path: _Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "cloud"
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.trainer.QLoRATrainer") as fake,
    ):
        result = runner.invoke(
            app,
            ["train", "run", "--corpus", str(tmp_path / "data.jsonl")],
        )
    assert result.exit_code == 2
    assert "privacy tier 'local'" in result.stdout
    fake.assert_not_called()


def _local_cfg() -> MagicMock:
    """A config mock whose ``.train`` is a real frozen ``TrainConfig`` so the
    CLI's ``model_copy(update=...)`` plumbing is exercised for real."""
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    cfg.train = TrainConfig()
    return cfg


def test_run_success_prints_summary(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / "adapters"
    trainer_instance = MagicMock()
    trainer_instance.train.return_value = _adapter(out)
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.trainer.QLoRATrainer", return_value=trainer_instance),
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "run",
                "--corpus",
                str(tmp_path / "data.jsonl"),
                "--output",
                str(out),
            ],
        )
    assert result.exit_code == 0, result.stdout
    assert "Trained" in result.stdout
    assert "1000 samples" in result.stdout
    trainer_instance.train.assert_called_once()


def test_run_cli_overrides_config(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / "adapters"
    trainer_instance = MagicMock()
    trainer_instance.train.return_value = _adapter(out)
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.trainer.QLoRATrainer", return_value=trainer_instance),
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "run",
                "--corpus",
                str(tmp_path / "data.jsonl"),
                "--output",
                str(out),
                "--epochs",
                "7",
                "--lora-rank",
                "8",
            ],
        )
    assert result.exit_code == 0, result.stdout
    passed_cfg = trainer_instance.train.call_args.kwargs["config"]
    assert passed_cfg.epochs == 7
    assert passed_cfg.lora_rank == 8


def test_run_handles_runtime_error(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    trainer_instance = MagicMock()
    trainer_instance.train.side_effect = RuntimeError("CUDA GPU not detected")
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.trainer.QLoRATrainer", return_value=trainer_instance),
    ):
        result = runner.invoke(
            app,
            ["train", "run", "--corpus", str(tmp_path / "data.jsonl")],
        )
    assert result.exit_code == 1
    assert "CUDA GPU not detected" in result.stdout


def test_run_handles_missing_corpus(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    trainer_instance = MagicMock()
    trainer_instance.train.side_effect = FileNotFoundError("corpus not found")
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.trainer.QLoRATrainer", return_value=trainer_instance),
    ):
        result = runner.invoke(
            app,
            ["train", "run", "--corpus", str(tmp_path / "missing.jsonl")],
        )
    assert result.exit_code == 1
    assert "corpus not found" in result.stdout
