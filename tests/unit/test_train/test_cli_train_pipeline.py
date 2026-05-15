"""Unit tests for the `dbsprout train` end-to-end pipeline callback (S-068).

The pipeline chains SampleExtractor -> DataPreparer -> select_trainer() ->
Exporter. Every heavy stage is mocked at its import site in
``dbsprout.cli.commands.train`` so no real DB, training, or export runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.train.config import LoRAAdapter, TrainConfig

if TYPE_CHECKING:
    from pathlib import Path as _Path

runner = CliRunner()


def _local_cfg() -> MagicMock:
    """Config mock with a real frozen ``TrainConfig`` so the CLI's
    ``model_copy(update=...)`` precedence plumbing is exercised for real."""
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    cfg.train = TrainConfig()
    return cfg


def _sample_result(out: _Path) -> MagicMock:
    res = MagicMock()
    res.tables = (MagicMock(sampled=300, fk_closure_added=20),)
    res.duration_seconds = 1.5
    res.output_dir = out / "training"
    res.manifest_path = out / "training" / "manifest.json"
    return res


def _serialize_result(out: _Path) -> MagicMock:
    res = MagicMock()
    res.total_rows = 320
    res.tables = (MagicMock(nulls_skipped=4),)
    res.duration_seconds = 0.7
    res.output_path = out / "training" / "data.jsonl"
    return res


def _adapter(out: _Path) -> LoRAAdapter:
    return LoRAAdapter(
        adapter_path=out / "models" / "adapters" / "default",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        epochs=3,
        train_samples=320,
        final_loss=0.21,
        duration_seconds=88.0,
    )


def _export_result(out: _Path) -> MagicMock:
    res = MagicMock()
    res.gguf_path = out / "models" / "custom" / "default-Q4_K_M.gguf"
    res.size_bytes = 950_000_000
    res.quant_type = "Q4_K_M"
    res.source_format = "peft"
    res.base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    return res


def _patches(
    cfg: MagicMock,
    extractor: MagicMock,
    preparer: MagicMock,
    trainer: MagicMock,
    exporter: MagicMock,
):
    return (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor", return_value=extractor),
        patch("dbsprout.train.serializer.DataPreparer", return_value=preparer),
        patch("dbsprout.train.mlx_trainer.select_trainer", return_value=trainer),
        patch("dbsprout.train.exporter.Exporter", return_value=exporter),
    )


# --------------------------------------------------------------------------- #
# Task 1 — privacy gate + subcommand regression guards
# --------------------------------------------------------------------------- #
def test_pipeline_privacy_gate_blocks_non_local_tier(tmp_path: _Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "cloud"
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor") as fake,
    ):
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db"])
    assert result.exit_code == 2
    assert "privacy tier 'local'" in result.stdout
    fake.assert_not_called()


def test_train_no_args_still_shows_help() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert "extract" in result.stdout
    assert "serialize" in result.stdout
    assert "run" in result.stdout


def test_existing_subcommands_still_resolve() -> None:
    for sub in ("extract", "serialize", "run"):
        result = runner.invoke(app, ["train", sub, "--help"])
        assert result.exit_code == 0, result.stdout
        assert sub in result.stdout


def test_pipeline_requires_db_flag() -> None:
    cfg = _local_cfg()
    with patch("dbsprout.cli.commands.train.load_config", return_value=cfg):
        result = runner.invoke(app, ["train"], catch_exceptions=False)
    # No --db and no subcommand -> help (no_args_is_help) so exit 0, no crash.
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# Task 2 — happy path orchestration + summary
# --------------------------------------------------------------------------- #
def test_pipeline_runs_all_stages_in_order(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(
            app,
            ["train", "--db", "sqlite:///x.db", "--output", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    extractor.extract.assert_called_once()
    preparer.prepare.assert_called_once()
    trainer.train.assert_called_once()
    exporter.to_gguf_result.assert_called_once()
    # corpus produced by serialize must flow into the trainer
    corpus = preparer.prepare.return_value.output_path
    assert trainer.train.call_args.kwargs["corpus_path"] == corpus
    # adapter produced by the trainer must flow into the exporter
    adapter = trainer.train.return_value
    assert exporter.to_gguf_result.call_args.args[0] == adapter


def test_pipeline_summary_reports_key_metrics(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    out_text = result.stdout
    assert "320" in out_text  # samples
    assert "88" in out_text  # training seconds
    assert "default-Q4_K_M.gguf" in out_text  # gguf path
    assert ".gguf" in out_text


# --------------------------------------------------------------------------- #
# Task 3 — config precedence (CLI > toml > defaults)
# --------------------------------------------------------------------------- #
def test_cli_epoch_flag_overrides_toml(tmp_path: _Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    cfg.train = TrainConfig(epochs=5)
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(
            app,
            ["train", "--db", "sqlite:///x.db", "--output", str(out), "--epochs", "9"],
        )
    assert result.exit_code == 0, result.stdout
    assert trainer.train.call_args.kwargs["config"].epochs == 9


def test_toml_epoch_used_when_no_cli_flag(tmp_path: _Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    cfg.train = TrainConfig(epochs=5)
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 0, result.stdout
    assert trainer.train.call_args.kwargs["config"].epochs == 5


def test_sample_rows_flag_threaded_to_extractor(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(
            app,
            [
                "train",
                "--db",
                "sqlite:///x.db",
                "--output",
                str(out),
                "--sample-rows",
                "250",
            ],
        )
    assert result.exit_code == 0, result.stdout
    passed_cfg = extractor.extract.call_args.kwargs["config"]
    assert passed_cfg.sample_rows == 250


# --------------------------------------------------------------------------- #
# Task 4 — trainer auto-detect error path
# --------------------------------------------------------------------------- #
def test_no_gpu_backend_exits_with_install_hint(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    exporter = MagicMock()
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor", return_value=extractor),
        patch("dbsprout.train.serializer.DataPreparer", return_value=preparer),
        patch(
            "dbsprout.train.mlx_trainer.select_trainer",
            side_effect=RuntimeError("MLX training requires Apple Silicon"),
        ),
        patch("dbsprout.train.exporter.Exporter", return_value=exporter),
    ):
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 1
    assert "Apple Silicon" in result.stdout
    exporter.to_gguf_result.assert_not_called()


# --------------------------------------------------------------------------- #
# Task 5 — stage failure handling + secret scrubbing
# --------------------------------------------------------------------------- #
def test_extractor_error_scrubs_db_password(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    secret_url = "postgresql://user:supersecret@host:5432/db"  # noqa: S105 - test DSN, not a real credential
    extractor.extract.side_effect = RuntimeError(f"could not connect to server {secret_url}")
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor", return_value=extractor),
    ):
        result = runner.invoke(app, ["train", "--db", secret_url, "--output", str(out)])
    assert result.exit_code == 1
    assert "supersecret" not in result.stdout


def test_serializer_filenotfound_exits_one(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.side_effect = FileNotFoundError("no samples directory")
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.train.extractor.SampleExtractor", return_value=extractor),
        patch("dbsprout.train.serializer.DataPreparer", return_value=preparer),
    ):
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 1
    assert "no samples directory" in result.stdout


# --------------------------------------------------------------------------- #
# Task 6 — --quiet plumbing
# --------------------------------------------------------------------------- #
def test_quiet_threads_to_stages_but_keeps_summary(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(
            app,
            ["train", "--db", "sqlite:///x.db", "--output", str(out), "--quiet"],
        )
    assert result.exit_code == 0, result.stdout
    assert preparer.prepare.call_args.kwargs["quiet"] is True
    assert trainer.train.call_args.kwargs["quiet"] is True
    # summary line still printed even when quiet
    assert ".gguf" in result.stdout


# --------------------------------------------------------------------------- #
# Task 5 (cont.) — trainer.train and export stage failures
# --------------------------------------------------------------------------- #
def test_trainer_runtime_error_exits_one(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.side_effect = RuntimeError("CUDA out of memory")
    exporter = MagicMock()

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 1
    assert "CUDA out of memory" in result.stdout
    exporter.to_gguf_result.assert_not_called()


def test_export_failure_exits_one(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.side_effect = FileNotFoundError("base model not found")

    p_cfg, p_ext, p_prep, p_sel, p_exp = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_ext, p_prep, p_sel, p_exp:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 1
    assert "base model not found" in result.stdout
