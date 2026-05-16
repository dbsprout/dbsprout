"""Unit tests for the `dbsprout train` end-to-end pipeline callback (S-068).

The pipeline chains SampleExtractor -> DataPreparer -> select_trainer() ->
Exporter. Components are injected via the ``_default_components`` seam (S-068
review #16) so these tests run WITHOUT the heavy ``[data]``/``[stats]`` extras
(no real ``polars``/``torch``/``mlx`` import) — patching the extractor's
import site used to force ``polars`` and fail core-only runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.train.config import LoRAAdapter, TrainConfig
from dbsprout.train.privacy import RedactionStats, TrainPrivacyConfig

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


def _components(  # noqa: PLR0913 - one mock per pipeline stage + optional seams
    extractor: MagicMock,
    preparer: MagicMock,
    trainer: MagicMock,
    exporter: MagicMock,
    *,
    select: object | None = None,
    redactor: MagicMock | None = None,
) -> MagicMock:
    """Build a fake ``_PipelineComponents`` (the injectable seam).

    No heavy import happens — the real ``_default_components`` (which lazily
    pulls polars/torch/mlx) is replaced wholesale. ``redactor`` defaults to a
    MagicMock whose ``redact_dir`` returns a real ``RedactionStats`` so the
    summary-printing path is exercised without Presidio.
    """
    if redactor is None:
        redactor = MagicMock(name="redactor")
        redactor.redact_dir.return_value = RedactionStats()
    comps = MagicMock(name="components")
    comps.make_extractor = MagicMock(return_value=extractor)
    comps.make_redactor = MagicMock(return_value=redactor)
    comps.make_preparer = MagicMock(return_value=preparer)
    comps.select_trainer = select if select is not None else MagicMock(return_value=trainer)
    comps.make_exporter = MagicMock(return_value=exporter)
    return comps


def _patches(  # noqa: PLR0913 - one mock per pipeline stage + cfg
    cfg: MagicMock,
    extractor: MagicMock,
    preparer: MagicMock,
    trainer: MagicMock,
    exporter: MagicMock,
    *,
    select: object | None = None,
    redactor: MagicMock | None = None,
):
    return (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch(
            "dbsprout.cli.commands.train._default_components",
            return_value=_components(
                extractor, preparer, trainer, exporter, select=select, redactor=redactor
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Task 1 — privacy gate + subcommand regression guards
# --------------------------------------------------------------------------- #
def test_pipeline_privacy_gate_blocks_non_local_tier(tmp_path: _Path) -> None:
    cfg = MagicMock()
    cfg.privacy.tier = "cloud"
    with (
        patch("dbsprout.cli.commands.train.load_config", return_value=cfg),
        patch("dbsprout.cli.commands.train._default_components") as fake,
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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
    select = MagicMock(side_effect=RuntimeError("MLX training requires Apple Silicon"))
    p_cfg, p_comps = _patches(cfg, extractor, preparer, MagicMock(), exporter, select=select)
    with p_cfg, p_comps:
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
    p_cfg, p_comps = _patches(cfg, extractor, MagicMock(), MagicMock(), MagicMock())
    with p_cfg, p_comps:
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
    p_cfg, p_comps = _patches(cfg, extractor, preparer, MagicMock(), MagicMock())
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
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

    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])
    assert result.exit_code == 1
    assert "base model not found" in result.stdout


# --------------------------------------------------------------------------- #
# Review #16 — injectable seam works without [data]; #17 — uniform scrubbing
# --------------------------------------------------------------------------- #
def test_train_command_module_import_is_lazy_no_polars() -> None:
    # Importing the train CLI command module must NOT pull polars/torch/mlx
    # (the heavy deps are imported lazily inside _default_components, called
    # only when the pipeline actually runs). This is what lets the suite run
    # under a dev-only env without the [data]/[stats] extras (review #16).
    import importlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    importlib.import_module("dbsprout.cli.commands.train")
    probe = (
        "import sys, dbsprout.cli.commands.train as t;"
        "assert 'polars' not in sys.modules;"
        "assert hasattr(t, '_default_components');"
        "assert {'make_extractor','make_preparer','select_trainer','make_exporter'}"
        " <= set(t._PipelineComponents.__dataclass_fields__);"
        "print('ok')"
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip() == "ok"


_SECRET = "postgresql://u:topsecretpw@h:5432/d"  # noqa: S105 - test DSN


def test_serializer_error_scrubs_secret(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.side_effect = ValueError(f"bad parquet for {_SECRET}")
    p_cfg, p_comps = _patches(cfg, extractor, preparer, MagicMock(), MagicMock())
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", _SECRET, "--output", str(out)])
    assert result.exit_code == 1
    assert "topsecretpw" not in result.stdout


def test_trainer_error_scrubs_secret(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.side_effect = RuntimeError(f"train failed connecting {_SECRET}")
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, MagicMock())
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", _SECRET, "--output", str(out)])
    assert result.exit_code == 1
    assert "topsecretpw" not in result.stdout


def test_export_error_scrubs_secret(tmp_path: _Path) -> None:
    cfg = _local_cfg()
    out = tmp_path / ".dbsprout"
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.side_effect = FileNotFoundError(f"missing for {_SECRET}")
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", _SECRET, "--output", str(out)])
    assert result.exit_code == 1
    assert "topsecretpw" not in result.stdout


# --------------------------------------------------------------------------- #
# S-070 — PII redaction + DP-SGD guard plumbing
# --------------------------------------------------------------------------- #
def _full_pipeline_mocks(out: _Path) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    extractor = MagicMock()
    extractor.extract.return_value = _sample_result(out)
    preparer = MagicMock()
    preparer.prepare.return_value = _serialize_result(out)
    trainer = MagicMock()
    trainer.train.return_value = _adapter(out)
    exporter = MagicMock()
    exporter.to_gguf_result.return_value = _export_result(out)
    return extractor, preparer, trainer, exporter


def test_pipeline_redacts_before_serialize(tmp_path: _Path) -> None:
    out = tmp_path / ".dbsprout"
    cfg = _local_cfg()
    extractor, preparer, trainer, exporter = _full_pipeline_mocks(out)
    redactor = MagicMock(name="redactor")
    redactor.redact_dir.return_value = RedactionStats(
        total_values_masked=5, entity_totals={"EMAIL_ADDRESS": 5}
    )
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter, redactor=redactor)
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])

    assert result.exit_code == 0, result.stdout
    redactor.redact_dir.assert_called_once()
    # redaction must happen before serialization
    assert redactor.redact_dir.call_args.args[0] == out / "training"
    assert "Redacted" in result.stdout
    assert "EMAIL_ADDRESS" in result.stdout


def test_pipeline_no_pii_redaction_flag_skips_redactor(tmp_path: _Path) -> None:
    out = tmp_path / ".dbsprout"
    cfg = _local_cfg()
    extractor, preparer, trainer, exporter = _full_pipeline_mocks(out)
    redactor = MagicMock(name="redactor")
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter, redactor=redactor)
    with p_cfg, p_comps:
        result = runner.invoke(
            app,
            ["train", "--db", "sqlite:///x.db", "--output", str(out), "--no-pii-redaction"],
        )

    assert result.exit_code == 0, result.stdout
    redactor.redact_dir.assert_not_called()


def test_pipeline_redaction_skipped_when_presidio_missing(tmp_path: _Path) -> None:
    out = tmp_path / ".dbsprout"
    cfg = _local_cfg()
    extractor, preparer, trainer, exporter = _full_pipeline_mocks(out)
    redactor = MagicMock(name="redactor")
    redactor.redact_dir.return_value = RedactionStats(presidio_available=False)
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter, redactor=redactor)
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])

    assert result.exit_code == 0, result.stdout
    assert "Presidio not installed" in result.stdout


def test_pipeline_dp_sgd_guard_exits_cleanly(tmp_path: _Path) -> None:
    out = tmp_path / ".dbsprout"
    cfg = MagicMock()
    cfg.privacy.tier = "local"
    cfg.train = TrainConfig(privacy=TrainPrivacyConfig(dp_sgd=True))
    extractor, preparer, trainer, exporter = _full_pipeline_mocks(out)
    p_cfg, p_comps = _patches(cfg, extractor, preparer, trainer, exporter)
    with p_cfg, p_comps:
        result = runner.invoke(app, ["train", "--db", "sqlite:///x.db", "--output", str(out)])

    assert result.exit_code == 1
    assert "DP-SGD" in result.stdout
    extractor.extract.assert_called_once()
    preparer.prepare.assert_not_called()
