"""Unit tests for dbsprout.train.mlx_trainer — MLX trainer (Apple Silicon).

All heavy optional deps (``mlx``, ``mlx_lm``, ``torch``) are mocked. Apple
Silicon detection is patched so backend selection is exercised on any host.
No real training ever runs; ``mlx``/``mlx-lm`` are never imported for real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

import dbsprout.train.mlx_trainer as mlx_mod
from dbsprout.train.config import LoRAAdapter, TrainConfig
from dbsprout.train.mlx_trainer import (
    MLXTrainer,
    _mlx_available,
    _select_backend,
    select_trainer,
)
from dbsprout.train.trainer import QLoRATrainer

if TYPE_CHECKING:
    from collections.abc import Iterator


# --- Task 1: _mlx_available -------------------------------------------------


def test_mlx_available_true_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert _mlx_available() is True


def test_mlx_available_false_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert _mlx_available() is False


def test_mlx_available_false_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert _mlx_available() is False


# --- Task 2: _select_backend ------------------------------------------------


def test_select_backend_mlx_on_darwin_without_cuda() -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        assert _select_backend() == "mlx"


def test_select_backend_raises_on_cuda_host() -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        _select_backend()


def test_select_backend_raises_off_apple_silicon() -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=False),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"),
    ):
        _select_backend()


# --- Task 3: select_trainer dispatcher --------------------------------------


def test_select_trainer_returns_qlora_on_cuda() -> None:
    with mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=True):
        assert isinstance(select_trainer(), QLoRATrainer)


def test_select_trainer_returns_mlx_on_apple_silicon() -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        assert isinstance(select_trainer(), MLXTrainer)


def test_select_trainer_raises_when_no_backend() -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=False),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"),
    ):
        select_trainer()


# --- Task 4 & 5: MLXTrainer.train -------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"text": "[users] id is 1, name is Ada", "table": "users"}\n'
        '{"text": "[users] id is 2, name is Linus", "table": "users"}\n',
        encoding="utf-8",
    )
    return p


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, mock.MagicMock]]:
    """Patch the REAL mlx-lm symbol seam (no fabricated module API).

    Mocks the *correct* entrypoints — ``load``/``linear_to_lora_layers``/
    ``TrainingArgs``/``train`` (the actual ``mlx_lm.tuner`` surface) — so a
    speculative non-existent API (e.g. ``mlx_lm.run_lora_training``) can no
    longer hide behind the mock. ``_load_mlx_lm_symbols`` is the single seam.
    """
    captured: dict[str, mock.MagicMock] = {}

    model = mock.MagicMock(name="model")
    tokenizer = mock.MagicMock(name="tokenizer")
    load = mock.MagicMock(name="load", return_value=(model, tokenizer))
    linear_to_lora_layers = mock.MagicMock(name="linear_to_lora_layers")
    training_args = mock.MagicMock(name="TrainingArgs")
    optimizer = mock.MagicMock(name="optimizer")
    make_optimizer = mock.MagicMock(name="make_optimizer", return_value=optimizer)
    load_dataset = mock.MagicMock(name="load_dataset", return_value="DATASET")

    def _train(*_a: object, **kw: object) -> None:
        # The real mlx_lm.tuner.trainer.train reports loss via a callback and
        # persists adapters.safetensors itself; emulate the persist so the
        # adapter-dir contract is exercised.
        adapter_file = kw.get("adapter_file")
        if isinstance(adapter_file, str):
            Path(adapter_file).write_bytes(b"\x00")
        cb = kw.get("training_callback")
        if cb is not None:
            cb(0.2468)  # type: ignore[operator]

    train = mock.MagicMock(name="train", side_effect=_train)

    captured.update(
        load=load,
        linear_to_lora_layers=linear_to_lora_layers,
        TrainingArgs=training_args,
        train=train,
        make_optimizer=make_optimizer,
        load_dataset=load_dataset,
    )

    monkeypatch.setattr(
        "dbsprout.train.mlx_trainer._load_mlx_lm_symbols",
        lambda: {
            "load": load,
            "linear_to_lora_layers": linear_to_lora_layers,
            "TrainingArgs": training_args,
            "train": train,
            "make_optimizer": make_optimizer,
            "load_dataset": load_dataset,
        },
    )
    return captured


def test_train_returns_lora_adapter(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        out = tmp_path / "adapters"
        adapter = MLXTrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=out
        )
    assert isinstance(adapter, LoRAAdapter)
    assert adapter.epochs == 1
    assert adapter.train_samples == 2
    assert adapter.final_loss == pytest.approx(0.2468)
    assert adapter.adapter_path.is_relative_to(out)
    assert adapter.duration_seconds >= 0


def test_train_writes_adapter_to_default_subdir(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        adapter = MLXTrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "adapters",
        )
    assert adapter.adapter_path == tmp_path / "adapters" / "default"
    assert adapter.adapter_path.is_dir()


def test_train_uses_schema_hash_subdir(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        adapter = MLXTrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "adapters",
            schema_hash="deadbeef",
        )
    assert adapter.adapter_path == tmp_path / "adapters" / "deadbeef"


def test_train_threads_config_into_mlx(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    cfg = TrainConfig(
        epochs=4, learning_rate=1e-3, lora_rank=8, lora_alpha=64, lora_dropout=0.1, batch_size=6
    )
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        MLXTrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
    # base model -> real load() entrypoint
    assert fake_mlx["load"].call_args.args[0] == cfg.base_model
    # LoRA rank/alpha/dropout -> real linear_to_lora_layers() config
    lora_cfg = fake_mlx["linear_to_lora_layers"].call_args.args[2]
    assert lora_cfg["rank"] == 8
    assert lora_cfg["scale"] == pytest.approx(64 / 8)  # alpha / rank
    assert lora_cfg["dropout"] == pytest.approx(0.1)
    # batch size + iters (epochs) -> real TrainingArgs
    ta_kwargs = fake_mlx["TrainingArgs"].call_args.kwargs
    assert ta_kwargs["batch_size"] == 6
    assert ta_kwargs["iters"] >= 4  # epochs threaded through
    # learning rate -> optimizer
    assert fake_mlx["make_optimizer"].call_args.kwargs["learning_rate"] == pytest.approx(1e-3)


def test_train_passes_corpus_and_adapter_dir_to_backend(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    out = tmp_path / "adapters"
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        MLXTrainer().train(corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=out)
    # corpus path flows into the real dataset loader
    assert fake_mlx["load_dataset"].call_args.args[0] == corpus
    # adapters.safetensors is written into the adapter dir by the real train()
    adapter_dir = out / "default"
    train_kwargs = fake_mlx["train"].call_args.kwargs
    assert train_kwargs["adapter_file"] == str(adapter_dir / "adapters.safetensors")
    assert (adapter_dir / "adapters.safetensors").is_file()


def test_train_quiet_suppresses_progress_but_returns(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        adapter = MLXTrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "a",
            quiet=True,
        )
    assert adapter.train_samples == 2


def test_train_handles_no_loss_history(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    # The real train() reports loss via callback; no callback invocation =>
    # no loss history => final_loss is None (never a crash).
    fake_mlx["train"].side_effect = None
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        adapter = MLXTrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "a",
        )
    assert adapter.final_loss is None


# --- Task 6: error paths ----------------------------------------------------


def test_train_raises_on_cuda_host(corpus: Path, tmp_path: Path) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        MLXTrainer().train(corpus_path=corpus, config=TrainConfig(), output_dir=tmp_path / "a")


def test_train_raises_off_apple_silicon(corpus: Path, tmp_path: Path) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=False),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"),
    ):
        MLXTrainer().train(corpus_path=corpus, config=TrainConfig(), output_dir=tmp_path / "a")


def test_train_raises_when_corpus_missing(tmp_path: Path) -> None:
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
        pytest.raises(FileNotFoundError, match="train serialize"),
    ):
        MLXTrainer().train(
            corpus_path=tmp_path / "missing.jsonl",
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_train_raises_when_corpus_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
        pytest.raises(ValueError, match="empty"),
    ):
        MLXTrainer().train(
            corpus_path=empty,
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_train_raises_when_mlx_lm_missing(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the lazy ``import mlx_lm`` to fail (package not installed).
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"),
    ):
        MLXTrainer().train(
            corpus_path=corpus,
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


# --- AC-7: module imports cleanly without mlx installed ---------------------


def test_module_imports_without_mlx_installed() -> None:
    # The top-of-file import already proves the module is importable in the
    # dev-only env (no mlx/mlx-lm); assert the public surface is intact.
    assert hasattr(mlx_mod, "MLXTrainer")
    assert hasattr(mlx_mod, "select_trainer")
    assert "mlx_lm" not in dir(mlx_mod)


# --- Review #8: select_trainer returns a structural _Trainer ---------------


def test_select_trainer_satisfies_trainer_protocol() -> None:
    from dbsprout.train.mlx_trainer import _Trainer  # noqa: PLC0415

    with mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=True):
        cuda = select_trainer()
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        mlx = select_trainer()
    # Both trainers satisfy the runtime-checkable _Trainer Protocol (removes
    # the S-068 `# type: ignore[attr-defined]` on trainer.train()).
    assert isinstance(cuda, _Trainer)
    assert isinstance(mlx, _Trainer)


# --- Review #6: real mlx-lm symbol surface (integration, never mocked) ------


@pytest.mark.integration
def test_real_mlx_lm_symbol_surface_exists() -> None:
    """Pin the REAL mlx-lm LoRA API so a fabricated symbol cannot regress.

    Skipped unless ``mlx_lm`` is genuinely importable AND on Apple Silicon —
    so a unit run on Linux/CI (where mlx-lm cannot install) never hides a
    non-existent-API bug behind a mock. Real Apple-Silicon execution remains
    hardware-validation-pending (like the perf ACs).
    """
    import importlib.util  # noqa: PLC0415
    import platform  # noqa: PLC0415

    if platform.system() != "Darwin" or importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("requires real mlx-lm on Apple Silicon (hardware-validation-pending)")

    from mlx_lm.tuner.trainer import TrainingArgs, train  # noqa: PLC0415
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415
    from mlx_lm.utils import load  # noqa: PLC0415

    assert callable(load)
    assert callable(linear_to_lora_layers)
    assert callable(train)
    assert callable(TrainingArgs)
    # The seam must resolve exactly these real symbols (no fabricated names).
    syms = mlx_mod._load_mlx_lm_symbols()
    assert {"load", "linear_to_lora_layers", "TrainingArgs", "train", "make_optimizer"} <= set(syms)
