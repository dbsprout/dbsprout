"""Unit tests for dbsprout.train.mlx_trainer — MLX trainer (Apple Silicon).

All heavy optional deps (``mlx``, ``mlx_lm``, ``torch``) are mocked. Apple
Silicon detection is patched so backend selection is exercised on any host.
No real training ever runs; ``mlx``/``mlx-lm`` are never imported for real.
"""

from __future__ import annotations

import sys
import types
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
    from pathlib import Path


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
    """Inject a mock ``mlx_lm`` whose ``train``/``lora`` surface is captured."""
    captured: dict[str, mock.MagicMock] = {}

    lora_fn = mock.MagicMock(name="train_lora", return_value=0.2468)
    captured["lora_fn"] = lora_fn

    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.run_lora_training = lora_fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)

    # mlx core stub so any incidental ``import mlx`` succeeds.
    mlx_core = types.ModuleType("mlx")
    monkeypatch.setitem(sys.modules, "mlx", mlx_core)
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
    kwargs = fake_mlx["lora_fn"].call_args.kwargs
    assert kwargs["epochs"] == 4
    assert kwargs["learning_rate"] == pytest.approx(1e-3)
    assert kwargs["lora_rank"] == 8
    assert kwargs["lora_alpha"] == 64
    assert kwargs["lora_dropout"] == pytest.approx(0.1)
    assert kwargs["batch_size"] == 6
    assert kwargs["base_model"] == cfg.base_model


def test_train_passes_corpus_and_adapter_dir_to_backend(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    out = tmp_path / "adapters"
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        MLXTrainer().train(corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=out)
    kwargs = fake_mlx["lora_fn"].call_args.kwargs
    assert kwargs["corpus_path"] == corpus
    assert kwargs["adapter_dir"] == out / "default"


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
    fake_mlx["lora_fn"].return_value = None
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
