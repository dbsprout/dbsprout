"""Unit tests for dbsprout.train.mlx_trainer — MLX trainer (Apple Silicon).

All heavy optional deps (``mlx``, ``mlx_lm``, ``torch``) are mocked. Apple
Silicon detection is patched so backend selection is exercised on any host.
No real training ever runs; ``mlx``/``mlx-lm`` are never imported for real.
"""

from __future__ import annotations

import sys
import types
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
from dbsprout.train.privacy import TrainPrivacyConfig
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


class _FakeTrainingCallback:
    """Stand-in for the real ``mlx_lm.tuner.trainer.TrainingCallback`` base.

    The real base class defines no-op ``on_train_loss_report(dict)`` /
    ``on_val_loss_report(dict)`` hooks; the trainer subclasses it. Mirroring
    the *real* base (not a MagicMock) means the subclass MRO/`super().__init__`
    behaves as it would on Apple Silicon.
    """

    def on_train_loss_report(self, info: dict[str, object]) -> None:
        pass

    def on_val_loss_report(self, info: dict[str, object]) -> None:
        pass


@pytest.fixture
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, mock.MagicMock]]:
    """Patch the VERIFIED-REAL mlx-lm symbol seam (no fabricated module API).

    Mocks exactly the entrypoints verified against the pinned ``mlx-lm``
    source — ``load``/``save_config``/``TextDataset``/``CacheDataset``/
    ``linear_to_lora_layers``/``TrainingArgs``/``train``/``TrainingCallback``/
    ``make_optimizer``/``seed`` — so a speculative non-existent API (e.g.
    ``mlx_lm.run_lora_training`` or a ``completion_only_loss`` ``train()``
    kwarg) can no longer hide behind the mock. ``_load_mlx_lm_symbols`` is the
    single seam. The fake ``train`` emulates the real contract: it reads
    ``adapter_file`` from the ``TrainingArgs`` instance (NOT from a ``train()``
    kwarg), persists ``adapters.safetensors``, and reports loss via
    ``training_callback.on_train_loss_report({...})`` (NOT ``cb(value)``).
    """
    captured: dict[str, mock.MagicMock] = {}

    model = mock.MagicMock(name="model")
    model.layers = [mock.MagicMock(name=f"layer{i}") for i in range(3)]
    tokenizer = mock.MagicMock(name="tokenizer")
    load = mock.MagicMock(name="load", return_value=(model, tokenizer))
    save_config = mock.MagicMock(name="save_config")
    text_dataset = mock.MagicMock(name="TextDataset", return_value="TEXT_DS")
    cache_dataset = mock.MagicMock(name="CacheDataset", return_value="CACHED_DS")
    linear_to_lora_layers = mock.MagicMock(name="linear_to_lora_layers")
    optimizer = mock.MagicMock(name="optimizer")
    make_optimizer = mock.MagicMock(name="make_optimizer", return_value=optimizer)
    seed = mock.MagicMock(name="seed")

    def _make_args(**kw: object) -> mock.MagicMock:
        # Real TrainingArgs is a dataclass; the instance carries adapter_file
        # and iters as attributes. Reflect the passed kwargs onto the instance.
        inst = mock.MagicMock(name="TrainingArgsInstance")
        for key, val in kw.items():
            setattr(inst, key, val)
        return inst

    training_args = mock.MagicMock(name="TrainingArgs", side_effect=_make_args)

    def _train(*_a: object, **kw: object) -> None:
        # The real mlx_lm.tuner.trainer.train persists adapters.safetensors at
        # args.adapter_file and reports loss to a TrainingCallback instance.
        args = kw.get("args")
        adapter_file = getattr(args, "adapter_file", None)
        if isinstance(adapter_file, str):
            Path(adapter_file).write_bytes(b"\x00")
        cb = kw.get("training_callback")
        if cb is not None:
            cb.on_train_loss_report({"iteration": 1, "train_loss": 0.2468})

    train = mock.MagicMock(name="train", side_effect=_train)

    captured.update(
        load=load,
        save_config=save_config,
        TextDataset=text_dataset,
        CacheDataset=cache_dataset,
        linear_to_lora_layers=linear_to_lora_layers,
        TrainingArgs=training_args,
        train=train,
        make_optimizer=make_optimizer,
        seed=seed,
    )

    monkeypatch.setattr(
        "dbsprout.train.mlx_trainer._load_mlx_lm_symbols",
        lambda: {
            "load": load,
            "save_config": save_config,
            "TextDataset": text_dataset,
            "CacheDataset": cache_dataset,
            "linear_to_lora_layers": linear_to_lora_layers,
            "TrainingArgs": training_args,
            "train": train,
            "TrainingCallback": _FakeTrainingCallback,
            "make_optimizer": make_optimizer,
            "seed": seed,
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
    # real linear_to_lora_layers(model, num_layers, config): 2nd positional
    # arg is num_layers (len(model.layers)==3 in the fixture), NOT the rank.
    lora_call = fake_mlx["linear_to_lora_layers"].call_args
    assert lora_call.args[1] == 3
    lora_cfg = lora_call.args[2]
    assert lora_cfg["rank"] == 8
    assert lora_cfg["scale"] == pytest.approx(64 / 8)  # alpha / rank
    assert lora_cfg["dropout"] == pytest.approx(0.1)
    # batch size + iters (epochs) -> real TrainingArgs (dataclass) kwargs
    ta_kwargs = fake_mlx["TrainingArgs"].call_args.kwargs
    assert ta_kwargs["batch_size"] == 6
    assert ta_kwargs["iters"] >= 4  # epochs threaded through
    # adapter_file is a TrainingArgs FIELD, never a train() kwarg
    assert "adapter_file" in ta_kwargs
    assert "adapter_file" not in fake_mlx["train"].call_args.kwargs
    # real train() is called with keyword args (verified signature shape)
    train_kwargs = fake_mlx["train"].call_args.kwargs
    assert set(train_kwargs) >= {"model", "optimizer", "train_dataset", "args", "training_callback"}
    # no fabricated completion_only_loss kwarg on real train()
    assert "completion_only_loss" not in train_kwargs
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
    # parsed JSONL rows flow into the real TextDataset(rows, tokenizer)
    td_rows = fake_mlx["TextDataset"].call_args.args[0]
    assert [r["text"] for r in td_rows] == [
        "[users] id is 1, name is Ada",
        "[users] id is 2, name is Linus",
    ]
    # TextDataset is wrapped in the real CacheDataset
    assert fake_mlx["CacheDataset"].call_args.args[0] == "TEXT_DS"
    # adapters.safetensors written by real train() via TrainingArgs.adapter_file
    adapter_dir = out / "default"
    ta_kwargs = fake_mlx["TrainingArgs"].call_args.kwargs
    assert ta_kwargs["adapter_file"] == str(adapter_dir / "adapters.safetensors")
    assert (adapter_dir / "adapters.safetensors").is_file()
    # minimal real adapter_config.json written for the S-066 GGUF exporter
    cfg_call = fake_mlx["save_config"].call_args
    cfg_payload = cfg_call.args[0]
    assert cfg_payload["fine_tune_type"] == "lora"
    assert cfg_payload["base_model"] == TrainConfig().base_model
    assert Path(cfg_call.args[1]) == adapter_dir / "adapter_config.json"


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


def test_train_tolerates_non_callable_seed(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    # If the seam ever resolves a non-callable ``seed`` the trainer must skip
    # seeding gracefully rather than crash (defensive guard in _run_mlx).
    seam = mlx_mod._load_mlx_lm_symbols()
    seam["seed"] = None  # type: ignore[assignment]
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
        mock.patch("dbsprout.train.mlx_trainer._load_mlx_lm_symbols", return_value=seam),
    ):
        adapter = MLXTrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=tmp_path / "a"
        )
    assert adapter.train_samples == 2
    fake_mlx["seed"].assert_not_called()


def test_train_tolerates_model_without_freeze_or_layers(
    corpus: Path, tmp_path: Path, fake_mlx: dict[str, mock.MagicMock]
) -> None:
    # A bare model object (no .freeze / no .layers) must not crash: num_layers
    # falls back to 0 and freeze is skipped (defensive guards in _run_mlx).
    bare_model = object()
    fake_mlx["load"].return_value = (bare_model, mock.MagicMock(name="tokenizer"))
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
    ):
        MLXTrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=tmp_path / "a"
        )
    # num_layers defaulted to 0 -> threaded into linear_to_lora_layers + config
    assert fake_mlx["linear_to_lora_layers"].call_args.args[1] == 0
    assert fake_mlx["save_config"].call_args.args[0]["num_layers"] == 0


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


# --- Task 1: _load_mlx_lm_symbols resolves the verified-real surface --------


_VERIFIED_SYMBOLS = {
    "load",
    "save_config",
    "TextDataset",
    "CacheDataset",
    "linear_to_lora_layers",
    "TrainingArgs",
    "train",
    "TrainingCallback",
    "make_optimizer",
    "seed",
}


def test_load_mlx_lm_symbols_returns_verified_real_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seam resolves exactly the verified-real mlx-lm symbols (no fabrications).

    mlx-lm cannot install on Linux/CI, so the real lazy imports are stubbed
    with fake modules placed on ``sys.modules`` at the *verified* paths
    (``mlx_lm.utils``, ``mlx_lm.tuner.datasets``, ``mlx_lm.tuner.trainer``,
    ``mlx_lm.tuner.utils``, ``mlx.optimizers``, ``mlx.core``). A fabricated
    name (e.g. ``run_lora_training``) cannot satisfy this test because the
    key set is asserted for *exact* equality.
    """
    adam = mock.MagicMock(name="Adam", return_value="OPTIMIZER")
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.random = mock.MagicMock(name="mlx.core.random")
    mlx_optim = types.ModuleType("mlx.optimizers")
    mlx_optim.Adam = adam  # type: ignore[attr-defined]

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_utils = types.ModuleType("mlx_lm.utils")
    mlx_lm_utils.load = mock.MagicMock(name="load")  # type: ignore[attr-defined]
    mlx_lm_utils.save_config = mock.MagicMock(name="save_config")  # type: ignore[attr-defined]
    mlx_lm_tuner = types.ModuleType("mlx_lm.tuner")
    mlx_lm_datasets = types.ModuleType("mlx_lm.tuner.datasets")
    mlx_lm_datasets.TextDataset = mock.MagicMock(name="TextDataset")  # type: ignore[attr-defined]
    mlx_lm_datasets.CacheDataset = mock.MagicMock(name="CacheDataset")  # type: ignore[attr-defined]
    mlx_lm_trainer = types.ModuleType("mlx_lm.tuner.trainer")
    mlx_lm_trainer.TrainingArgs = mock.MagicMock(name="TrainingArgs")  # type: ignore[attr-defined]
    mlx_lm_trainer.train = mock.MagicMock(name="train")  # type: ignore[attr-defined]
    mlx_lm_trainer.TrainingCallback = type("TrainingCallback", (), {})  # type: ignore[attr-defined]
    mlx_lm_tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    mlx_lm_tuner_utils.linear_to_lora_layers = mock.MagicMock(  # type: ignore[attr-defined]
        name="linear_to_lora_layers"
    )

    for name, module in {
        "mlx": mlx_pkg,
        "mlx.core": mlx_core,
        "mlx.optimizers": mlx_optim,
        "mlx_lm": mlx_lm_pkg,
        "mlx_lm.utils": mlx_lm_utils,
        "mlx_lm.tuner": mlx_lm_tuner,
        "mlx_lm.tuner.datasets": mlx_lm_datasets,
        "mlx_lm.tuner.trainer": mlx_lm_trainer,
        "mlx_lm.tuner.utils": mlx_lm_tuner_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    syms = mlx_mod._load_mlx_lm_symbols()

    assert set(syms) == _VERIFIED_SYMBOLS
    # make_optimizer must wrap the real mlx.optimizers.Adam factory.
    assert syms["make_optimizer"](learning_rate=1e-3) == "OPTIMIZER"
    adam.assert_called_once_with(learning_rate=1e-3)


def test_load_mlx_lm_symbols_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing mlx-lm raises an actionable RuntimeError (never silent success)."""
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"):
        mlx_mod._load_mlx_lm_symbols()


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

    import mlx.core as mx  # noqa: PLC0415
    import mlx.optimizers as optim  # noqa: PLC0415
    from mlx_lm.tuner.datasets import CacheDataset, TextDataset  # noqa: PLC0415
    from mlx_lm.tuner.trainer import (  # noqa: PLC0415
        TrainingArgs,
        TrainingCallback,
        train,
    )
    from mlx_lm.tuner.utils import linear_to_lora_layers  # noqa: PLC0415
    from mlx_lm.utils import load, save_config  # noqa: PLC0415

    # Every real symbol the trainer relies on must genuinely resolve.
    assert callable(load)
    assert callable(save_config)
    assert isinstance(TextDataset, type)
    assert isinstance(CacheDataset, type)
    assert callable(linear_to_lora_layers)
    assert callable(TrainingArgs)
    assert callable(train)
    assert isinstance(TrainingCallback, type)
    assert callable(optim.Adam)
    assert callable(mx.random.seed)
    # The seam must resolve EXACTLY these verified-real symbols — exact-set
    # equality means a stray fabricated key (e.g. ``run_lora_training``) or a
    # dropped real one fails the pin, so the S-065 class of bug cannot recur.
    syms = mlx_mod._load_mlx_lm_symbols()
    assert set(syms) == {
        "load",
        "save_config",
        "TextDataset",
        "CacheDataset",
        "linear_to_lora_layers",
        "TrainingArgs",
        "train",
        "TrainingCallback",
        "make_optimizer",
        "seed",
    }


# --- S-097: MLX backend rejects DP-SGD -------------------------------------


def test_mlx_train_rejects_dp_sgd(tmp_path: Path) -> None:
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"text": "x"}\n', encoding="utf-8")
    cfg = TrainConfig(privacy=TrainPrivacyConfig(dp_sgd=True, dp_target_epsilon=8.0))
    with (
        mock.patch("dbsprout.train.mlx_trainer._cuda_available", return_value=False),
        mock.patch("dbsprout.train.mlx_trainer._mlx_available", return_value=True),
        pytest.raises(RuntimeError, match="not supported on the MLX"),
    ):
        MLXTrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
