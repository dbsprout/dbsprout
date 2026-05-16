"""Unit tests for dbsprout.train.trainer — QLoRA trainer with CUDA auto-detect.

All heavy optional deps (``torch``, ``unsloth``, ``trl``) are mocked. The
CUDA-detection helper is patched so backend selection is exercised without a
GPU. No real training ever runs.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from dbsprout.train.config import LoRAAdapter, TrainConfig
from dbsprout.train.privacy import TrainPrivacyConfig
from dbsprout.train.trainer import (
    QLoRATrainer,
    _cuda_available,
    _make_private,
    _select_backend,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# --- Task 3: _cuda_available ------------------------------------------------


def test_cuda_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _cuda_available() is True


def test_cuda_available_false_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _cuda_available() is False


def test_cuda_available_false_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError
    assert _cuda_available() is False


# --- Task 4: _select_backend ------------------------------------------------


def test_select_backend_unsloth_on_cuda() -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        assert _select_backend() == "unsloth"


def test_select_backend_raises_without_cuda() -> None:
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=False),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        _select_backend()


# --- Task 5: QLoRATrainer.train --------------------------------------------


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
def fake_unsloth(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, mock.MagicMock]]:
    """Inject mock ``unsloth`` / ``trl`` / ``datasets`` / ``transformers``."""
    captured: dict[str, mock.MagicMock] = {}

    model = mock.MagicMock(name="model")
    tokenizer = mock.MagicMock(name="tokenizer")
    fast_lm = mock.MagicMock(name="FastLanguageModel")
    fast_lm.from_pretrained.return_value = (model, tokenizer)
    fast_lm.get_peft_model.return_value = model
    captured["FastLanguageModel"] = fast_lm

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = fast_lm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)

    trainer_obj = mock.MagicMock(name="SFTTrainer_instance")
    trainer_obj.train.return_value = types.SimpleNamespace(
        training_loss=0.1234,
    )
    sft_trainer = mock.MagicMock(name="SFTTrainer", return_value=trainer_obj)
    captured["SFTTrainer"] = sft_trainer
    captured["trainer_obj"] = trainer_obj
    sft_config = mock.MagicMock(name="SFTConfig")
    captured["SFTConfig"] = sft_config
    trl_mod = types.ModuleType("trl")
    trl_mod.SFTTrainer = sft_trainer  # type: ignore[attr-defined]
    trl_mod.SFTConfig = sft_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl_mod)

    load_dataset = mock.MagicMock(name="load_dataset", return_value="DATASET")
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = load_dataset  # type: ignore[attr-defined]
    captured["load_dataset"] = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return captured


def test_train_returns_lora_adapter(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        trainer = QLoRATrainer()
        out = tmp_path / "adapters"
        adapter = trainer.train(corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=out)
    assert isinstance(adapter, LoRAAdapter)
    assert adapter.epochs == 1
    assert adapter.train_samples == 2
    assert adapter.final_loss == pytest.approx(0.1234)
    assert adapter.adapter_path.is_relative_to(out)
    assert adapter.duration_seconds >= 0


def test_train_writes_adapter_to_default_subdir(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "adapters",
        )
    assert adapter.adapter_path == tmp_path / "adapters" / "default"
    fake_unsloth["trainer_obj"].save_model.assert_called_once()


def test_train_uses_schema_hash_subdir(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "adapters",
            schema_hash="abc123",
        )
    assert adapter.adapter_path == tmp_path / "adapters" / "abc123"


def test_train_threads_config_into_peft(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    cfg = TrainConfig(epochs=4, learning_rate=1e-3, lora_rank=8, lora_alpha=64, lora_dropout=0.1)
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        QLoRATrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
    peft_kwargs = fake_unsloth["FastLanguageModel"].get_peft_model.call_args.kwargs
    assert peft_kwargs["r"] == 8
    assert peft_kwargs["lora_alpha"] == 64
    assert peft_kwargs["lora_dropout"] == pytest.approx(0.1)
    sft_kwargs = fake_unsloth["SFTConfig"].call_args.kwargs
    assert sft_kwargs["num_train_epochs"] == 4
    assert sft_kwargs["learning_rate"] == pytest.approx(1e-3)
    assert sft_kwargs["per_device_train_batch_size"] == cfg.batch_size


def test_train_enforces_completion_only_loss_privacy_safeguard(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    # Even if the user opts out at the config layer, the trainer forces the
    # completion-only collator on (privacy safeguard — no prompt memorization).
    cfg = TrainConfig(completion_only_loss=False)
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        QLoRATrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
    sft_kwargs = fake_unsloth["SFTConfig"].call_args.kwargs
    assert sft_kwargs["completion_only_loss"] is True


def test_train_default_config_keeps_completion_only_loss_on(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        QLoRATrainer().train(corpus_path=corpus, config=TrainConfig(), output_dir=tmp_path / "a")
    assert fake_unsloth["SFTConfig"].call_args.kwargs["completion_only_loss"] is True


def test_train_raises_without_cuda(corpus: Path, tmp_path: Path) -> None:
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=False),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_train_raises_when_corpus_missing(tmp_path: Path) -> None:
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=True),
        pytest.raises(FileNotFoundError, match="train serialize"),
    ):
        QLoRATrainer().train(
            corpus_path=tmp_path / "missing.jsonl",
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_train_raises_when_corpus_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=True),
        pytest.raises(ValueError, match="empty"),
    ):
        QLoRATrainer().train(
            corpus_path=empty,
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_train_quiet_suppresses_progress_but_returns(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "a",
            quiet=True,
        )
    assert adapter.train_samples == 2


def test_train_handles_no_loss_history(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    fake_unsloth["trainer_obj"].train.return_value = types.SimpleNamespace(training_loss=None)
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "a",
        )
    assert adapter.final_loss is None


# --- Review #2: partial-install (datasets/trl) -> friendly RuntimeError ------


def test_train_partial_install_missing_datasets_raises_hint(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # unsloth present but ``datasets`` absent (a real partial-install state):
    # must raise the friendly install hint, not a bare ImportError.
    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = mock.MagicMock(name="FastLanguageModel")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    monkeypatch.setitem(sys.modules, "datasets", None)  # forces ImportError
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        QLoRATrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=tmp_path / "a"
        )


def test_train_partial_install_missing_trl_raises_hint(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = mock.MagicMock(name="FastLanguageModel")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = mock.MagicMock(name="load_dataset")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "trl", None)  # forces ImportError
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"),
    ):
        QLoRATrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=tmp_path / "a"
        )


# --- Review #3: cheap local precondition checked before GPU/backend ---------


def test_corpus_missing_checked_before_backend_selection(tmp_path: Path) -> None:
    # CPU host (no CUDA) + missing corpus: the user should get the actionable
    # "run train serialize" corpus error, not the GPU/install hint. This pins
    # the precondition order so a cheap local check never hides behind the
    # backend probe.
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=False),
        pytest.raises(FileNotFoundError, match="train serialize"),
    ):
        QLoRATrainer().train(
            corpus_path=tmp_path / "missing.jsonl",
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


def test_corpus_empty_checked_before_backend_selection(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=False),
        pytest.raises(ValueError, match="empty"),
    ):
        QLoRATrainer().train(
            corpus_path=empty,
            config=TrainConfig(),
            output_dir=tmp_path / "a",
        )


# --- Review #4: GReaT single-text corpus has no prompt by construction ------


def test_corpus_format_invariant_single_text_field_no_prompt_split(corpus: Path) -> None:
    # The GReaT serializer (S-063) emits a single ``text`` field per row with
    # NO prompt/completion split. ``completion_only_loss`` is therefore a
    # structural no-op safeguard: there is nothing to memorize by construction.
    # This test pins that corpus-format invariant so the privacy AC stays
    # honest even if the serializer changes.
    import json  # noqa: PLC0415

    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert set(row) <= {"text", "table"}
        assert "text" in row
        assert "prompt" not in row
        assert "completion" not in row


# --- Review #5: training_loss coercion guarded against bad types ------------


def test_train_handles_non_numeric_training_loss(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    # A backend returning a non-numeric ``training_loss`` (e.g. a string) must
    # degrade to ``final_loss=None`` rather than crash the whole run.
    fake_unsloth["trainer_obj"].train.return_value = types.SimpleNamespace(
        training_loss="not-a-number"
    )
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus,
            config=TrainConfig(epochs=1),
            output_dir=tmp_path / "a",
        )
    assert adapter.final_loss is None


# --- S-097: _make_private seam ---------------------------------------------


@pytest.fixture
def fake_opacus(monkeypatch: pytest.MonkeyPatch) -> dict[str, mock.MagicMock]:
    captured: dict[str, mock.MagicMock] = {}
    engine = mock.MagicMock(name="PrivacyEngine_instance")
    engine.make_private_with_epsilon.side_effect = lambda **kw: (
        kw["module"],
        kw["optimizer"],
        kw["data_loader"],
    )
    engine.make_private.side_effect = lambda **kw: (
        kw["module"],
        kw["optimizer"],
        kw["data_loader"],
    )
    engine.get_epsilon.return_value = 5.5
    engine_cls = mock.MagicMock(name="PrivacyEngine", return_value=engine)
    captured["engine"] = engine
    captured["engine_cls"] = engine_cls
    opacus_mod = types.ModuleType("opacus")
    opacus_mod.PrivacyEngine = engine_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opacus", opacus_mod)
    return captured


def test_make_private_epsilon_mode(fake_opacus: dict[str, mock.MagicMock]) -> None:
    priv = TrainPrivacyConfig(dp_sgd=True, dp_target_epsilon=8.0, dp_max_grad_norm=1.2)
    m, o, d, eps = _make_private(privacy=priv, model="M", optimizer="O", data_loader="D", epochs=3)
    fake_opacus["engine"].make_private_with_epsilon.assert_called_once()
    kw = fake_opacus["engine"].make_private_with_epsilon.call_args.kwargs
    assert kw["target_epsilon"] == pytest.approx(8.0)
    assert kw["target_delta"] == pytest.approx(1e-5)
    assert kw["max_grad_norm"] == pytest.approx(1.2)
    assert kw["epochs"] == 3
    assert (m, o, d) == ("M", "O", "D")
    assert eps == pytest.approx(8.0)


def test_make_private_noise_mode(fake_opacus: dict[str, mock.MagicMock]) -> None:
    priv = TrainPrivacyConfig(dp_sgd=True, dp_noise_multiplier=1.1)
    _m, _o, _d, eps = _make_private(
        privacy=priv, model="M", optimizer="O", data_loader="D", epochs=2
    )
    fake_opacus["engine"].make_private.assert_called_once()
    kw = fake_opacus["engine"].make_private.call_args.kwargs
    assert kw["noise_multiplier"] == pytest.approx(1.1)
    assert kw["max_grad_norm"] == pytest.approx(1.0)
    assert eps == pytest.approx(5.5)  # from engine.get_epsilon


def test_make_private_noise_mode_accountant_unavailable(
    fake_opacus: dict[str, mock.MagicMock],
) -> None:
    fake_opacus["engine"].get_epsilon.side_effect = AttributeError("no accountant")
    priv = TrainPrivacyConfig(dp_sgd=True, dp_noise_multiplier=1.1)
    _m, _o, _d, eps = _make_private(
        privacy=priv, model="M", optimizer="O", data_loader="D", epochs=2
    )
    assert eps is None


def test_make_private_raises_when_opacus_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opacus", None)  # forces ImportError
    priv = TrainPrivacyConfig(dp_sgd=True, dp_target_epsilon=8.0)
    with pytest.raises(RuntimeError, match=r"dbsprout\[train-dp\]"):
        _make_private(privacy=priv, model="M", optimizer="O", data_loader="D", epochs=1)


# --- S-097: DP-SGD wired into QLoRATrainer._run_unsloth --------------------


def test_train_no_dp_leaves_adapter_epsilon_none(
    corpus: Path, tmp_path: Path, fake_unsloth: dict[str, mock.MagicMock]
) -> None:
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(
            corpus_path=corpus, config=TrainConfig(epochs=1), output_dir=tmp_path / "a"
        )
    assert adapter.achieved_epsilon is None
    assert adapter.dp_delta is None


def test_train_dp_epsilon_mode_threads_guarantee(
    corpus: Path,
    tmp_path: Path,
    fake_unsloth: dict[str, mock.MagicMock],
    fake_opacus: dict[str, mock.MagicMock],
) -> None:
    cfg = TrainConfig(
        epochs=2,
        privacy=TrainPrivacyConfig(dp_sgd=True, dp_target_epsilon=6.0),
    )
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
    fake_opacus["engine"].make_private_with_epsilon.assert_called_once()
    assert adapter.achieved_epsilon == pytest.approx(6.0)
    assert adapter.dp_delta == pytest.approx(1e-5)
    # privatized objects reassigned onto the SFTTrainer before train()
    assert fake_unsloth["trainer_obj"].train.called


def test_train_dp_noise_mode_threads_accountant_epsilon(
    corpus: Path,
    tmp_path: Path,
    fake_unsloth: dict[str, mock.MagicMock],
    fake_opacus: dict[str, mock.MagicMock],
) -> None:
    cfg = TrainConfig(
        epochs=1,
        privacy=TrainPrivacyConfig(dp_sgd=True, dp_noise_multiplier=1.1),
    )
    with mock.patch("dbsprout.train.trainer._cuda_available", return_value=True):
        adapter = QLoRATrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
    fake_opacus["engine"].make_private.assert_called_once()
    assert adapter.achieved_epsilon == pytest.approx(5.5)
    assert adapter.dp_delta == pytest.approx(1e-5)


def test_train_dp_missing_opacus_raises_hint(
    corpus: Path,
    tmp_path: Path,
    fake_unsloth: dict[str, mock.MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opacus", None)  # forces ImportError
    cfg = TrainConfig(privacy=TrainPrivacyConfig(dp_sgd=True, dp_target_epsilon=6.0))
    with (
        mock.patch("dbsprout.train.trainer._cuda_available", return_value=True),
        pytest.raises(RuntimeError, match=r"dbsprout\[train-dp\]"),
    ):
        QLoRATrainer().train(corpus_path=corpus, config=cfg, output_dir=tmp_path / "a")
