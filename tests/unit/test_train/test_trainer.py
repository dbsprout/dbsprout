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
from dbsprout.train.trainer import QLoRATrainer, _cuda_available, _select_backend

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
