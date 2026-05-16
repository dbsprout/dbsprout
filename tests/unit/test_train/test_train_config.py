"""Unit tests for dbsprout.train.config — TrainConfig and LoRAAdapter models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dbsprout.train.config import LoRAAdapter, TrainConfig

# --- Task 1: TrainConfig ----------------------------------------------------


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.epochs == 3
    assert cfg.learning_rate == pytest.approx(2e-4)
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == pytest.approx(0.05)
    assert cfg.batch_size == 2
    assert cfg.completion_only_loss is True
    assert cfg.base_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_train_config_is_frozen() -> None:
    cfg = TrainConfig()
    with pytest.raises(ValidationError):
        cfg.epochs = 9  # type: ignore[misc]


def test_train_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(unknown_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"lora_rank": 0},
        {"lora_alpha": 0},
        {"lora_dropout": 1.0},
        {"lora_dropout": -0.1},
        {"batch_size": 0},
    ],
)
def test_train_config_validation_errors(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_train_config_accepts_overrides() -> None:
    cfg = TrainConfig(
        epochs=5,
        learning_rate=1e-3,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        batch_size=4,
        base_model="some/other-model",
    )
    assert cfg.epochs == 5
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.lora_rank == 8
    assert cfg.base_model == "some/other-model"


def test_train_config_completion_only_loss_can_be_set_but_defaults_on() -> None:
    # The privacy default is True; explicit opt-out is allowed at the model
    # layer but the trainer always asserts the safeguard before training.
    assert TrainConfig().completion_only_loss is True
    assert TrainConfig(completion_only_loss=False).completion_only_loss is False


# --- Task 2: LoRAAdapter ----------------------------------------------------


def test_lora_adapter_fields() -> None:
    adapter = LoRAAdapter(
        adapter_path=Path(".dbsprout/models/adapters/default"),
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        epochs=3,
        train_samples=1000,
        final_loss=0.42,
        duration_seconds=12.5,
    )
    assert adapter.adapter_path == Path(".dbsprout/models/adapters/default")
    assert adapter.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert adapter.epochs == 3
    assert adapter.train_samples == 1000
    assert adapter.final_loss == pytest.approx(0.42)
    assert adapter.duration_seconds == pytest.approx(12.5)


def test_lora_adapter_is_frozen() -> None:
    adapter = LoRAAdapter(
        adapter_path=Path("x"),
        base_model="m",
        epochs=1,
        train_samples=1,
        final_loss=None,
        duration_seconds=0.0,
    )
    with pytest.raises(ValidationError):
        adapter.epochs = 2  # type: ignore[misc]


def test_lora_adapter_final_loss_optional() -> None:
    adapter = LoRAAdapter(
        adapter_path=Path("x"),
        base_model="m",
        epochs=1,
        train_samples=1,
        final_loss=None,
        duration_seconds=0.0,
    )
    assert adapter.final_loss is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": -1},
        {"train_samples": -1},
        {"duration_seconds": -0.1},
    ],
)
def test_lora_adapter_validation_errors(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "adapter_path": Path("x"),
        "base_model": "m",
        "epochs": 1,
        "train_samples": 1,
        "final_loss": None,
        "duration_seconds": 0.0,
    }
    base.update(kwargs)
    with pytest.raises(ValidationError):
        LoRAAdapter(**base)  # type: ignore[arg-type]


# --- S-097: LoRAAdapter DP guarantee fields --------------------------------


def test_lora_adapter_dp_fields_default_none() -> None:
    a = LoRAAdapter(
        adapter_path=Path("x"),
        base_model="m",
        epochs=1,
        train_samples=2,
        final_loss=0.1,
        duration_seconds=1.0,
    )
    assert a.achieved_epsilon is None
    assert a.dp_delta is None


def test_lora_adapter_dp_fields_set() -> None:
    a = LoRAAdapter(
        adapter_path=Path("x"),
        base_model="m",
        epochs=1,
        train_samples=2,
        final_loss=0.1,
        duration_seconds=1.0,
        achieved_epsilon=7.9,
        dp_delta=1e-5,
    )
    assert a.achieved_epsilon == pytest.approx(7.9)
    assert a.dp_delta == pytest.approx(1e-5)
