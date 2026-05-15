"""Pydantic v2 models for the QLoRA training pipeline (S-064).

``TrainConfig`` carries the user-tunable knobs (epochs, learning rate, LoRA
rank/alpha/dropout, batch size) plus the ``completion_only_loss`` privacy
safeguard. ``LoRAAdapter`` is the immutable result returned by
``QLoRATrainer.train``. Both are frozen with ``extra="forbid"`` so an unknown
``[train]`` TOML key fails fast with a clear validation error — mirroring the
rest of :mod:`dbsprout.config`.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - runtime use by Pydantic field annotations

from pydantic import BaseModel, ConfigDict, Field

# Defaults match the story: rank=16, alpha=32, dropout=0.05, base model is the
# same Qwen2.5-1.5B used by the embedded inference provider so a fine-tuned
# adapter hot-swaps onto it (S-067).
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class TrainConfig(BaseModel):
    """User-supplied configuration for one ``QLoRATrainer.train`` call.

    Plumbed from the ``[train]`` section of ``dbsprout.toml`` (see
    :class:`dbsprout.config.models.DBSproutConfig`). ``completion_only_loss``
    defaults to ``True`` as a privacy safeguard — the trainer masks the prompt
    so the model is never trained to reproduce it verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    batch_size: int = Field(default=2, ge=1)
    completion_only_loss: bool = True
    base_model: str = _DEFAULT_BASE_MODEL


class LoRAAdapter(BaseModel):
    """Immutable result of a completed QLoRA training run.

    ``adapter_path`` points at the on-disk PEFT adapter directory under
    ``.dbsprout/models/adapters/``. ``final_loss`` is ``None`` when the backend
    reported no loss history (e.g. zero training steps).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter_path: Path
    base_model: str
    epochs: int = Field(ge=0)
    train_samples: int = Field(ge=0)
    final_loss: float | None
    duration_seconds: float = Field(ge=0)
