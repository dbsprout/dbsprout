"""MLX LoRA fine-tuning trainer for Apple Silicon (S-065).

The CUDA path (:class:`dbsprout.train.trainer.QLoRATrainer`, S-064) covers
NVIDIA GPUs via Unsloth. This module adds the Apple-Silicon path: when no CUDA
GPU is present and ``platform.system() == "Darwin"``, fine-tune a LoRA adapter
on the GReaT-style JSONL corpus (S-063) using `MLX-LM
<https://github.com/ml-explore/mlx-lm>`_ on Apple's unified-memory runtime.

Both trainers share the immutable :class:`~dbsprout.train.config.TrainConfig`
and :class:`~dbsprout.train.config.LoRAAdapter` contract, so the rest of the
pipeline (CLI, GGUF export) is backend-agnostic. No shared ``Trainer`` base
class is introduced — the contract *is* the config/result pair, mirroring how
``QLoRATrainer`` is self-contained.

All heavy optional dependencies (``mlx``, ``mlx_lm``) are imported **lazily
inside methods** — never at module import time — so ``import dbsprout`` works
without ``dbsprout[train-mlx]`` and the ``<500 ms`` CLI startup budget holds.
This mirrors the lazy-import contract in
:class:`dbsprout.train.trainer.QLoRATrainer`.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from dbsprout.train.config import LoRAAdapter
from dbsprout.train.trainer import _cuda_available

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.train.config import TrainConfig

logger = logging.getLogger(__name__)

# Apple Silicon LoRA on unified memory: MLX-LM defaults are sane; we only
# thread the user-tunable knobs through (parity with the CUDA QLoRA path).
_CUDA_HINT = (
    "A CUDA GPU was detected. Use the CUDA QLoRA trainer "
    "('pip install dbsprout[train-cuda]') instead of the MLX trainer."
)
_MLX_HINT = (
    "MLX training requires Apple Silicon (macOS/arm64) and "
    "'pip install dbsprout[train-mlx]' (installs MLX + MLX-LM, ~100 MB). "
    "On an NVIDIA GPU use the CUDA trainer instead (dbsprout[train-cuda])."
)


def _mlx_available() -> bool:
    """Return ``True`` iff the host is Apple Silicon (Darwin).

    Only the *platform* is probed here — the ``mlx_lm`` package itself is
    imported lazily at training time so this helper is safe to call on any
    host without the optional extra installed.
    """
    return platform.system() == "Darwin"


def _select_backend() -> str:
    """Return the training backend name, or raise with an install hint.

    A CUDA host is redirected to :class:`QLoRATrainer` (the MLX path is for
    Apple Silicon only). A non-Darwin host without CUDA cannot train at all.
    """
    if _cuda_available():
        raise RuntimeError(_CUDA_HINT)
    if _mlx_available():
        return "mlx"
    raise RuntimeError(_MLX_HINT)


def select_trainer() -> object:
    """Pick the right trainer for this host.

    Returns a :class:`~dbsprout.train.trainer.QLoRATrainer` on CUDA hosts and
    an :class:`MLXTrainer` on Apple Silicon. Raises :class:`RuntimeError` with
    an install hint when neither backend is usable. Both returned objects
    expose the same ``train(...)`` signature and return a ``LoRAAdapter``.
    """
    if _cuda_available():
        from dbsprout.train.trainer import QLoRATrainer  # noqa: PLC0415

        return QLoRATrainer()
    if _mlx_available():
        return MLXTrainer()
    raise RuntimeError(_MLX_HINT)


class MLXTrainer:
    """Apple-Silicon LoRA trainer (MLX-LM backend).

    Reads the JSONL corpus from S-063, fine-tunes a LoRA adapter on the base
    model with MLX-LM, and writes the adapter under
    ``<output_dir>/<schema_hash or 'default'>/`` — the same on-disk layout as
    :class:`~dbsprout.train.trainer.QLoRATrainer`, so the GGUF export pipeline
    is backend-agnostic.
    """

    def train(
        self,
        *,
        corpus_path: Path,
        config: TrainConfig,
        output_dir: Path,
        schema_hash: str | None = None,
        quiet: bool = False,
    ) -> LoRAAdapter:
        """Fine-tune a LoRA adapter with MLX-LM and return its ``LoRAAdapter``.

        Raises
        ------
        RuntimeError
            CUDA host (use the CUDA trainer), not Apple Silicon, or
            ``mlx-lm`` is not installed — each with an install hint.
        FileNotFoundError
            *corpus_path* does not exist (run ``dbsprout train serialize``).
        ValueError
            *corpus_path* exists but is empty.
        """
        start = time.perf_counter()
        backend = _select_backend()

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"training corpus not found: {corpus_path}. Run 'dbsprout train serialize' first."
            )
        sample_count = sum(
            1 for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if sample_count == 0:
            raise ValueError(f"training corpus {corpus_path} is empty; nothing to train on.")

        adapter_dir = output_dir / (schema_hash or "default")
        adapter_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting MLX training: backend=%s base=%s samples=%d epochs=%d rank=%d -> %s",
            backend,
            config.base_model,
            sample_count,
            config.epochs,
            config.lora_rank,
            adapter_dir,
        )

        final_loss = self._run_mlx(
            corpus_path=corpus_path,
            config=config,
            adapter_dir=adapter_dir,
            quiet=quiet,
        )

        duration = time.perf_counter() - start
        logger.info(
            "MLX training complete in %.1fs (final_loss=%s) -> %s",
            duration,
            final_loss,
            adapter_dir,
        )
        return LoRAAdapter(
            adapter_path=adapter_dir,
            base_model=config.base_model,
            epochs=config.epochs,
            train_samples=sample_count,
            final_loss=final_loss,
            duration_seconds=duration,
        )

    def _run_mlx(
        self,
        *,
        corpus_path: Path,
        config: TrainConfig,
        adapter_dir: Path,
        quiet: bool,
    ) -> float | None:
        """Run the MLX-LM LoRA training loop; return the final training loss.

        ``mlx_lm`` is imported here so module import stays light. The LoRA
        knobs mirror the CUDA QLoRA path (rank/alpha/dropout/epochs/lr/batch).
        Apple's unified memory tolerates the same or larger batch sizes than
        equivalent-VRAM GPUs, so ``config.batch_size`` is passed through as-is.
        """
        try:
            import mlx_lm  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(_MLX_HINT) from exc

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            disable=quiet,
        ) as progress:
            task = progress.add_task("Loading base model", total=None)
            progress.update(task, description="Training (MLX)")
            loss = mlx_lm.run_lora_training(
                base_model=config.base_model,
                corpus_path=corpus_path,
                adapter_dir=adapter_dir,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                batch_size=config.batch_size,
            )
            progress.update(task, description="Done", completed=1, total=1)

        return float(loss) if loss is not None else None
