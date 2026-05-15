"""QLoRA fine-tuning trainer with CUDA auto-detection (S-064).

Detects an NVIDIA GPU via ``torch.cuda.is_available()`` and, when present,
fine-tunes a 4-bit QLoRA adapter on the GReaT-style JSONL corpus produced by
:mod:`dbsprout.train.serializer` (S-063) using `Unsloth
<https://github.com/unslothai/unsloth>`_ (2-4x faster than vanilla HF).

All heavy optional dependencies (``torch``, ``unsloth``, ``trl``,
``datasets``) are imported **lazily inside methods** — never at module import
time — so the ``<500 ms`` CLI startup budget is preserved and ``import
dbsprout`` works without ``dbsprout[train-cuda]`` installed. This mirrors the
lazy-import contract in :class:`dbsprout.spec.providers.embedded.EmbeddedProvider`.

Privacy safeguard: the trainer **always** forces completion-only loss so the
model is trained on the serialized row text only and never to reproduce a
prompt verbatim, regardless of the user's ``TrainConfig.completion_only_loss``
value.
"""

from __future__ import annotations

import logging
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

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.train.config import TrainConfig

logger = logging.getLogger(__name__)

# 4-bit QLoRA: the standard quantization for single-GPU 1.5B fine-tuning.
_LOAD_IN_4BIT = True
_MAX_SEQ_LENGTH = 2048
_INSTALL_HINT = (
    "CUDA GPU not detected. QLoRA training requires an NVIDIA GPU and "
    "'pip install dbsprout[train-cuda]' (installs Unsloth + PyTorch, ~2 GB). "
    "On Apple Silicon use the MLX trainer instead (dbsprout[train-mlx])."
)


def _cuda_available() -> bool:
    """Return ``True`` iff a CUDA GPU is usable.

    ``torch`` is imported lazily; a missing/broken install is treated as
    "no CUDA" rather than raising, so the helper is safe to call on any host.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _select_backend() -> str:
    """Return the training backend name, or raise if no GPU is available.

    Today only the ``unsloth`` CUDA backend exists; the Apple-Silicon MLX
    backend is S-065. A clear :class:`RuntimeError` tells the user how to
    install the right extra.
    """
    if _cuda_available():
        return "unsloth"
    raise RuntimeError(_INSTALL_HINT)


class QLoRATrainer:
    """Built-in CUDA QLoRA trainer.

    Reads the JSONL corpus from S-063, fine-tunes a LoRA adapter on the base
    model with Unsloth, and writes the adapter under
    ``<output_dir>/<schema_hash or 'default'>/``.
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
        """Fine-tune a QLoRA adapter and return its :class:`LoRAAdapter`.

        Raises
        ------
        RuntimeError
            No CUDA GPU available (with an install hint).
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
            "Starting QLoRA training: backend=%s base=%s samples=%d epochs=%d rank=%d -> %s",
            backend,
            config.base_model,
            sample_count,
            config.epochs,
            config.lora_rank,
            adapter_dir,
        )

        final_loss = self._run_unsloth(
            corpus_path=corpus_path,
            config=config,
            adapter_dir=adapter_dir,
            quiet=quiet,
        )

        duration = time.perf_counter() - start
        logger.info(
            "QLoRA training complete in %.1fs (final_loss=%s) -> %s",
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

    def _run_unsloth(
        self,
        *,
        corpus_path: Path,
        config: TrainConfig,
        adapter_dir: Path,
        quiet: bool,
    ) -> float | None:
        """Run the Unsloth QLoRA training loop; return the final training loss.

        Heavy deps are imported here so module import stays light. The
        completion-only-loss privacy safeguard is forced ``True`` into the
        trainer config independently of ``config.completion_only_loss``.
        """
        try:
            from unsloth import FastLanguageModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via mock injection
            raise RuntimeError(_INSTALL_HINT) from exc
        from datasets import load_dataset  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            disable=quiet,
        ) as progress:
            task = progress.add_task("Loading base model", total=None)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.base_model,
                max_seq_length=_MAX_SEQ_LENGTH,
                load_in_4bit=_LOAD_IN_4BIT,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

            progress.update(task, description="Loading corpus")
            # nosec B615 - "json" is the builtin local-file loader reading our
            # own offline corpus_path; no Hugging Face Hub download happens, so
            # revision pinning is not applicable.
            dataset = load_dataset(  # nosec B615
                "json", data_files=str(corpus_path), split="train"
            )

            progress.update(task, description="Training")
            sft_config = SFTConfig(
                output_dir=str(adapter_dir),
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                dataset_text_field="text",
                # Privacy safeguard: always True regardless of user config so
                # the model never memorizes prompts (no prompt-only rows here,
                # but this future-proofs prompt/completion corpora).
                completion_only_loss=True,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=sft_config,
            )
            stats = trainer.train()
            trainer.save_model(str(adapter_dir))
            progress.update(task, description="Done", completed=1, total=1)

        loss = getattr(stats, "training_loss", None)
        return float(loss) if loss is not None else None
