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

Privacy safeguard: the **structural** guarantee is the GReaT corpus format
itself (S-063) — every training row is a single ``text`` field with NO
prompt/completion split, so by construction there is no prompt for the model
to memorize verbatim. ``completion_only_loss`` is still forced ``True``
regardless of ``TrainConfig.completion_only_loss``, but for this single-text
corpus it is a no-op kept only to future-proof any later prompt/completion
corpus; it is not the load-bearing safeguard.
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
    from dbsprout.train.privacy import TrainPrivacyConfig

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


_DP_INSTALL_HINT = (
    "DP-SGD requested but Opacus is not installed. Install it with "
    "'pip install dbsprout[train-dp]' (Opacus, PyTorch/CUDA only) or set "
    "[train.privacy] dp_sgd = false."
)


def _make_private(
    *,
    privacy: TrainPrivacyConfig,
    model: object,
    optimizer: object,
    data_loader: object,
    epochs: int,
) -> tuple[object, object, object, float | None]:
    """Wrap model/optimizer/dataloader with Opacus; return the privatized
    trio plus the achieved epsilon.

    Epsilon-targeted accounting (``dp_target_epsilon`` set) uses
    ``make_private_with_epsilon`` and reports the configured target as the
    guarantee; otherwise ``make_private`` runs with the explicit
    ``dp_noise_multiplier`` and the achieved epsilon is read from the engine
    accountant (``None`` if unavailable). ``opacus`` is imported lazily; a
    missing install raises a friendly install hint, never a bare ImportError.
    """
    try:
        from opacus import PrivacyEngine  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(_DP_INSTALL_HINT) from exc

    engine = PrivacyEngine()
    if privacy.dp_target_epsilon is not None:
        new_model, new_opt, new_loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=privacy.dp_target_epsilon,
            target_delta=privacy.dp_target_delta,
            max_grad_norm=privacy.dp_max_grad_norm,
        )
        return new_model, new_opt, new_loader, privacy.dp_target_epsilon

    new_model, new_opt, new_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=privacy.dp_noise_multiplier,
        max_grad_norm=privacy.dp_max_grad_norm,
    )
    achieved: float | None
    try:
        achieved = float(engine.get_epsilon(delta=privacy.dp_target_delta))
    except (AttributeError, TypeError, ValueError):
        logger.warning("Opacus accountant unavailable; achieved epsilon=None")
        achieved = None
    return new_model, new_opt, new_loader, achieved


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

        # Validate the cheap *local* preconditions BEFORE probing for a
        # GPU/backend. On a CPU host with a missing corpus the user should get
        # the actionable corpus error, not a misleading "install CUDA" hint.
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"training corpus not found: {corpus_path}. Run 'dbsprout train serialize' first."
            )
        sample_count = sum(
            1 for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if sample_count == 0:
            raise ValueError(f"training corpus {corpus_path} is empty; nothing to train on.")

        backend = _select_backend()

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

        final_loss, achieved_epsilon = self._run_unsloth(
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
            achieved_epsilon=achieved_epsilon,
            dp_delta=config.privacy.dp_target_delta if achieved_epsilon is not None else None,
        )

    def _run_unsloth(
        self,
        *,
        corpus_path: Path,
        config: TrainConfig,
        adapter_dir: Path,
        quiet: bool,
    ) -> tuple[float | None, float | None]:
        """Run the Unsloth QLoRA training loop.

        Returns the ``(final training loss, achieved epsilon)`` pair. The
        achieved epsilon is ``None`` unless DP-SGD ran
        (``config.privacy.dp_sgd``), in which case the optimizer + dataloader
        are wrapped with Opacus :func:`_make_private` before training.

        Heavy deps are imported here so module import stays light. The
        completion-only-loss flag is forced ``True`` independently of
        ``config.completion_only_loss`` — but note it is a no-op for the GReaT
        single-text corpus (no prompt exists to memorize *by construction*);
        the corpus format is the real privacy safeguard. A non-numeric backend
        loss degrades to ``None`` rather than raising.
        """
        try:
            # All three live in the same ``dbsprout[train-cuda]`` extra. Folding
            # them into one guarded block means a *partial* install (e.g.
            # unsloth present but ``datasets``/``trl`` missing) still surfaces
            # the friendly install hint instead of a bare ImportError.
            from datasets import load_dataset  # noqa: PLC0415
            from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
            from unsloth import FastLanguageModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

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

            progress.update(task, description="Building trainer")
            sft_config = SFTConfig(
                output_dir=str(adapter_dir),
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.batch_size,
                dataset_text_field="text",
                # Privacy: forced True regardless of ``config.completion_only_loss``.
                # NOTE: for the GReaT corpus (S-063) this is a *no-op by
                # construction* — every row is a single ``text`` field with NO
                # prompt/completion split, so there is no prompt to memorize.
                # The real structural safeguard is the corpus format itself;
                # this flag is kept only to future-proof any later
                # prompt/completion corpus and is not load-bearing today.
                completion_only_loss=True,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=sft_config,
            )

            achieved_epsilon: float | None = None
            if config.privacy.dp_sgd:
                progress.update(task, description="Wrapping with Opacus DP-SGD")
                priv_model, priv_opt, priv_loader, achieved_epsilon = _make_private(
                    privacy=config.privacy,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    data_loader=trainer.get_train_dataloader(),
                    epochs=config.epochs,
                )
                # Reassign the Opacus-privatized objects onto the SFTTrainer
                # so trainer.train() steps the noised/clipped optimizer over
                # the Poisson-sampled loader. The exact attribute names on the
                # real trl/HF Trainer (and Unsloth's patched optimizer) are
                # runtime-validation-pending: CI cannot install
                # CUDA/Unsloth/Opacus, so this wiring is asserted structurally
                # in tests (same precedent as the mlx-lm
                # hardware-validation-pending integration test). A CUDA-enabled
                # run must confirm trainer.train() consumes these.
                trainer.model = priv_model
                trainer.optimizer = priv_opt
                trainer.train_dataloader = priv_loader
                logger.info(
                    "DP-SGD enabled (Opacus): achieved (epsilon=%s, delta=%s)",
                    achieved_epsilon,
                    config.privacy.dp_target_delta,
                )

            progress.update(task, description="Training")
            stats = trainer.train()
            trainer.save_model(str(adapter_dir))
            progress.update(task, description="Done", completed=1, total=1)

        loss = getattr(stats, "training_loss", None)
        if loss is None:
            return None, achieved_epsilon
        try:
            return float(loss), achieved_epsilon
        except (TypeError, ValueError):
            # A backend reporting a non-numeric loss must not crash a
            # completed run; degrade to "no loss history" instead.
            logger.warning("non-numeric training_loss %r; reporting final_loss=None", loss)
            return None, achieved_epsilon
