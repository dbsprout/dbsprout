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

The training loop is wired against the **verified-real** ``mlx-lm`` LoRA API
(``mlx_lm.utils.load`` → ``(model, tokenizer)``; ``mlx_lm.utils.save_config``;
``mlx_lm.tuner.datasets.TextDataset``/``CacheDataset``;
``mlx_lm.tuner.utils.linear_to_lora_layers``;
``mlx_lm.tuner.trainer.TrainingArgs`` + ``train`` + ``TrainingCallback``; an
``mlx.optimizers.Adam`` optimizer; ``mlx.core.random.seed``) resolved behind
the single :func:`_load_mlx_lm_symbols` seam. Every symbol was verified
against the pinned ``mlx-lm>=0.20`` upstream source (``ml-explore/mlx-lm``):
there is no ``run_lora_training``, no ``completion_only_loss`` ``train()``
keyword, and ``adapter_file`` is a ``TrainingArgs`` field. If those real
symbols are unavailable at runtime a clear actionable error is raised — the
trainer never *silently* succeeds.

.. note::
   Real Apple-Silicon execution is **hardware-validation-pending** (like the
   perf ACs): mlx-lm cannot install on Linux/CI, so end-to-end training is
   exercised only by the Apple-Silicon ``@pytest.mark.integration`` test that
   asserts the real symbol surface. Unit tests mock the *correct* seam.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dbsprout.train.config import LoRAAdapter
from dbsprout.train.trainer import _cuda_available

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.train.config import TrainConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class _Trainer(Protocol):
    """Structural contract shared by :class:`QLoRATrainer` and :class:`MLXTrainer`.

    ``select_trainer()`` returns a ``_Trainer``; both concrete trainers satisfy
    it structurally, so the CLI no longer needs an ``# type: ignore`` on the
    dispatched ``trainer.train(...)`` call (S-068 review #16/#8).
    """

    def train(
        self,
        *,
        corpus_path: Path,
        config: TrainConfig,
        output_dir: Path,
        schema_hash: str | None = ...,
        quiet: bool = ...,
    ) -> LoRAAdapter: ...


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
_MLX_DP_UNSUPPORTED = (
    "DP-SGD (Opacus) is not supported on the MLX (Apple Silicon) backend -- "
    "Opacus is PyTorch/CUDA-only. Use the CUDA QLoRA trainer "
    "('pip install dbsprout[train-cuda] dbsprout[train-dp]') for a "
    "differential-privacy guarantee, or set [train.privacy] dp_sgd = false."
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


def select_trainer() -> _Trainer:
    """Pick the right trainer for this host.

    Returns a :class:`~dbsprout.train.trainer.QLoRATrainer` on CUDA hosts and
    an :class:`MLXTrainer` on Apple Silicon. Raises :class:`RuntimeError` with
    an install hint when neither backend is usable. The :class:`_Trainer`
    return type (both trainers satisfy it structurally) lets the CLI call
    ``trainer.train(...)`` without an ``# type: ignore``.
    """
    if _cuda_available():
        from dbsprout.train.trainer import QLoRATrainer  # noqa: PLC0415

        return QLoRATrainer()
    if _mlx_available():
        return MLXTrainer()
    raise RuntimeError(_MLX_HINT)


def _load_mlx_lm_symbols() -> dict[str, Any]:
    """Resolve the **verified-real** mlx-lm LoRA training symbols (mock seam).

    Every symbol is a genuine public entrypoint of the pinned ``mlx-lm>=0.20``
    package (verified against the ``ml-explore/mlx-lm`` source — ``lora.py``,
    ``tuner/trainer.py``, ``tuner/utils.py``, ``tuner/datasets.py``,
    ``tuner/callbacks.py``). There is **no** ``run_lora_training`` convenience
    API, no ``completion_only_loss`` parameter, and ``adapter_file`` is a
    :class:`mlx_lm.tuner.trainer.TrainingArgs` *field* — never a ``train()``
    keyword:

    * ``load``                  — ``mlx_lm.utils.load`` → ``(model, tokenizer)``
    * ``save_config``           — ``mlx_lm.utils.save_config`` (writes JSON)
    * ``TextDataset``           — ``mlx_lm.tuner.datasets.TextDataset``
      (``__init__(data: list[dict], tokenizer, text_key="text")``)
    * ``CacheDataset``          — ``mlx_lm.tuner.datasets.CacheDataset``
    * ``linear_to_lora_layers`` — ``mlx_lm.tuner.utils.linear_to_lora_layers``
      (``(model, num_layers, config, use_dora=False)``)
    * ``TrainingArgs``/``train``— ``mlx_lm.tuner.trainer``
      (``train(model, optimizer, train_dataset, val_dataset=None, args=,
      loss=, iterate_batches=, training_callback=)``)
    * ``TrainingCallback``      — ``mlx_lm.tuner.trainer.TrainingCallback``
      (base class: ``on_train_loss_report(dict)`` / ``on_val_loss_report``)
    * ``make_optimizer``        — an ``mlx.optimizers.Adam`` factory
    * ``seed``                  — ``mlx.core.random.seed`` (best-effort RNG seed)

    Raises a clear, actionable :class:`RuntimeError` (never a silent success)
    if mlx-lm is missing or its real API has moved. This is the *only* place
    the optional ``mlx_lm``/``mlx`` packages are imported, so unit tests mock
    exactly these correct symbols and a non-existent API cannot hide.
    """
    try:
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
    except ImportError as exc:
        raise RuntimeError(_MLX_HINT) from exc

    def make_optimizer(*, learning_rate: float) -> Any:
        return optim.Adam(learning_rate=learning_rate)

    return {
        "load": load,
        "save_config": save_config,
        "TextDataset": TextDataset,
        "CacheDataset": CacheDataset,
        "linear_to_lora_layers": linear_to_lora_layers,
        "TrainingArgs": TrainingArgs,
        "train": train,
        "TrainingCallback": TrainingCallback,
        "make_optimizer": make_optimizer,
        "seed": mx.random.seed,
    }


# Epochs are mapped to the real ``mlx_lm.tuner.trainer.TrainingArgs.iters``
# (an iteration count, not an epoch count) via a fixed steps-per-epoch
# heuristic; the true iteration count depends on the dataset/batch size at
# runtime. Apple-Silicon validation is pending, so this is a documented
# approximation, not a tuned value.
_ITERS_PER_EPOCH = 100

# Real mlx-lm has NO ``completion_only_loss`` parameter (verified against the
# pinned upstream source). For parity with the CUDA/Unsloth path we keep the
# documented intent: the GReaT corpus (S-063) is a single ``text`` field per
# row with no prompt/completion split, so completion-only loss is a structural
# no-op *by construction* — the corpus format itself is the privacy safeguard.
# We therefore record the intent (constant + log line) but never pass a
# non-existent kwarg to the real ``train()``.
_COMPLETION_ONLY_LOSS = True

# Real mlx-lm ``TrainingArgs.max_seq_length`` default (kept explicit so the
# GReaT serialized rows are not silently truncated by a smaller default).
_MAX_SEQ_LENGTH = 2048


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
            CUDA host (use the CUDA trainer), not Apple Silicon,
            ``mlx-lm`` is not installed, or DP-SGD was requested (Opacus is
            PyTorch/CUDA-only — use the CUDA trainer) — each with a hint.
        FileNotFoundError
            *corpus_path* does not exist (run ``dbsprout train serialize``).
        ValueError
            *corpus_path* exists but is empty.
        """
        start = time.perf_counter()
        if config.privacy.dp_sgd:
            raise RuntimeError(_MLX_DP_UNSUPPORTED)
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
        """Run the verified-real mlx-lm LoRA training loop; return final loss.

        Wired against the genuine ``mlx_lm.tuner`` API via
        :func:`_load_mlx_lm_symbols`, every symbol verified against the pinned
        ``mlx-lm>=0.20`` upstream source:

        1. ``load(base_model)`` → ``(model, tokenizer)``
        2. parse the GReaT JSONL ourselves → ``TextDataset(rows, tokenizer)``
           wrapped in ``CacheDataset`` (we deliberately bypass the real
           ``load_dataset(args, tokenizer)`` — it needs a fake argparse
           namespace plus a directory of ``train/valid/test.jsonl``; the
           public dataset classes are the cleaner real entrypoint here)
        3. ``model.freeze()`` then
           ``linear_to_lora_layers(model, num_layers, lora_cfg)`` over all
           transformer blocks (``num_layers = len(model.layers)``); knobs
           mirror the CUDA QLoRA path (``scale = lora_alpha / lora_rank``)
        4. build the real ``TrainingArgs`` dataclass — ``adapter_file`` is a
           **field of TrainingArgs**, never a ``train()`` keyword
        5. ``train(model=, optimizer=, train_dataset=, args=,
           training_callback=)`` — keyword args matching the verified
           signature; a :class:`mlx_lm.tuner.trainer.TrainingCallback`
           subclass drives a determinate progress bar (loss + ETA)
        6. write a minimal real ``adapter_config.json`` next to
           ``adapters.safetensors`` for the S-066 GGUF exporter

        Privacy parity with the CUDA path (S-064 review #7): the GReaT corpus
        is a single ``text`` field per row with NO prompt/completion split, so
        completion-only loss is a structural no-op *by construction*. The
        intent is recorded (``_COMPLETION_ONLY_LOSS`` + log line); real mlx-lm
        has no such parameter so nothing fabricated is passed to ``train()``.

        Real Apple-Silicon execution is hardware-validation-pending: mlx-lm
        cannot install on Linux/CI, so a missing/changed real API raises a
        clear error rather than succeeding silently.
        """
        syms = _load_mlx_lm_symbols()
        loss_box: dict[str, float | None] = {"loss": None}

        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_dir / "adapters.safetensors"
        rows = self._read_corpus_rows(corpus_path)
        iters = config.epochs * _ITERS_PER_EPOCH
        logger.info(
            "MLX completion-only-loss intent=%s (structural no-op on the "
            "single-text GReaT corpus; parity with the CUDA path)",
            _COMPLETION_ONLY_LOSS,
        )

        # Real TrainingCallback subclass: drives the determinate progress bar
        # and records the latest training loss for the returned LoRAAdapter.
        callback_factory = self._make_progress_callback(syms["TrainingCallback"], loss_box)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            disable=quiet,
        ) as progress:
            task = progress.add_task("Loading base model", total=iters)
            model, tokenizer = syms["load"](config.base_model)

            progress.update(task, description="Applying LoRA layers")
            num_layers, lora_cfg = self._apply_lora(syms, model, config)

            progress.update(task, description="Preparing dataset")
            train_dataset = syms["CacheDataset"](syms["TextDataset"](rows, tokenizer))
            optimizer = syms["make_optimizer"](learning_rate=config.learning_rate)
            args = syms["TrainingArgs"](
                batch_size=config.batch_size,
                iters=iters,
                adapter_file=str(adapter_file),
                max_seq_length=_MAX_SEQ_LENGTH,
            )

            progress.update(task, description="Training (MLX)")
            syms["train"](
                model=model,
                optimizer=optimizer,
                train_dataset=train_dataset,
                args=args,
                training_callback=callback_factory(progress, task, iters),
            )

            # Minimal real adapter config for the S-066 GGUF exporter.
            syms["save_config"](
                {
                    "fine_tune_type": "lora",
                    "num_layers": num_layers,
                    "lora_parameters": lora_cfg,
                    "base_model": config.base_model,
                },
                adapter_dir / "adapter_config.json",
            )
            progress.update(task, description="Done", completed=iters)

        return loss_box["loss"]

    @staticmethod
    def _read_corpus_rows(corpus_path: Path) -> list[dict[str, Any]]:
        """Parse the GReaT JSONL corpus into a list of ``{"text": ...}`` dicts.

        The corpus is produced by the S-063 serializer (a trusted internal
        artifact); blank lines are skipped so a trailing newline is harmless.
        Constructed directly for ``mlx_lm.tuner.datasets.TextDataset`` —
        deliberately bypassing the real ``load_dataset(args, tokenizer)``
        which needs a fake argparse namespace plus a directory of
        ``train/valid/test.jsonl``.
        """
        return [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _apply_lora(
        syms: dict[str, Any],
        model: Any,
        config: TrainConfig,
    ) -> tuple[int, dict[str, float]]:
        """Seed, freeze, and LoRA-ify all transformer blocks of *model*.

        Mirrors the real ``mlx_lm/lora.py`` setup: ``mx.random.seed`` →
        ``model.freeze()`` → ``linear_to_lora_layers(model, num_layers,
        cfg)`` over every block (``num_layers = len(model.layers)``). The
        ``seed``/``freeze``/``layers`` accesses are guarded so a bare model
        stub never crashes. The LoRA knobs mirror the CUDA QLoRA path
        (``scale = lora_alpha / lora_rank``). Returns ``(num_layers,
        lora_cfg)`` for the adapter-config write.
        """
        # Deterministic LoRA init (mirrors real lora.py ``mx.random.seed``).
        seed = syms.get("seed")
        if callable(seed):
            seed(0)
        freeze = getattr(model, "freeze", None)
        if callable(freeze):
            freeze()
        num_layers = len(getattr(model, "layers", []))
        lora_cfg = {
            "rank": config.lora_rank,
            "scale": config.lora_alpha / config.lora_rank,
            "dropout": config.lora_dropout,
        }
        syms["linear_to_lora_layers"](model, num_layers, lora_cfg)
        return num_layers, lora_cfg

    @staticmethod
    def _make_progress_callback(
        training_callback_cls: type,
        loss_box: dict[str, float | None],
    ) -> Any:
        """Build a real ``TrainingCallback`` subclass bound to a progress bar.

        The real ``mlx_lm.tuner.trainer.train`` reports progress by calling
        ``training_callback.on_train_loss_report(info)`` where ``info`` carries
        ``iteration`` and ``train_loss``. We subclass the verified-real base
        class (resolved through the seam) so the bar advances deterministically
        (with ETA) and the final loss is captured for the ``LoRAAdapter``.
        """

        def _factory(progress: Progress, task: Any, total: int) -> Any:
            class _ProgressCallback(training_callback_cls):  # type: ignore[misc]
                def on_train_loss_report(self, info: dict[str, Any]) -> None:
                    loss = info.get("train_loss")
                    if loss is not None:
                        loss_box["loss"] = float(loss)
                    completed = info.get("iteration")
                    if isinstance(completed, int):
                        progress.update(
                            task,
                            completed=min(completed, total),
                            description=(
                                f"Training (MLX) loss={loss:.4f}"
                                if loss is not None
                                else "Training (MLX)"
                            ),
                        )

                def on_val_loss_report(self, info: dict[str, Any]) -> None:  # noqa: ARG002
                    return None

            return _ProgressCallback()

        return _factory
