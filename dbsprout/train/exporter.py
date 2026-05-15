"""GGUF export: merge LoRA adapter -> safetensors -> GGUF Q4_K_M (S-066).

Takes a :class:`~dbsprout.train.config.LoRAAdapter` produced by the QLoRA
trainer (S-064, Unsloth/CUDA PEFT layout) or the MLX trainer (S-065) and
exports a single quantized ``.gguf`` file under ``.dbsprout/models/custom/``
that the embedded ``llama-cpp-python`` inference provider (S-025) can load.

Pipeline (three individually-testable seams):

1. ``_detect_adapter_format`` — filesystem-only sniff: PEFT (Unsloth/CUDA) vs
   MLX adapter layout.
2. ``_merge_adapter`` — load the base model + adapter and merge adapter weights
   into the base, writing a merged HF model directory whose ``.safetensors``
   shards are the documented *intermediate* artefact.
3. ``_quantize_to_gguf`` — convert the merged directory to a single
   Q4_K_M-quantized ``.gguf`` via the llama.cpp converter.

An optional ``_verify_gguf`` seam load-checks the produced file.

All heavy/optional dependencies (``torch``, ``transformers``, ``peft``,
``mlx_lm``, the llama.cpp ``convert_hf_to_gguf`` converter, ``llama_cpp``) are
imported **lazily inside the backend helpers** — never at module import time —
so the ``<500 ms`` CLI startup budget holds and ``import dbsprout`` (and
dev-only CI collection) works without ``dbsprout[train-cuda]`` /
``dbsprout[train-mlx]`` / ``dbsprout[llm]`` installed. This mirrors the
lazy-import contract in :class:`dbsprout.train.trainer.QLoRATrainer`.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.train.config import LoRAAdapter

logger = logging.getLogger(__name__)

# Q4_K_M: the recommended quality/size tradeoff for a 1.5B model (~1 GB).
_QUANT_TYPE = "Q4_K_M"
_DEFAULT_OUTPUT_DIR = Path(".dbsprout/models/custom")

_HINT_CUDA = (
    "PEFT adapter merge requires PyTorch + Transformers + PEFT. Install with "
    "'pip install dbsprout[train-cuda]' (Unsloth/PyTorch, ~2 GB)."
)
_HINT_MLX = (
    "MLX adapter merge requires the MLX runtime. Install with "
    "'pip install dbsprout[train-mlx]' (Apple Silicon only)."
)
_HINT_LLM = (
    "GGUF conversion/quantization requires llama.cpp tooling. Install with "
    "'pip install dbsprout[llm]' (installs llama-cpp-python)."
)

AdapterFormat = Literal["peft", "mlx"]


class ExportResult(BaseModel):
    """Immutable, typed result of one :meth:`Exporter.to_gguf_result` call.

    ``gguf_path`` is the final quantized model on disk under
    ``.dbsprout/models/custom/``. ``size_bytes`` is its ``stat`` size.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gguf_path: Path
    size_bytes: int = Field(ge=0)
    quant_type: str
    source_format: str
    base_model: str


def _detect_adapter_format(adapter_path: Path) -> AdapterFormat:
    """Sniff the on-disk adapter layout (filesystem only, no heavy imports).

    - PEFT/Unsloth: an ``adapter_config.json`` written by ``peft.save_pretrained``.
    - MLX: an ``adapters.safetensors`` (mlx-lm fuse input) with no PEFT config.
    """
    if (adapter_path / "adapter_config.json").is_file():
        return "peft"
    if (adapter_path / "adapters.safetensors").is_file():
        return "mlx"
    raise ValueError(
        f"unrecognized adapter format at {adapter_path}: expected a PEFT "
        "'adapter_config.json' (Unsloth/CUDA) or an MLX 'adapters.safetensors'."
    )


def _load_merge_deps(fmt: AdapterFormat) -> dict[str, Any]:
    """Lazily import the merge backend for *fmt*; raise a clear hint if missing."""
    if fmt == "mlx":
        try:
            from mlx_lm.fuse import fuse_model  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(_HINT_MLX) from exc
        return {"fuse_model": fuse_model}
    try:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(_HINT_CUDA) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _load_convert() -> Callable[..., Any]:
    """Lazily import the llama.cpp HF->GGUF converter; raise a hint if missing."""
    try:
        from llama_cpp import convert_hf_to_gguf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(_HINT_LLM) from exc
    converter: Callable[..., Any] = convert_hf_to_gguf
    return converter


def _load_verifier() -> Callable[[Path], Any]:
    """Lazily build the GGUF load-verify callable used by S-025's provider.

    A successful ``llama_cpp.Llama`` construction is the canonical "this GGUF
    loads under the embedded inference provider" smoke check. The import is
    deferred so the optional ``[llm]`` extra is not required at module import.
    """
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(_HINT_LLM) from exc

    def _verify(path: Path) -> None:
        # Constructing the model is enough to prove the GGUF is loadable by the
        # embedded provider; n_ctx kept tiny to make the smoke load cheap.
        Llama(model_path=str(path), n_ctx=8, verbose=False)

    return _verify


def _merge_adapter(
    adapter: LoRAAdapter,
    base_model: Path,
    work_dir: Path,
    *,
    fmt: AdapterFormat,
) -> Path:
    """Merge adapter weights into the base model; return the merged HF dir.

    The returned directory holds ``.safetensors`` shards — the documented
    intermediate artefact. Heavy deps are imported via :func:`_load_merge_deps`.
    """
    deps = _load_merge_deps(fmt)
    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "mlx":
        deps["fuse_model"](
            model=str(base_model),
            adapter_path=str(adapter.adapter_path),
            save_path=str(merged_dir),
        )
        return merged_dir

    base = deps["AutoModelForCausalLM"].from_pretrained(
        str(base_model), torch_dtype=deps["torch"].float16
    )
    peft_loaded = deps["PeftModel"].from_pretrained(base, str(adapter.adapter_path))
    merged = peft_loaded.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    deps["AutoTokenizer"].from_pretrained(str(base_model)).save_pretrained(str(merged_dir))
    return merged_dir


def _quantize_to_gguf(merged_dir: Path, out_path: Path) -> Path:
    """Convert *merged_dir* to a single Q4_K_M-quantized GGUF at *out_path*."""
    convert = _load_convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(
        model=str(merged_dir),
        outfile=str(out_path),
        outtype=_QUANT_TYPE,
    )
    return out_path


def _verify_gguf(path: Path) -> None:
    """Load-verify the produced GGUF; re-raise failures as a clear RuntimeError."""
    verifier = _load_verifier()
    try:
        verifier(path)
    except Exception as exc:
        raise RuntimeError(f"exported GGUF failed verification ({path}): {exc}") from exc


class Exporter:
    """Export a fine-tuned LoRA adapter to a quantized GGUF model.

    Accepts both Unsloth/CUDA PEFT adapters (S-064) and MLX adapters (S-065),
    auto-detected from the on-disk layout. The result lands under
    ``.dbsprout/models/custom/`` ready for the embedded inference provider.
    """

    def to_gguf(
        self,
        adapter: LoRAAdapter,
        base_model: Path,
        *,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        quiet: bool = False,
        verify: bool = True,
    ) -> Path:
        """Run merge -> safetensors -> GGUF Q4_K_M and return the GGUF path.

        Raises
        ------
        FileNotFoundError
            *base_model* or the adapter directory does not exist.
        ValueError
            The adapter directory has no recognizable PEFT/MLX layout.
        RuntimeError
            A required optional extra is missing, or load-verify failed.
        """
        return self._run(adapter, base_model, output_dir, quiet=quiet, verify=verify)[0]

    def to_gguf_result(
        self,
        adapter: LoRAAdapter,
        base_model: Path,
        *,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        quiet: bool = False,
        verify: bool = True,
    ) -> ExportResult:
        """Like :meth:`to_gguf` but return a typed :class:`ExportResult`."""
        path, fmt = self._run(adapter, base_model, output_dir, quiet=quiet, verify=verify)
        return ExportResult(
            gguf_path=path,
            size_bytes=path.stat().st_size,
            quant_type=_QUANT_TYPE,
            source_format=fmt,
            base_model=adapter.base_model,
        )

    def _run(
        self,
        adapter: LoRAAdapter,
        base_model: Path,
        output_dir: Path,
        *,
        quiet: bool,
        verify: bool,
    ) -> tuple[Path, AdapterFormat]:
        if not base_model.exists():
            raise FileNotFoundError(f"base model not found: {base_model}")
        if not adapter.adapter_path.exists():
            raise FileNotFoundError(f"adapter directory not found: {adapter.adapter_path}")
        fmt = _detect_adapter_format(adapter.adapter_path)
        out_path = output_dir / f"{adapter.adapter_path.name}-{_QUANT_TYPE}.gguf"

        with (
            tempfile.TemporaryDirectory(prefix="dbsprout-gguf-") as tmp,
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                disable=quiet,
            ) as progress,
        ):
            work = Path(tmp)
            task = progress.add_task("Merging adapter", total=3)
            merged_dir = _merge_adapter(adapter, base_model, work, fmt=fmt)
            progress.update(task, advance=1, description="Quantizing (Q4_K_M)")
            _quantize_to_gguf(merged_dir, out_path)
            progress.update(task, advance=1, description="Verifying")
            if verify:
                _verify_gguf(out_path)
            progress.update(task, advance=1, description="Done")

        logger.info(
            "Exported GGUF: adapter=%s format=%s quant=%s -> %s",
            adapter.adapter_path,
            fmt,
            _QUANT_TYPE,
            out_path,
        )
        return out_path, fmt
