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
   Q4_K_M-quantized ``.gguf`` by invoking llama.cpp's ``convert_hf_to_gguf.py``
   then ``llama-quantize`` as subprocesses (the real, supported mechanism —
   ``llama-cpp-python`` does **not** export a ``convert_hf_to_gguf`` symbol).

An optional ``_verify_gguf`` seam load-checks the produced file.

The quantization tool paths are resolved via :func:`_resolve_converter` /
:func:`_resolve_quantizer`: an env override (``DBSPROUT_LLAMA_CONVERT`` /
``DBSPROUT_LLAMA_QUANTIZE``) → known locations / ``PATH``, with a clear
actionable error if not found. Subprocesses are launched with explicit
list-args, ``check=True`` and ``capture_output=True`` — never ``shell=True``.

Atomicity: merge → quantize → verify all happen inside a
``TemporaryDirectory``; the final ``.gguf`` is moved into
``.dbsprout/models/custom/`` **only after verification succeeds**, so a failed
export never leaves a partial/corrupt model behind (also closes the prior
TOCTOU window).

All heavy/optional dependencies (``torch``, ``transformers``, ``peft``,
``mlx_lm``, ``llama_cpp`` for the verify smoke-load) are imported **lazily
inside the backend helpers** — never at module import time — so the
``<500 ms`` CLI startup budget holds and ``import dbsprout`` (and dev-only CI
collection) works without ``dbsprout[train-cuda]`` / ``dbsprout[train-mlx]`` /
``dbsprout[llm]`` installed. This mirrors the lazy-import contract in
:class:`dbsprout.train.trainer.QLoRATrainer`.

.. note::
   Real GGUF conversion is **toolchain-validation-pending**: it requires a
   local llama.cpp checkout/build. Unit tests mock ``subprocess.run`` at the
   correct seam and assert argv construction; an integration-marked test
   pins the real resolution contract.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess  # nosec B404 - list-arg, shell=False subprocess only (see _run_tool)
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

# llama.cpp toolchain resolution. ``convert_hf_to_gguf.py`` ships in a
# llama.cpp checkout; ``llama-quantize`` is a built binary. There is no
# importable ``llama_cpp.convert_hf_to_gguf`` symbol — these MUST be invoked
# as subprocesses.
_ENV_CONVERT = "DBSPROUT_LLAMA_CONVERT"
_ENV_QUANTIZE = "DBSPROUT_LLAMA_QUANTIZE"
_KNOWN_CONVERTERS = (
    Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
    Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
)
_QUANTIZE_BIN = "llama-quantize"
_HINT_TOOLCHAIN = (
    "llama.cpp conversion tooling not found. Set {env} to the path of "
    "{what}, or install a llama.cpp build on PATH. (toolchain-validation-pending)"
)
# Portable adapter directory names only: word chars, dot, dash. No path
# separators / traversal so the value is safe to interpolate into a filename.
_SAFE_STEM = re.compile(r"^[\w.-]+$")

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


def _safe_adapter_stem(name: str) -> str:
    """Return *name* iff it is a portable filename component, else raise.

    Rejects path separators / traversal / non-portable chars before the value
    is interpolated into the output ``.gguf`` filename (path-traversal fix).
    """
    if not _SAFE_STEM.match(name) or name in {".", ".."}:
        msg = (
            f"unsafe adapter directory name {name!r}: only [A-Za-z0-9._-] are "
            "allowed (no path separators or traversal)."
        )
        raise ValueError(msg)
    return name


def _resolve_converter() -> Path:
    """Resolve llama.cpp's ``convert_hf_to_gguf.py``; raise an actionable error.

    Resolution order: ``$DBSPROUT_LLAMA_CONVERT`` → known install locations.
    """
    override = os.environ.get(_ENV_CONVERT)
    if override:
        p = Path(override)
        if p.is_file():
            return p.resolve()
        raise RuntimeError(_HINT_TOOLCHAIN.format(env=_ENV_CONVERT, what="convert_hf_to_gguf.py"))
    for candidate in _KNOWN_CONVERTERS:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(_HINT_TOOLCHAIN.format(env=_ENV_CONVERT, what="convert_hf_to_gguf.py"))


def _resolve_quantizer() -> Path:
    """Resolve the ``llama-quantize`` binary; raise an actionable error.

    Resolution order: ``$DBSPROUT_LLAMA_QUANTIZE`` → ``PATH``.
    """
    override = os.environ.get(_ENV_QUANTIZE)
    if override:
        p = Path(override)
        if p.is_file():
            return p.resolve()
        raise RuntimeError(_HINT_TOOLCHAIN.format(env=_ENV_QUANTIZE, what="llama-quantize"))
    on_path = shutil.which(_QUANTIZE_BIN)
    if on_path:
        return Path(on_path).resolve()
    raise RuntimeError(_HINT_TOOLCHAIN.format(env=_ENV_QUANTIZE, what="llama-quantize"))


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


def _run_tool(argv: list[str]) -> None:
    """Run a llama.cpp tool as a subprocess (list args, no shell, checked).

    ``shell=False`` is implicit (list argv); ``check=True`` raises on non-zero
    exit; ``capture_output=True`` keeps stderr for the error message.
    """
    try:
        # argv is a fixed list (resolved tool path + our own staged file
        # paths), shell=False, no untrusted-string interpolation.
        subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603  # nosec B603
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"llama.cpp tool failed ({argv[0]}): {stderr or exc}") from exc


def _quantize_to_gguf(merged_dir: Path, out_path: Path) -> Path:
    """Convert *merged_dir* to a Q4_K_M GGUF at *out_path* via llama.cpp tools.

    Two real subprocess steps (no fabricated ``llama_cpp`` symbol):
    ``convert_hf_to_gguf.py`` (HF dir → f16 GGUF) then ``llama-quantize``
    (f16 → Q4_K_M). The f16 intermediate lives next to *out_path* (already a
    TemporaryDirectory at the call site).
    """
    converter = _resolve_converter()
    quantizer = _resolve_quantizer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f16_path = out_path.with_name(out_path.stem + "-f16.gguf")

    import sys  # noqa: PLC0415 - resolve the active interpreter for the .py script

    _run_tool(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )
    _run_tool([str(quantizer), str(f16_path), str(out_path), _QUANT_TYPE])
    return out_path


def _verify_gguf(path: Path) -> None:
    """Load-verify the produced GGUF; re-raise failures as a clear RuntimeError.

    Only the expected llama.cpp load failures are caught (``RuntimeError`` /
    ``ValueError`` / ``OSError``); the failure is logged with a traceback
    before being re-raised so a corrupt export is diagnosable.
    """
    verifier = _load_verifier()
    try:
        verifier(path)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("exported GGUF failed verification: %s", path)
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

        # Sanitize the adapter dir name before interpolating it into the
        # output filename, then assert the resolved destination stays inside
        # the (resolved) output dir — rejects ``../`` style traversal.
        stem = _safe_adapter_stem(adapter.adapter_path.name)
        out_dir = output_dir.resolve()
        out_path = (out_dir / f"{stem}-{_QUANT_TYPE}.gguf").resolve()
        if out_path.parent != out_dir:
            msg = f"refusing to write outside {out_dir}: {out_path}"
            raise ValueError(msg)

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
            # Build + verify the GGUF entirely inside the TemporaryDirectory;
            # only move it into the real destination once verify succeeds, so
            # a failure never leaves a partial/corrupt model behind (also
            # closes the prior TOCTOU window).
            staged = work / out_path.name
            task = progress.add_task("Merging adapter", total=3)
            merged_dir = _merge_adapter(adapter, base_model, work, fmt=fmt)
            progress.update(task, advance=1, description="Quantizing (Q4_K_M)")
            _quantize_to_gguf(merged_dir, staged)
            progress.update(task, advance=1, description="Verifying")
            if verify:
                _verify_gguf(staged)
            progress.update(task, advance=1, description="Done")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(staged), str(out_path))
            except OSError:
                out_path.unlink(missing_ok=True)
                raise

        logger.info(
            "Exported GGUF: adapter=%s format=%s quant=%s -> %s",
            adapter.adapter_path,
            fmt,
            _QUANT_TYPE,
            out_path,
        )
        return out_path, fmt
