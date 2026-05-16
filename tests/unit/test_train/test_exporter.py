"""Unit tests for dbsprout.train.exporter — GGUF export pipeline (S-066).

All heavy optional deps (``torch``, ``transformers``, ``peft``, ``mlx_lm``,
the llama.cpp ``convert_hf_to_gguf`` converter, ``llama_cpp``) are mocked. The
adapter-format detector is exercised on real temp dirs (filesystem only). No
real model load, weight merge, quantization, or network ever runs — tests
assert pipeline wiring (stage order, paths, format selection) only.
"""

from __future__ import annotations

import inspect
import struct
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from pydantic import ValidationError

import dbsprout.train as train_pkg
from dbsprout.train import Exporter, ExportResult
from dbsprout.train import exporter as exporter_mod
from dbsprout.train.config import LoRAAdapter
from dbsprout.train.exporter import (
    _GGUF_MAGIC,
    _QUANT_TYPE,
    _assert_gguf_magic,
    _detect_adapter_format,
    _load_merge_deps,
    _merge_adapter,
    _quantize_to_gguf,
    _resolve_converter,
    _resolve_quantizer,
    _safe_adapter_stem,
)


def _gguf_bytes(version: int = 3, *, extra: bytes = b"") -> bytes:
    """Return a minimal valid GGUF header prefix (magic + LE uint32 version)."""
    return _GGUF_MAGIC + struct.pack("<I", version) + extra


if TYPE_CHECKING:
    from collections.abc import Iterator


# --- Task 1: ExportResult model --------------------------------------------


def test_export_result_is_frozen(tmp_path: Path) -> None:
    res = ExportResult(
        gguf_path=tmp_path / "m.gguf",
        size_bytes=1024,
        quant_type="Q4_K_M",
        source_format="peft",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    with pytest.raises(ValidationError):
        res.size_bytes = 2048  # type: ignore[misc]


def test_export_result_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExportResult(
            gguf_path=tmp_path / "m.gguf",
            size_bytes=1,
            quant_type="Q4_K_M",
            source_format="peft",
            base_model="b",
            bogus=1,  # type: ignore[call-arg]
        )


def test_export_result_rejects_negative_size(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExportResult(
            gguf_path=tmp_path / "m.gguf",
            size_bytes=-1,
            quant_type="Q4_K_M",
            source_format="peft",
            base_model="b",
        )


# --- Task 2: adapter-format detection --------------------------------------


def test_detect_adapter_format_peft(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"\x00")
    assert _detect_adapter_format(tmp_path) == "peft"


def test_detect_adapter_format_mlx(tmp_path: Path) -> None:
    (tmp_path / "adapters.safetensors").write_bytes(b"\x00")
    assert _detect_adapter_format(tmp_path) == "mlx"


def test_detect_adapter_format_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="adapter format"):
        _detect_adapter_format(tmp_path)


# --- Task 3: missing-extras guards -----------------------------------------


def test_load_merge_deps_peft_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)  # forces ImportError
    with pytest.raises(RuntimeError, match=r"dbsprout\[train-cuda\]"):
        _load_merge_deps("peft")


def test_load_merge_deps_mlx_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(RuntimeError, match=r"dbsprout\[train-mlx\]"):
        _load_merge_deps("mlx")


# --- Review #9: real llama.cpp converter/quantizer resolution --------------


def test_resolve_converter_no_env_known_locations_miss_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(exporter_mod._ENV_CONVERT, raising=False)
    monkeypatch.setattr(exporter_mod, "_KNOWN_CONVERTERS", ())
    with pytest.raises(RuntimeError, match="DBSPROUT_LLAMA_CONVERT"):
        _resolve_converter()


def test_resolve_quantizer_no_env_not_on_path_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(exporter_mod._ENV_QUANTIZE, raising=False)
    monkeypatch.setattr(exporter_mod.shutil, "which", lambda _n: None)
    with pytest.raises(RuntimeError, match="DBSPROUT_LLAMA_QUANTIZE"):
        _resolve_quantizer()


def test_resolve_quantizer_found_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "llama-quantize"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv(exporter_mod._ENV_QUANTIZE, raising=False)
    monkeypatch.setattr(exporter_mod.shutil, "which", lambda _n: str(binary))
    assert _resolve_quantizer() == binary.resolve()


def test_resolve_converter_uses_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter", encoding="utf-8")
    monkeypatch.setenv("DBSPROUT_LLAMA_CONVERT", str(script))
    assert _resolve_converter() == script.resolve()


def test_resolve_converter_missing_raises_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DBSPROUT_LLAMA_CONVERT", str(tmp_path / "nope.py"))
    with pytest.raises(RuntimeError, match="DBSPROUT_LLAMA_CONVERT"):
        _resolve_converter()


def test_resolve_quantizer_uses_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llama-quantize"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("DBSPROUT_LLAMA_QUANTIZE", str(binary))
    assert _resolve_quantizer() == binary.resolve()


def test_resolve_quantizer_missing_raises_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DBSPROUT_LLAMA_QUANTIZE", str(tmp_path / "nope"))
    monkeypatch.setattr(exporter_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="DBSPROUT_LLAMA_QUANTIZE"):
        _resolve_quantizer()


# --- Review #11: output-path containment + filename sanitization -----------


def test_safe_adapter_stem_rejects_path_separators() -> None:
    with pytest.raises(ValueError, match="adapter"):
        _safe_adapter_stem("../evil")
    with pytest.raises(ValueError, match="adapter"):
        _safe_adapter_stem("a/b")
    with pytest.raises(ValueError, match="adapter"):
        _safe_adapter_stem("a\\b")


def test_safe_adapter_stem_accepts_portable_name() -> None:
    assert _safe_adapter_stem("abc123") == "abc123"
    assert _safe_adapter_stem("schema_deadbeef-1") == "schema_deadbeef-1"


def test_load_verifier_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(RuntimeError, match=r"dbsprout\[llm\]"):
        exporter_mod._load_verifier()


def test_load_verifier_smoke_loads_with_llama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llama_cls = mock.MagicMock(name="Llama")
    llama_mod = types.ModuleType("llama_cpp")
    llama_mod.Llama = llama_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_mod)
    verifier = exporter_mod._load_verifier()
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    verifier(gguf)
    llama_cls.assert_called_once_with(model_path=str(gguf), n_ctx=8, verbose=False)


# --- S-066b Task 1: GGUF magic/header validation ---------------------------


def test_assert_gguf_magic_accepts_valid_header(tmp_path: Path) -> None:
    gguf = tmp_path / "ok.gguf"
    # magic + version 3 + a couple of stub counts (tensor/kv) — only the
    # 8-byte magic+version prefix is validated.
    gguf.write_bytes(_gguf_bytes(3, extra=b"\x00" * 16))
    _assert_gguf_magic(gguf)  # must not raise


def test_assert_gguf_magic_rejects_wrong_magic(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"PK\x03\x04" + struct.pack("<I", 3))
    with pytest.raises(RuntimeError, match=r"not a valid GGUF"):
        _assert_gguf_magic(bad)


def test_assert_gguf_magic_rejects_truncated_file(tmp_path: Path) -> None:
    short = tmp_path / "short.gguf"
    short.write_bytes(b"GGU")  # fewer than magic+version bytes
    with pytest.raises(RuntimeError, match=r"not a valid GGUF"):
        _assert_gguf_magic(short)


def test_assert_gguf_magic_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match=r"not a valid GGUF"):
        _assert_gguf_magic(empty)


def test_assert_gguf_magic_rejects_unsupported_version(tmp_path: Path) -> None:
    bad_ver = tmp_path / "v99.gguf"
    bad_ver.write_bytes(_gguf_bytes(99))
    with pytest.raises(RuntimeError, match=r"unsupported GGUF version"):
        _assert_gguf_magic(bad_ver)


def test_assert_gguf_magic_rejects_zero_version(tmp_path: Path) -> None:
    bad_ver = tmp_path / "v0.gguf"
    bad_ver.write_bytes(_gguf_bytes(0))
    with pytest.raises(RuntimeError, match=r"unsupported GGUF version"):
        _assert_gguf_magic(bad_ver)


def test_assert_gguf_magic_error_names_path(tmp_path: Path) -> None:
    bad = tmp_path / "namedme.gguf"
    bad.write_bytes(b"nope")
    with pytest.raises(RuntimeError, match=r"namedme\.gguf"):
        _assert_gguf_magic(bad)


# --- shared fixtures --------------------------------------------------------


@pytest.fixture
def peft_adapter(tmp_path: Path) -> LoRAAdapter:
    adir = tmp_path / "adapters" / "abc123"
    adir.mkdir(parents=True)
    (adir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adir / "adapter_model.safetensors").write_bytes(b"\x00")
    return LoRAAdapter(
        adapter_path=adir,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        epochs=1,
        train_samples=2,
        final_loss=0.1,
        duration_seconds=1.0,
    )


@pytest.fixture
def base_model_dir(tmp_path: Path) -> Path:
    p = tmp_path / "base"
    p.mkdir()
    (p / "config.json").write_text("{}", encoding="utf-8")
    return p


@pytest.fixture
def fake_merge_deps(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, mock.MagicMock]]:
    """Inject mock ``torch`` / ``transformers`` / ``peft`` modules."""
    captured: dict[str, mock.MagicMock] = {}

    base = mock.MagicMock(name="base_model")
    merged = mock.MagicMock(name="merged_model")
    base.merge_and_unload.return_value = merged

    auto_model = mock.MagicMock(name="AutoModelForCausalLM")
    auto_model.from_pretrained.return_value = mock.MagicMock(name="hf_base")
    auto_tok = mock.MagicMock(name="AutoTokenizer")
    auto_tok.from_pretrained.return_value = mock.MagicMock(name="tok")
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = auto_model  # type: ignore[attr-defined]
    transformers_mod.AutoTokenizer = auto_tok  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    captured["AutoModelForCausalLM"] = auto_model
    captured["AutoTokenizer"] = auto_tok

    peft_model = mock.MagicMock(name="PeftModel")
    peft_model.from_pretrained.return_value = base
    peft_mod = types.ModuleType("peft")
    peft_mod.PeftModel = peft_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", peft_mod)
    captured["PeftModel"] = peft_model
    captured["base"] = base
    captured["merged"] = merged

    fake_torch = types.SimpleNamespace(float16="float16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return captured


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> mock.MagicMock:
    """Mock ``subprocess.run`` at the real seam + stub tool resolution.

    The conversion is now a real ``subprocess.run([...], check=True)`` call to
    llama.cpp's ``convert_hf_to_gguf.py`` + ``llama-quantize`` — no fabricated
    ``llama_cpp`` symbol. Tests assert the argv (list args, no shell) instead
    of a fake function.
    """
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("# converter", encoding="utf-8")
    quant = tmp_path / "llama-quantize"
    quant.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("dbsprout.train.exporter._resolve_converter", conv.resolve)
    monkeypatch.setattr("dbsprout.train.exporter._resolve_quantizer", quant.resolve)

    run = mock.MagicMock(name="subprocess.run")

    def _run(argv: list[str], **_kw: object) -> mock.MagicMock:
        run(argv, **_kw)
        # Materialize whatever output file the stage expects so the pipeline
        # proceeds (converter -> f16 gguf; quantizer -> final gguf).
        # converter: `... --outfile <f16.gguf> --outtype f16`
        # quantizer: `<bin> <in.gguf> <out.gguf> Q4_K_M`
        out = next(
            (a for a in reversed(argv) if isinstance(a, str) and a.endswith(".gguf")),
            None,
        )
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            # A real conversion writes a GGUF; emit a valid magic+version
            # header so the always-on _assert_gguf_magic gate passes.
            Path(out).write_bytes(_gguf_bytes(3, extra=b"\x00" * 9))
        return mock.MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(exporter_mod.subprocess, "run", _run)
    return run


# --- Task 4: merge stage ----------------------------------------------------


def test_merge_adapter_peft_returns_safetensors_dir(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    fake_merge_deps: dict[str, mock.MagicMock],
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    out = _merge_adapter(peft_adapter, base_model_dir, work, fmt="peft")
    assert out.is_relative_to(work)
    fake_merge_deps["base"].merge_and_unload.assert_called_once()
    fake_merge_deps["merged"].save_pretrained.assert_called_once()


def test_merge_adapter_mlx_path(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fuse = mock.MagicMock(name="mlx_fuse")
    mlx_lm_mod = types.ModuleType("mlx_lm")
    fuse_sub = types.ModuleType("mlx_lm.fuse")
    fuse_sub.fuse_model = fuse  # type: ignore[attr-defined]
    mlx_lm_mod.fuse = fuse_sub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.fuse", fuse_sub)

    work = tmp_path / "work"
    work.mkdir()
    out = _merge_adapter(peft_adapter, base_model_dir, work, fmt="mlx")
    assert out.is_relative_to(work)
    fuse.assert_called_once()


# --- Task 5: quantize stage -------------------------------------------------


def test_quantize_to_gguf_invokes_convert_then_quantize_no_shell(
    tmp_path: Path, fake_subprocess: mock.MagicMock
) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "models" / "custom" / "x.gguf"
    result = _quantize_to_gguf(merged, out)
    assert result == out
    assert out.parent.is_dir()
    assert _QUANT_TYPE == "Q4_K_M"

    # Two subprocess.run calls: HF->GGUF f16 convert, then llama-quantize.
    assert fake_subprocess.call_count == 2
    convert_call, quant_call = fake_subprocess.call_args_list
    convert_argv = convert_call.args[0]
    quant_argv = quant_call.args[0]
    # list-args (not a shell string) + no shell=True anywhere.
    assert isinstance(convert_argv, list)
    assert isinstance(quant_argv, list)
    for call in (convert_call, quant_call):
        assert call.kwargs.get("shell", False) is False
        assert call.kwargs.get("check") is True
        assert call.kwargs.get("capture_output") is True
        # A (generous) timeout bounds a genuinely hung toolchain process.
        assert isinstance(call.kwargs.get("timeout"), (int, float))
        assert call.kwargs["timeout"] > 0
    # llama-quantize CLI is `<bin> <in.gguf> <out.gguf> <TYPE>`.
    assert quant_argv[-1] == _QUANT_TYPE
    assert quant_argv[-2] == str(out)


def test_quantize_to_gguf_validates_gguf_magic_on_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The f16 convert output AND the final Q4_K_M output must each be passed
    # through _assert_gguf_magic so a non-GGUF subprocess result is caught.
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("# converter", encoding="utf-8")
    quant = tmp_path / "llama-quantize"
    quant.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("dbsprout.train.exporter._resolve_converter", conv.resolve)
    monkeypatch.setattr("dbsprout.train.exporter._resolve_quantizer", quant.resolve)

    valid = _GGUF_MAGIC + struct.pack("<I", 3) + b"\x00" * 16

    def _run(argv: list[str], **_kw: object) -> mock.MagicMock:
        out = next(
            (a for a in reversed(argv) if isinstance(a, str) and a.endswith(".gguf")),
            None,
        )
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(valid)
        return mock.MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(exporter_mod.subprocess, "run", _run)

    seen: list[Path] = []
    real_assert = exporter_mod._assert_gguf_magic

    def _spy(p: Path) -> None:
        seen.append(p)
        real_assert(p)

    monkeypatch.setattr(exporter_mod, "_assert_gguf_magic", _spy)

    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "models" / "x.gguf"
    _quantize_to_gguf(merged, out)
    # f16 intermediate + final output both validated.
    assert len(seen) == 2
    assert any(p.name.endswith("-f16.gguf") for p in seen)
    assert out in seen


def test_quantize_to_gguf_aborts_when_converter_output_not_gguf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The converter "succeeds" but writes a non-GGUF file: quantize must NOT
    # run and a clear error is raised.
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("x", encoding="utf-8")
    quant = tmp_path / "llama-quantize"
    quant.write_text("x", encoding="utf-8")
    monkeypatch.setattr("dbsprout.train.exporter._resolve_converter", conv.resolve)
    monkeypatch.setattr("dbsprout.train.exporter._resolve_quantizer", quant.resolve)

    calls: list[str] = []

    def _run(argv: list[str], **_kw: object) -> mock.MagicMock:
        calls.append(argv[0])
        out = next(
            (a for a in reversed(argv) if isinstance(a, str) and a.endswith(".gguf")),
            None,
        )
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"NOT-A-GGUF-FILE")  # convert produced junk
        return mock.MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(exporter_mod.subprocess, "run", _run)

    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(RuntimeError, match=r"not a valid GGUF"):
        _quantize_to_gguf(merged, tmp_path / "o.gguf")
    # Only the converter ran; the quantizer was never invoked.
    assert len(calls) == 1


def test_quantize_to_gguf_rejects_missing_merged_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("x", encoding="utf-8")
    quant = tmp_path / "llama-quantize"
    quant.write_text("x", encoding="utf-8")
    monkeypatch.setattr("dbsprout.train.exporter._resolve_converter", conv.resolve)
    monkeypatch.setattr("dbsprout.train.exporter._resolve_quantizer", quant.resolve)
    ran = mock.MagicMock(name="should_not_run")
    monkeypatch.setattr(exporter_mod.subprocess, "run", ran)
    with pytest.raises((RuntimeError, ValueError), match=r"merged"):
        _quantize_to_gguf(tmp_path / "no-such-merged", tmp_path / "o.gguf")
    ran.assert_not_called()


def test_quantize_to_gguf_argv_uses_absolute_paths(
    tmp_path: Path, fake_subprocess: mock.MagicMock
) -> None:
    import os  # noqa: PLC0415

    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "models" / "custom" / "x.gguf"
    _quantize_to_gguf(merged, out)
    convert_call, quant_call = fake_subprocess.call_args_list
    convert_argv = convert_call.args[0]
    quant_argv = quant_call.args[0]
    # Every path-like token interpolated into either argv is absolute, so the
    # subprocess cannot resolve a relative path nor be option-injected.
    for token in convert_argv[1:] + quant_argv:
        if isinstance(token, str) and ("/" in token or token.endswith(".gguf")):
            assert os.path.isabs(token), f"non-absolute path token in argv: {token}"
        assert not token.startswith("-") or token in {
            "--outfile",
            "--outtype",
        }, f"unexpected option-like token: {token}"


def test_quantize_to_gguf_subprocess_failure_raises_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess  # noqa: PLC0415

    conv = tmp_path / "convert_hf_to_gguf.py"
    conv.write_text("x", encoding="utf-8")
    quant = tmp_path / "llama-quantize"
    quant.write_text("x", encoding="utf-8")
    monkeypatch.setattr("dbsprout.train.exporter._resolve_converter", conv.resolve)
    monkeypatch.setattr("dbsprout.train.exporter._resolve_quantizer", quant.resolve)

    def _boom(argv: list[str], **_kw: object) -> None:
        raise subprocess.CalledProcessError(1, argv, stderr="convert exploded")

    monkeypatch.setattr(exporter_mod.subprocess, "run", _boom)
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(RuntimeError, match="convert exploded"):
        _quantize_to_gguf(merged, tmp_path / "o.gguf")


def test_run_tool_surfaces_timeout_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A hung toolchain process must not block forever: subprocess.TimeoutExpired
    # is surfaced as a clear DBSprout RuntimeError naming the tool.
    import subprocess  # noqa: PLC0415

    def _hang(argv: list[str], **_kw: object) -> None:
        raise subprocess.TimeoutExpired(argv, timeout=1)

    monkeypatch.setattr(exporter_mod.subprocess, "run", _hang)
    with pytest.raises(RuntimeError, match=r"timed out"):
        exporter_mod._run_tool(["/bin/true", "arg"])


# --- Task 6: load-verify seam ----------------------------------------------


def test_verify_gguf_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    verifier = mock.MagicMock(name="verifier")
    monkeypatch.setattr(exporter_mod, "_load_verifier", lambda: verifier)
    exporter_mod._verify_gguf(gguf)
    verifier.assert_called_once_with(gguf)


def test_verify_gguf_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")

    def _boom(_: Path) -> None:
        raise ValueError("bad gguf")

    monkeypatch.setattr(exporter_mod, "_load_verifier", lambda: _boom)
    with pytest.raises(RuntimeError, match="verif"):
        exporter_mod._verify_gguf(gguf)


# --- Task 7: Exporter.to_gguf orchestration --------------------------------


@pytest.fixture
def stubbed_exporter(monkeypatch: pytest.MonkeyPatch) -> dict[str, mock.MagicMock]:
    """Stub every heavy stage so to_gguf wiring can be asserted in isolation."""
    calls: list[str] = []
    rec: dict[str, mock.MagicMock] = {}

    def _merge(adapter, base, work, fmt):  # type: ignore[no-untyped-def]
        calls.append("merge")
        d = work / "merged"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _quant(merged, out):  # type: ignore[no-untyped-def]
        calls.append("quant")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 17)
        return out

    verify = mock.MagicMock(name="verify")

    def _verify(p):  # type: ignore[no-untyped-def]
        calls.append("verify")
        verify(p)

    monkeypatch.setattr(exporter_mod, "_merge_adapter", _merge)
    monkeypatch.setattr(exporter_mod, "_quantize_to_gguf", _quant)
    monkeypatch.setattr(exporter_mod, "_verify_gguf", _verify)
    rec["verify"] = verify
    rec["calls"] = calls  # type: ignore[assignment]
    return rec


def test_to_gguf_runs_pipeline_in_order(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    out_dir = tmp_path / "models" / "custom"
    path = Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=out_dir, quiet=True)
    assert path.suffix == ".gguf"
    assert path.parent == out_dir.resolve()
    assert stubbed_exporter["calls"] == ["merge", "quant", "verify"]  # type: ignore[comparison-overlap]
    # Verify runs on the STAGED tmp file (atomic export, finding #10), then the
    # file is moved into the real destination — so it is verified exactly once
    # and on a different (temp) path than the final one.
    stubbed_exporter["verify"].assert_called_once()
    verified = stubbed_exporter["verify"].call_args.args[0]
    assert verified.name == path.name
    assert verified != path


def test_to_gguf_filename_is_deterministic(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    out_dir = tmp_path / "c"
    p1 = Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=out_dir, quiet=True)
    p2 = Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=out_dir, quiet=True)
    assert p1 == p2
    assert "abc123" in p1.name


def test_to_gguf_default_output_dir(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    sig = inspect.signature(Exporter.to_gguf)
    default = sig.parameters["output_dir"].default
    assert default == Path(".dbsprout/models/custom")


def test_to_gguf_quiet_and_progress_paths(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    # exercise the non-quiet (Rich progress) branch too
    path = Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=tmp_path / "c", quiet=False)
    assert path.exists()


def test_to_gguf_missing_base_model_raises(
    peft_adapter: LoRAAdapter,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    with pytest.raises(FileNotFoundError, match="base model"):
        Exporter().to_gguf(peft_adapter, tmp_path / "nope", output_dir=tmp_path / "c", quiet=True)


def test_to_gguf_missing_adapter_raises(
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    adapter = LoRAAdapter(
        adapter_path=tmp_path / "gone",
        base_model="b",
        epochs=1,
        train_samples=1,
        final_loss=None,
        duration_seconds=0.0,
    )
    with pytest.raises(FileNotFoundError, match="adapter"):
        Exporter().to_gguf(adapter, base_model_dir, output_dir=tmp_path / "c", quiet=True)


def test_to_gguf_verify_false_skips_verify(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    Exporter().to_gguf(
        peft_adapter,
        base_model_dir,
        output_dir=tmp_path / "c",
        quiet=True,
        verify=False,
    )
    stubbed_exporter["verify"].assert_not_called()


def test_to_gguf_result_returns_typed_export_result(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    res = Exporter().to_gguf_result(
        peft_adapter, base_model_dir, output_dir=tmp_path / "c", quiet=True
    )
    assert isinstance(res, ExportResult)
    assert res.quant_type == "Q4_K_M"
    assert res.source_format == "peft"
    assert res.size_bytes == 17
    assert res.base_model == peft_adapter.base_model
    assert res.gguf_path.exists()


# --- Task 8: package export -------------------------------------------------


def test_exporter_exported_from_package() -> None:
    assert "Exporter" in train_pkg.__all__
    assert "ExportResult" in train_pkg.__all__
    assert train_pkg.Exporter is exporter_mod.Exporter
    assert train_pkg.ExportResult is exporter_mod.ExportResult


# --- Review #10: atomic output / partial-leak cleanup ----------------------


def test_to_gguf_no_partial_file_on_verify_failure(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # merge+quant succeed but verify fails: the final output dir must NOT
    # contain a half-written .gguf (it was built in a TemporaryDirectory and
    # only moved into place AFTER verify passed).
    def _merge(adapter, base, work, fmt):  # type: ignore[no-untyped-def]
        d = work / "merged"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _quant(merged, out):  # type: ignore[no-untyped-def]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 17)
        return out

    monkeypatch.setattr(exporter_mod, "_merge_adapter", _merge)
    monkeypatch.setattr(exporter_mod, "_quantize_to_gguf", _quant)
    monkeypatch.setattr(
        exporter_mod,
        "_verify_gguf",
        mock.MagicMock(side_effect=RuntimeError("bad gguf")),
    )

    out_dir = tmp_path / "models" / "custom"
    with pytest.raises(RuntimeError, match="bad gguf"):
        Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=out_dir, quiet=True)
    # No partial leak in the real destination.
    assert list(out_dir.glob("*.gguf")) == [] if out_dir.exists() else True


def test_to_gguf_output_lands_in_destination_after_success(
    peft_adapter: LoRAAdapter,
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    out_dir = tmp_path / "models" / "custom"
    path = Exporter().to_gguf(peft_adapter, base_model_dir, output_dir=out_dir, quiet=True)
    assert path.parent == out_dir
    assert path.is_file()
    assert path.stat().st_size == 17


# --- Review #11: output-path containment -----------------------------------


def test_to_gguf_rejects_nonportable_adapter_name(
    base_model_dir: Path,
    tmp_path: Path,
    stubbed_exporter: dict[str, mock.MagicMock],
) -> None:
    # An adapter dir whose *name* has a non-portable char (space) must be
    # rejected before being interpolated into the output .gguf filename.
    evil_dir = tmp_path / "adapters" / "evil name"
    evil_dir.mkdir(parents=True)
    (evil_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    bad = LoRAAdapter(
        adapter_path=evil_dir,
        base_model="b",
        epochs=1,
        train_samples=1,
        final_loss=None,
        duration_seconds=0.0,
    )
    with pytest.raises(ValueError, match="adapter"):
        Exporter().to_gguf(bad, base_model_dir, output_dir=tmp_path / "c", quiet=True)


# --- Review #12: narrowed verify exception handling ------------------------


def test_verify_gguf_logs_and_reraises_on_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")

    def _boom(_: Path) -> None:
        raise ValueError("could not load gguf")

    monkeypatch.setattr(exporter_mod, "_load_verifier", lambda: _boom)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="verif"):
        exporter_mod._verify_gguf(gguf)
    assert "verification" in caplog.text.lower()


# --- Review #9: real llama.cpp toolchain resolution (integration) ----------


@pytest.mark.integration
def test_real_llama_cpp_toolchain_resolves() -> None:
    """Pin the real converter/quantizer resolution contract.

    Skipped unless a real llama.cpp toolchain is reachable (env override or
    on PATH) — so a unit run on a runner without the toolchain never hides a
    regression. Real conversion remains toolchain-validation-pending.
    """
    import os  # noqa: PLC0415
    import shutil as _shutil  # noqa: PLC0415

    if (
        os.environ.get(exporter_mod._ENV_CONVERT) is None
        and _shutil.which("llama-quantize") is None
    ):
        pytest.skip("requires a real llama.cpp toolchain (toolchain-validation-pending)")
    converter = _resolve_converter()
    quantizer = _resolve_quantizer()
    assert converter.is_file()
    assert quantizer.is_file()
