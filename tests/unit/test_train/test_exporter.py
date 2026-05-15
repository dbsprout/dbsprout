"""Unit tests for dbsprout.train.exporter — GGUF export pipeline (S-066).

All heavy optional deps (``torch``, ``transformers``, ``peft``, ``mlx_lm``,
the llama.cpp ``convert_hf_to_gguf`` converter, ``llama_cpp``) are mocked. The
adapter-format detector is exercised on real temp dirs (filesystem only). No
real model load, weight merge, quantization, or network ever runs — tests
assert pipeline wiring (stage order, paths, format selection) only.
"""

from __future__ import annotations

import inspect
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
    _QUANT_TYPE,
    _detect_adapter_format,
    _load_convert,
    _load_merge_deps,
    _merge_adapter,
    _quantize_to_gguf,
)

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


def test_load_convert_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(RuntimeError, match=r"dbsprout\[llm\]"):
        _load_convert()


def test_load_convert_returns_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = mock.MagicMock(name="convert_hf_to_gguf")
    llama_mod = types.ModuleType("llama_cpp")
    llama_mod.convert_hf_to_gguf = conv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_mod)
    assert _load_convert() is conv


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
def fake_convert(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Patch the converter seam so quantization is a no-op recording call."""
    conv = mock.MagicMock(name="convert_hf_to_gguf")

    def _fake_load_convert() -> mock.MagicMock:
        return conv

    monkeypatch.setattr("dbsprout.train.exporter._load_convert", _fake_load_convert)
    return conv


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


def test_quantize_to_gguf_uses_q4_k_m(tmp_path: Path, fake_convert: mock.MagicMock) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "models" / "custom" / "x.gguf"
    result = _quantize_to_gguf(merged, out)
    assert result == out
    assert out.parent.is_dir()
    kwargs = fake_convert.call_args.kwargs
    args = fake_convert.call_args.args
    assert _QUANT_TYPE == "Q4_K_M"
    assert _QUANT_TYPE in (kwargs.get("outtype"), kwargs.get("quantization"), *args)


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
    assert path.parent == out_dir
    assert stubbed_exporter["calls"] == ["merge", "quant", "verify"]  # type: ignore[comparison-overlap]
    stubbed_exporter["verify"].assert_called_once_with(path)


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
