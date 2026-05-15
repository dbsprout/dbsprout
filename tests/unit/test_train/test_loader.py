"""Unit tests for dbsprout.train.loader — LoRA hot-swap loader (S-067).

The optional heavy dep ``llama_cpp`` is mocked via ``sys.modules`` (same
pattern as ``test_exporter``). No real model load, download, or inference ever
runs. The ``<2s`` swap budget is asserted with a mocked monotonic clock, never
a real load.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from pydantic import ValidationError

from dbsprout.train import loader as loader_mod
from dbsprout.train.loader import _MAX_SWAP_SECONDS, LoadedModel, ModelLoader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_llama(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Install a mock ``llama_cpp`` module exposing a ``Llama`` factory.

    Each ``Llama(...)`` call returns a fresh MagicMock so distinct loads have
    distinct handles (lets eviction/cache-hit tests assert identity).
    """
    llama_factory = mock.MagicMock(name="Llama")
    llama_factory.side_effect = lambda *_a, **_k: mock.MagicMock(name="LlamaHandle")
    llama_mod = types.ModuleType("llama_cpp")
    llama_mod.Llama = llama_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_mod)
    return llama_factory


# --- Task 1: LoadedModel result model --------------------------------------


def test_loaded_model_is_frozen(tmp_path: Path) -> None:
    lm = LoadedModel(
        model_path=tmp_path / "base.gguf",
        lora_path=None,
        cache_hit=False,
        swap_seconds=0.1,
    )
    with pytest.raises(ValidationError):
        lm.cache_hit = True  # type: ignore[misc]


def test_loaded_model_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        LoadedModel(
            model_path=tmp_path / "base.gguf",
            lora_path=None,
            cache_hit=False,
            swap_seconds=0.0,
            bogus=1,  # type: ignore[call-arg]
        )


def test_loaded_model_optional_lora_defaults_none(tmp_path: Path) -> None:
    lm = LoadedModel(
        model_path=tmp_path / "base.gguf",
        cache_hit=False,
        swap_seconds=0.0,
    )
    assert lm.lora_path is None


def test_loaded_model_rejects_negative_swap_seconds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        LoadedModel(
            model_path=tmp_path / "base.gguf",
            cache_hit=False,
            swap_seconds=-0.1,
        )


# --- Task 2: construction + lazy llama_cpp import --------------------------


def test_loader_init_default_capacity() -> None:
    assert ModelLoader().capacity == 2


def test_loader_init_rejects_capacity_below_one() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ModelLoader(capacity=0)


def test_loader_load_llama_missing_dep_raises_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # forces ImportError
    with pytest.raises(RuntimeError, match=r"dbsprout\[llm\]"):
        ModelLoader()._load_llama()


# --- Task 3: load base model, cache miss + hit -----------------------------


def test_load_base_model_constructs_llama(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    ld = ModelLoader()
    result = ld.load(base)
    assert isinstance(result, LoadedModel)
    assert result.lora_path is None
    assert result.cache_hit is False
    fake_llama.assert_called_once()
    _, kwargs = fake_llama.call_args
    assert kwargs["model_path"] == str(base)
    assert "lora_path" not in kwargs


def test_load_base_model_cache_hit_no_reload(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    ld = ModelLoader()
    ld.load(base)
    second = ld.load(base)
    assert second.cache_hit is True
    fake_llama.assert_called_once()


def test_get_handle_returns_cached_llama(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    ld = ModelLoader()
    ld.load(base)
    handle = ld.get_handle(base)
    assert handle is not None
    assert ld.get_handle(tmp_path / "missing.gguf") is None


# --- Task 4: load with LoRA adapter present --------------------------------


def test_load_with_lora_passes_lora_path(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    adapter = tmp_path / "myschema.gguf"
    adapter.write_bytes(b"\x00")
    ld = ModelLoader()
    result = ld.load(base, lora_path=adapter)
    assert result.lora_path == adapter
    _, kwargs = fake_llama.call_args
    assert kwargs["lora_path"] == str(adapter)


def test_load_with_lora_distinct_cache_key_from_base(
    fake_llama: mock.MagicMock, tmp_path: Path
) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    adapter = tmp_path / "myschema.gguf"
    adapter.write_bytes(b"\x00")
    ld = ModelLoader(capacity=4)
    ld.load(base)
    ld.load(base, lora_path=adapter)
    assert fake_llama.call_count == 2


# --- Task 5: missing adapter falls back to base + warning ------------------


def test_missing_adapter_falls_back_to_base_with_warning(
    fake_llama: mock.MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    missing = tmp_path / "nope.gguf"
    ld = ModelLoader()
    with caplog.at_level("WARNING"):
        result = ld.load(base, lora_path=missing)
    assert result.lora_path is None
    assert "nope.gguf" in caplog.text
    _, kwargs = fake_llama.call_args
    assert "lora_path" not in kwargs


# --- Task 6: capacity eviction unloads LRU (memory budget) -----------------


def test_capacity_eviction_unloads_lru(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    c = tmp_path / "c.gguf"
    for p in (a, b, c):
        p.write_bytes(b"\x00")
    ld = ModelLoader(capacity=2)
    ld.load(a)
    a_handle = ld.get_handle(a)
    ld.load(b)
    ld.load(c)  # evicts A (LRU)
    assert ld.get_handle(a) is None
    a_handle.close.assert_called_once()  # type: ignore[union-attr]
    ld.load(a)  # A reloaded -> 4 constructions total
    assert fake_llama.call_count == 4


def test_switch_back_within_capacity_is_cache_hit(
    fake_llama: mock.MagicMock, tmp_path: Path
) -> None:
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    for p in (a, b):
        p.write_bytes(b"\x00")
    ld = ModelLoader(capacity=2)
    ld.load(a)
    ld.load(b)
    again = ld.load(a)
    assert again.cache_hit is True
    assert fake_llama.call_count == 2


def test_eviction_handles_handle_without_close(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    """A handle lacking close() must not break eviction (best-effort unload)."""
    handles = [object(), object(), object()]
    fake_llama.side_effect = lambda *_a, **_k: handles.pop(0)
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    c = tmp_path / "c.gguf"
    for p in (a, b, c):
        p.write_bytes(b"\x00")
    ld = ModelLoader(capacity=2)
    ld.load(a)
    ld.load(b)
    ld.load(c)  # evicts A; A handle has no close() -> must not raise
    assert ld.get_handle(a) is None


def test_eviction_swallows_close_error(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    """A handle whose close() raises must not break eviction (logged, swallowed)."""
    bad = mock.MagicMock(name="BadHandle")
    bad.close.side_effect = RuntimeError("native free failed")
    good = mock.MagicMock(name="GoodHandle")
    handles = [bad, good, mock.MagicMock(name="ThirdHandle")]
    fake_llama.side_effect = lambda *_a, **_k: handles.pop(0)
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    c = tmp_path / "c.gguf"
    for p in (a, b, c):
        p.write_bytes(b"\x00")
    ld = ModelLoader(capacity=2)
    ld.load(a)
    ld.load(b)
    ld.load(c)  # evicts A; A.close() raises -> swallowed, eviction proceeds
    assert ld.get_handle(a) is None
    bad.close.assert_called_once()


# --- Task 7: <2s swap budget recorded (mocked clock) -----------------------


def test_swap_seconds_under_budget(
    fake_llama: mock.MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    ticks = iter([10.0, 10.05])
    monkeypatch.setattr(loader_mod.time, "monotonic", lambda: next(ticks))
    ld = ModelLoader()
    result = ld.load(base)
    assert result.swap_seconds == pytest.approx(0.05)
    assert result.swap_seconds < _MAX_SWAP_SECONDS
    assert ld.last_swap_seconds == pytest.approx(0.05)


def test_cache_hit_has_zero_swap_seconds(fake_llama: mock.MagicMock, tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    base.write_bytes(b"\x00")
    ld = ModelLoader()
    ld.load(base)
    hit = ld.load(base)
    assert hit.swap_seconds == 0.0


def test_swap_budget_constant() -> None:
    assert _MAX_SWAP_SECONDS == 2.0
