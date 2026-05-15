"""LoRA hot-swap model loader (S-067).

Loads / swaps a GGUF LoRA adapter at inference time **without restarting** the
``llama-cpp-python`` process. There is no stable in-place adapter unload API
across ``llama_cpp`` versions, so a swap is implemented as: construct a fresh
``Llama(model_path=..., lora_path=...)`` handle and drop the previous one. A
*merged* GGUF (the S-066 :class:`dbsprout.train.exporter.Exporter` output under
``.dbsprout/models/custom/``) is just a different ``model_path`` with no
``lora_path``; an *unmerged* adapter uses ``llama_cpp``'s native ``lora_path``
support.

Design (see ``docs/stories/S-067.md`` for the full brainstorm):

* A small **LRU cache** (default capacity 2) keeps recently-loaded handles so
  switching *back* to a recent (model, adapter) pair is a cache hit and avoids
  a reload — while still bounding memory: when the cache is full the
  least-recently-used handle is **unloaded** (``close()`` if available, else
  dropped) *before* the new one is constructed.
* A missing adapter path **falls back to the base model with a WARNING** —
  never an exception (acceptance criterion).
* The ``<2s`` hot-swap budget (:data:`_MAX_SWAP_SECONDS`) is documented here
  and asserted in the unit tests via a mocked monotonic clock — never a real
  model load.

``llama_cpp`` is the optional ``[llm]`` extra and is **lazy-imported inside**
:meth:`ModelLoader._load_llama` only — never at module import — mirroring the
contract in :class:`dbsprout.train.exporter.Exporter` and
:class:`dbsprout.spec.providers.embedded.EmbeddedProvider`. It is mocked in
tests; no real load / download / inference ever runs in CI.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Acceptance criterion: hot-swap overhead must stay under this budget. The
# assertion lives in the unit tests (mocked clock); a real load is never run.
_MAX_SWAP_SECONDS = 2.0

_DEFAULT_CAPACITY = 2

_HINT_LLM = "LoRA hot-swap requires llama-cpp-python. Install with 'pip install dbsprout[llm]'."

# Cache key: (resolved model path, resolved adapter path or None).
_CacheKey = tuple[str, str | None]


class LoadedModel(BaseModel):
    """Immutable, typed result of one :meth:`ModelLoader.load` call.

    ``lora_path`` is ``None`` when no adapter was applied (base model, or the
    requested adapter was missing and we fell back). ``cache_hit`` is ``True``
    when the handle was reused from the in-memory LRU cache (no reload).
    ``swap_seconds`` is the measured construction time (``0.0`` on a cache
    hit) — see :data:`_MAX_SWAP_SECONDS`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_path: Path
    lora_path: Path | None = None
    cache_hit: bool
    swap_seconds: float = Field(ge=0.0)


class ModelLoader:
    """In-memory LRU cache of ``llama_cpp.Llama`` handles for hot-swapping.

    Parameters
    ----------
    capacity:
        Maximum number of distinct (model, adapter) handles kept resident at
        once. Must be ``>= 1``. Defaults to ``2`` so the common
        "switch between two schemas" workflow is reload-free while still
        bounding memory.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            msg = f"capacity must be >= 1, got {capacity}"
            raise ValueError(msg)
        self._capacity = capacity
        self._cache: OrderedDict[_CacheKey, Any] = OrderedDict()
        self.last_swap_seconds: float = 0.0

    @property
    def capacity(self) -> int:
        """Maximum number of resident handles."""
        return self._capacity

    def _load_llama(self) -> Any:
        """Lazily import ``llama_cpp.Llama``; raise a clear hint if missing."""
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(_HINT_LLM) from exc
        return Llama

    @staticmethod
    def _unload(handle: Any) -> None:
        """Best-effort free of a handle's native resources before eviction.

        Newer ``llama_cpp.Llama`` exposes ``close()``; older builds rely on
        ``__del__``. Either way dropping the cache reference is what frees
        memory — closing first is a courtesy and must never raise.
        """
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("ignored error closing evicted model handle")

    def _evict_if_full(self) -> None:
        """Drop + unload the least-recently-used handle when at capacity."""
        while len(self._cache) >= self._capacity:
            _, handle = self._cache.popitem(last=False)
            self._unload(handle)

    def get_handle(self, model_path: Path | str, lora_path: Path | str | None = None) -> Any:
        """Return the cached ``Llama`` handle for a key, or ``None`` if absent.

        Does not construct or evict; purely a cache lookup (used by the
        embedded provider and tests).
        """
        key = self._key(model_path, lora_path)
        return self._cache.get(key)

    @staticmethod
    def _key(model_path: Path | str, lora_path: Path | str | None) -> _CacheKey:
        return (str(model_path), str(lora_path) if lora_path is not None else None)

    def load(
        self,
        model_path: Path | str,
        lora_path: Path | str | None = None,
    ) -> LoadedModel:
        """Load (or reuse) a base model, optionally with a LoRA adapter.

        If *lora_path* is given but does not exist, logs a WARNING and falls
        back to the base model (no exception). On a cache hit the existing
        handle is reused (no reload, ``swap_seconds == 0.0``). On a miss the
        LRU is evicted to capacity, then a fresh handle is constructed.
        """
        base = Path(model_path)
        adapter: Path | None = Path(lora_path) if lora_path is not None else None

        if adapter is not None and not adapter.exists():
            logger.warning(
                "LoRA adapter not found: %s — falling back to base model %s",
                adapter,
                base,
            )
            adapter = None

        key = self._key(base, adapter)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return LoadedModel(
                model_path=base,
                lora_path=adapter,
                cache_hit=True,
                swap_seconds=0.0,
            )

        llama_cls = self._load_llama()
        self._evict_if_full()

        kwargs: dict[str, Any] = {"model_path": str(base), "verbose": False}
        if adapter is not None:
            kwargs["lora_path"] = str(adapter)

        start = time.monotonic()
        handle = llama_cls(**kwargs)
        elapsed = time.monotonic() - start

        self._cache[key] = handle
        self.last_swap_seconds = elapsed
        return LoadedModel(
            model_path=base,
            lora_path=adapter,
            cache_hit=False,
            swap_seconds=elapsed,
        )
