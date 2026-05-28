"""AC-3 (S-095): ``get_registry()`` is safe under concurrent first-call.

Two threads racing the first call must observe (a) a single
``PluginRegistry`` instance and (b) at most one ``PluginRegistry``
construction (no double entry-point walk / duplicate warnings).
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import dbsprout.plugins.registry as reg_mod
from dbsprout.plugins.registry import PluginRegistry, get_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    get_registry.cache_clear()
    yield
    get_registry.cache_clear()


def test_get_registry_single_instance_under_concurrent_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = 8
    constructions = 0
    constructions_lock = threading.Lock()
    barrier = threading.Barrier(workers)

    real_init = PluginRegistry.__init__

    def counting_init(self: PluginRegistry) -> None:
        nonlocal constructions
        with constructions_lock:
            constructions += 1
        # Widen the race window so a broken lock would let a second
        # thread enter construction before the first finishes.
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=2.0)
        real_init(self)

    monkeypatch.setattr(PluginRegistry, "__init__", counting_init)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: get_registry(), range(workers)))

    # (a) every thread observed the exact same instance.
    assert len({id(r) for r in results}) == 1
    # (b) the registry was constructed at most once (single entry-point walk).
    assert constructions == 1


def test_cache_clear_resets_and_rebuilds(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_registry()
    assert get_registry() is first

    get_registry.cache_clear()
    second = get_registry()
    assert second is not first


def test_reset_for_tests_alias_clears_registry() -> None:
    first = get_registry()
    reg_mod._reset_for_tests()
    assert get_registry() is not first
