"""Workspace.persist_spec + hydrate_from_cache tests (S-122).

The two helpers wire the in-memory :class:`~dbsprout.web.workspace.Workspace`
to the disk-backed :class:`~dbsprout.spec.cache.SpecCache` so that:

* every successful spec edit (S-119 column PUT, S-121 table row_count PUT)
  persists the new ``DataSpec`` to the cache, keyed by ``schema.schema_hash()``,
  and
* on schema (re)load (POST /api/connect, POST /api/schema/load) the workspace
  hydrates the spec from the cache when a matching entry exists, instead of
  re-running the heuristic builder.

The cache helpers must NEVER raise — cache I/O is best-effort: a broken cache
must degrade gracefully (the Studio edit still succeeds; hydration falls back
to the lazy heuristic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
from dbsprout.web.workspace import Workspace

if TYPE_CHECKING:
    from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[ColumnSchema(name="id", data_type=ColumnType.INTEGER, primary_key=True)],
                primary_key=["id"],
            )
        ]
    )


def _spec(seed: int = 42) -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                row_count=100,
                columns={"id": GeneratorConfig(provider="numeric", method="sequence")},
            )
        ],
        global_seed=seed,
    )


class _ExplodingCache:
    """Test double whose every method raises — proves persist/hydrate swallow."""

    def get(self, schema_hash: str) -> DataSpec | None:
        _ = schema_hash
        msg = "boom (get)"
        raise RuntimeError(msg)

    def put(self, schema_hash: str, spec: DataSpec) -> None:
        _ = (schema_hash, spec)
        msg = "boom (put)"
        raise RuntimeError(msg)


# ── constructor: optional cache injection ─────────────────────────────


def test_workspace_accepts_injected_cache(tmp_path: Path) -> None:
    """The constructor must accept an optional ``spec_cache=`` for tests."""
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        ws = Workspace(spec_cache=cache)
        assert ws.get_schema() is None  # ordinary state still default
    finally:
        cache.close()


def test_workspace_default_construction_still_works() -> None:
    """The new optional kwarg must not break the existing no-arg ctor."""
    ws = Workspace()
    assert ws.get_schema() is None
    assert ws.get_spec() is None


# ── persist_spec ──────────────────────────────────────────────────────


def test_persist_spec_writes_to_cache_keyed_by_schema_hash(tmp_path: Path) -> None:
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        ws = Workspace(spec_cache=cache)
        schema = _schema()
        spec = _spec()
        ws.set_schema(schema)
        ws.set_spec(spec)

        ws.persist_spec()

        cached = cache.get(schema.schema_hash())
        assert cached is not None
        assert cached == spec
    finally:
        cache.close()


def test_persist_spec_noop_without_spec(tmp_path: Path) -> None:
    """Schema set but no spec → cache must remain empty; no raise."""
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        ws = Workspace(spec_cache=cache)
        ws.set_schema(_schema())

        ws.persist_spec()  # must not raise

        assert cache.get(_schema().schema_hash()) is None
    finally:
        cache.close()


def test_persist_spec_noop_without_schema(tmp_path: Path) -> None:
    """Spec set but no schema → nothing to key on; must not raise."""
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        ws = Workspace(spec_cache=cache)
        ws.set_spec(_spec())

        ws.persist_spec()  # must not raise — no schema hash to write under
    finally:
        cache.close()


def test_persist_spec_swallows_cache_errors() -> None:
    """A broken cache must not propagate exceptions to the route handler."""
    ws = Workspace(spec_cache=_ExplodingCache())
    ws.set_schema(_schema())
    ws.set_spec(_spec())

    # No raise — Studio edit must succeed even when persistence is unhappy.
    ws.persist_spec()


# ── hydrate_from_cache ────────────────────────────────────────────────


def test_hydrate_from_cache_sets_spec_on_hit(tmp_path: Path) -> None:
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        schema = _schema()
        spec = _spec(seed=999)
        cache.put(schema.schema_hash(), spec)

        ws = Workspace(spec_cache=cache)
        ws.set_schema(schema)

        hit = ws.hydrate_from_cache(schema.schema_hash())

        assert hit is True
        assert ws.get_spec() == spec
        assert ws.get_spec() is not spec  # rehydrated from JSON; new object
    finally:
        cache.close()


def test_hydrate_from_cache_noop_on_miss(tmp_path: Path) -> None:
    """An empty cache must leave ``workspace.spec`` untouched."""
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        ws = Workspace(spec_cache=cache)
        ws.set_schema(_schema())
        assert ws.get_spec() is None

        hit = ws.hydrate_from_cache(_schema().schema_hash())

        assert hit is False
        assert ws.get_spec() is None
    finally:
        cache.close()


def test_hydrate_from_cache_swallows_cache_errors() -> None:
    """A cache that raises on ``get`` must yield a clean miss, not a 500."""
    ws = Workspace(spec_cache=_ExplodingCache())
    ws.set_schema(_schema())

    hit = ws.hydrate_from_cache(_schema().schema_hash())

    assert hit is False
    assert ws.get_spec() is None


def test_hydrate_from_cache_does_not_overwrite_when_caller_already_set(
    tmp_path: Path,
) -> None:
    """If ``self.spec`` is already set, hydrate still replaces it on a hit.

    The hydration step runs *before* any heuristic build, but if the caller has
    pre-populated ``self.spec`` (rare — fresh schema load always nulls first),
    a hit should still win: the cached version is the authoritative one.
    """
    cache = SpecCache(cache_dir=tmp_path / "cache")
    try:
        schema = _schema()
        cached = _spec(seed=7)
        cache.put(schema.schema_hash(), cached)

        ws = Workspace(spec_cache=cache)
        ws.set_schema(schema)
        ws.set_spec(_spec(seed=42))  # stale in-memory

        hit = ws.hydrate_from_cache(schema.schema_hash())

        assert hit is True
        new_spec = ws.get_spec()
        assert new_spec is not None
        assert new_spec.global_seed == 7
    finally:
        cache.close()


# ── default cache wiring (no injection) ───────────────────────────────


def test_default_cache_used_when_none_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no cache is injected, ``persist_spec`` should still work using the
    lazily-initialised default cache. We isolate it by running from a tmp cwd."""
    monkeypatch.chdir(tmp_path)
    ws = Workspace()
    schema = _schema()
    ws.set_schema(schema)
    ws.set_spec(_spec(seed=11))

    ws.persist_spec()  # must not raise; writes to .dbsprout/cache under tmp cwd

    # Fresh workspace at the same cwd should be able to hydrate the cached entry.
    ws2 = Workspace()
    ws2.set_schema(schema)
    hit = ws2.hydrate_from_cache(schema.schema_hash())
    assert hit is True
    assert ws2.get_spec() is not None
    assert ws2.get_spec().global_seed == 11  # type: ignore[union-attr]
