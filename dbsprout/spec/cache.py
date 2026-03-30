"""Spec cache — disk-backed DataSpec caching keyed by schema hash.

Uses diskcache for persistent, thread-safe, LRU-evicting storage.
Subsequent runs with an unchanged schema skip LLM calls entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import diskcache  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.spec.models import DataSpec

_DEFAULT_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB


class SpecCache:
    """Typed wrapper around diskcache.Cache for DataSpec storage."""

    def __init__(
        self,
        cache_dir: Path | str = ".dbsprout/cache",
        size_limit: int = _DEFAULT_SIZE_LIMIT,
    ) -> None:
        self._cache = diskcache.Cache(str(cache_dir), size_limit=size_limit)

    def get(self, schema_hash: str) -> DataSpec | None:
        """Retrieve a cached DataSpec by schema hash.

        Returns ``None`` on cache miss.
        """
        raw: str | None = self._cache.get(schema_hash)
        if raw is None:
            return None

        from dbsprout.spec.models import DataSpec as _DataSpec  # noqa: PLC0415

        return _DataSpec.model_validate_json(raw)

    def put(self, schema_hash: str, spec: DataSpec) -> None:
        """Store a DataSpec in the cache, keyed by schema hash."""
        self._cache.set(schema_hash, spec.model_dump_json())

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying diskcache connection."""
        self._cache.close()
