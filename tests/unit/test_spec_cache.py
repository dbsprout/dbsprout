"""Tests for dbsprout.spec.cache — spec caching with diskcache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

if TYPE_CHECKING:
    from pathlib import Path


def _sample_spec(schema_hash: str = "abc123") -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                columns={"email": GeneratorConfig(provider="mimesis.Person.email")},
            ),
        ],
        schema_hash=schema_hash,
        model_used="test-model",
    )


class TestCacheMiss:
    def test_get_returns_none_on_empty_cache(self, tmp_path: Path) -> None:
        """Get on empty cache returns None."""
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            result = cache.get("nonexistent_hash")
            assert result is None
        finally:
            cache.close()


class TestCacheRoundTrip:
    def test_put_and_get_returns_identical_spec(self, tmp_path: Path) -> None:
        """Put then get returns identical DataSpec."""
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            spec = _sample_spec("hash123")
            cache.put("hash123", spec)

            result = cache.get("hash123")
            assert result is not None
            assert result == spec
            assert result.schema_hash == "hash123"
            assert result.model_used == "test-model"
            assert result.tables[0].table_name == "users"
        finally:
            cache.close()


class TestCacheDifferentHash:
    def test_different_hash_misses(self, tmp_path: Path) -> None:
        """Get with different hash returns None."""
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            cache.put("hash_a", _sample_spec("hash_a"))

            result = cache.get("hash_b")
            assert result is None
        finally:
            cache.close()


class TestCacheClear:
    def test_clear_removes_all(self, tmp_path: Path) -> None:
        """Clear removes all cached entries."""
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            cache.put("hash1", _sample_spec("hash1"))
            cache.put("hash2", _sample_spec("hash2"))

            cache.clear()

            assert cache.get("hash1") is None
            assert cache.get("hash2") is None
        finally:
            cache.close()


class TestCacheDirCreated:
    def test_dir_created_automatically(self, tmp_path: Path) -> None:
        """Cache directory created on first use."""
        cache_dir = tmp_path / "nested" / "deep" / "cache"
        assert not cache_dir.exists()

        cache = SpecCache(cache_dir=cache_dir)
        try:
            cache.put("h", _sample_spec())
            assert cache_dir.exists()
        finally:
            cache.close()


class TestCacheOverwrite:
    def test_put_overwrites_existing(self, tmp_path: Path) -> None:
        """Put with same key overwrites previous value."""
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            spec_v1 = _sample_spec("hash1")
            spec_v2 = DataSpec(
                tables=[
                    TableSpec(
                        table_name="orders",
                        columns={"id": GeneratorConfig(provider="builtin.autoincrement")},
                    ),
                ],
                schema_hash="hash1",
                model_used="updated-model",
            )

            cache.put("hash1", spec_v1)
            cache.put("hash1", spec_v2)

            result = cache.get("hash1")
            assert result is not None
            assert result.tables[0].table_name == "orders"
            assert result.model_used == "updated-model"
        finally:
            cache.close()
