"""Tests for dbsprout.spec.providers.cloud — Cloud LLM provider (mock-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
from dbsprout.spec.providers.cloud import CloudProvider, _build_cloud_prompt

if TYPE_CHECKING:
    from pathlib import Path


def _simple_schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="email",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
    )


def _mock_dataspec(schema_hash: str = "") -> DataSpec:
    return DataSpec(
        tables=[
            TableSpec(
                table_name="users",
                columns={
                    "id": GeneratorConfig(provider="builtin.autoincrement"),
                    "email": GeneratorConfig(provider="mimesis.Person.email"),
                },
            ),
        ],
        schema_hash=schema_hash,
        model_used="gpt-4o-mini",
    )


class TestCacheHit:
    def test_cache_hit_skips_api(self, tmp_path: Path) -> None:
        """Cached spec returned without API call."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        cache = SpecCache(cache_dir=tmp_path / "cache")
        spec = _mock_dataspec(schema.schema_hash())
        cache.put(schema.schema_hash(), spec)
        cache.close()

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        try:
            result = provider.generate_spec(schema)
            assert result == spec
        finally:
            provider.close()


class TestCacheMiss:
    def test_calls_api_and_caches(self, tmp_path: Path) -> None:
        """API called on cache miss, result cached."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        provider._call_llm = MagicMock(return_value=mock_spec)  # type: ignore[assignment]

        try:
            result = provider.generate_spec(schema)

            provider._call_llm.assert_called_once()  # type: ignore[union-attr]
            assert result.tables[0].table_name == "users"

            # Verify cached
            cache = SpecCache(cache_dir=tmp_path / "cache")
            cached = cache.get(schema.schema_hash())
            cache.close()
            assert cached is not None
        finally:
            provider.close()


class TestResultCached:
    def test_result_stored(self, tmp_path: Path) -> None:
        """Spec stored in cache after API call."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        provider._call_llm = MagicMock(return_value=mock_spec)  # type: ignore[assignment]

        try:
            provider.generate_spec(schema)

            cache = SpecCache(cache_dir=tmp_path / "cache")
            assert cache.get(schema.schema_hash()) is not None
            cache.close()
        finally:
            provider.close()


class TestImportError:
    def test_import_error_message(self, tmp_path: Path) -> None:
        """Missing litellm gives clear install instructions."""
        schema = _simple_schema()

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        provider._call_llm = MagicMock(  # type: ignore[assignment]
            side_effect=ImportError("No module named 'litellm'"),
        )

        try:
            with pytest.raises(ImportError, match="litellm"):
                provider.generate_spec(schema)
        finally:
            provider.close()


class TestPrompt:
    def test_prompt_includes_ddl(self) -> None:
        """Prompt contains schema DDL."""
        schema = _simple_schema()
        prompt = _build_cloud_prompt(schema)

        assert "CREATE TABLE" in prompt
        assert "users" in prompt
        assert "email" in prompt
        assert "DataSpec" in prompt or "JSON" in prompt
