"""Tests for dbsprout.spec.providers.ollama — Ollama LLM provider (mock-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec
from dbsprout.spec.providers.ollama import OllamaProvider, _build_ollama_prompt

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
        model_used="ollama/llama3.2",
    )


class TestCacheHit:
    def test_cache_hit_skips_ollama(self, tmp_path: Path) -> None:
        """Cached spec returned without Ollama call."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        cache = SpecCache(cache_dir=tmp_path / "cache")
        spec = _mock_dataspec(schema.schema_hash())
        cache.put(schema.schema_hash(), spec)
        cache.close()

        provider = OllamaProvider(cache_dir=str(tmp_path / "cache"))
        try:
            result = provider.generate_spec(schema)
            assert result == spec
        finally:
            provider.close()


class TestCacheMiss:
    def test_calls_ollama_and_caches(self, tmp_path: Path) -> None:
        """Ollama called on cache miss, result cached."""
        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        provider = OllamaProvider(cache_dir=str(tmp_path / "cache"))
        provider._check_health = MagicMock()  # type: ignore[assignment]
        provider._call_llm = MagicMock(return_value=mock_spec)  # type: ignore[assignment]

        try:
            result = provider.generate_spec(schema)
            provider._check_health.assert_called_once()  # type: ignore[union-attr]
            provider._call_llm.assert_called_once()  # type: ignore[union-attr]
            assert result.tables[0].table_name == "users"
        finally:
            provider.close()


class TestHealthCheck:
    def test_health_check_success(self, tmp_path: Path) -> None:
        """Successful health check does not raise."""
        provider = OllamaProvider(cache_dir=str(tmp_path / "cache"))
        try:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                provider._check_health()  # Should not raise
        finally:
            provider.close()

    def test_health_check_fails(self, tmp_path: Path) -> None:
        """Connection refused → clear error."""
        provider = OllamaProvider(cache_dir=str(tmp_path / "cache"))
        try:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = ConnectionError("refused")

                with pytest.raises(ConnectionError, match="Ollama"):
                    provider._check_health()
        finally:
            provider.close()


class TestCustomHost:
    def test_host_from_env(self, tmp_path: Path) -> None:
        """OLLAMA_HOST env var respected."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://myhost:9999"}):
            provider = OllamaProvider(cache_dir=str(tmp_path / "cache"))
            assert provider._host == "http://myhost:9999"
            provider.close()

    def test_explicit_host(self, tmp_path: Path) -> None:
        """Explicit host parameter takes precedence."""
        provider = OllamaProvider(
            host="http://custom:8080",
            cache_dir=str(tmp_path / "cache"),
        )
        assert provider._host == "http://custom:8080"
        provider.close()


class TestPrompt:
    def test_prompt_includes_ddl(self) -> None:
        """Prompt contains schema DDL."""
        schema = _simple_schema()
        prompt = _build_ollama_prompt(schema)

        assert "CREATE TABLE" in prompt
        assert "users" in prompt
        assert "email" in prompt
