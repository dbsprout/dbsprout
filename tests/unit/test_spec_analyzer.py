"""Tests for dbsprout.spec.analyzer — LLM spec analyzer with retry and fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dbsprout.privacy.enforcer import PrivacyError, PrivacyTier
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.analyzer import SpecAnalyzer, _build_spec_prompt, heuristic_fallback
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

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
        model_used="test-mock",
    )


class TestCacheHit:
    def test_cache_hit_skips_provider(self, tmp_path: Path) -> None:
        """Cache hit returns cached spec without calling provider."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()

        # Pre-populate cache
        cache = SpecCache(cache_dir=tmp_path / "cache")
        spec = _mock_dataspec(schema.schema_hash())
        cache.put(schema.schema_hash(), spec)
        cache.close()

        mock_provider = MagicMock()
        analyzer = SpecAnalyzer(provider=mock_provider, cache_dir=tmp_path / "cache")
        try:
            result = analyzer.analyze(schema)

            mock_provider.generate_spec.assert_not_called()
            assert result == spec
        finally:
            analyzer.close()


class TestCacheMiss:
    def test_calls_provider_and_caches(self, tmp_path: Path) -> None:
        """Cache miss calls provider, result is cached."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        mock_provider = MagicMock()
        mock_provider.generate_spec.return_value = mock_spec

        analyzer = SpecAnalyzer(provider=mock_provider, cache_dir=tmp_path / "cache")
        try:
            result = analyzer.analyze(schema)

            mock_provider.generate_spec.assert_called_once()
            assert result.tables[0].table_name == "users"

            # Verify cached
            cache = SpecCache(cache_dir=tmp_path / "cache")
            cached = cache.get(schema.schema_hash())
            cache.close()
            assert cached is not None
        finally:
            analyzer.close()


class TestRetry:
    def test_retries_on_failure(self, tmp_path: Path) -> None:
        """Provider failure triggers retry, second call succeeds."""
        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        mock_provider = MagicMock()
        # First call raises, second succeeds
        mock_provider.generate_spec.side_effect = [
            ValueError("LLM output invalid"),
            mock_spec,
        ]

        analyzer = SpecAnalyzer(provider=mock_provider, cache_dir=tmp_path / "cache")
        try:
            result = analyzer.analyze(schema)

            assert mock_provider.generate_spec.call_count == 2
            assert result.tables[0].table_name == "users"
        finally:
            analyzer.close()


class TestFallback:
    def test_fallback_on_total_failure(self, tmp_path: Path) -> None:
        """All retries fail → heuristic fallback produces valid DataSpec."""
        schema = _simple_schema()

        mock_provider = MagicMock()
        mock_provider.generate_spec.side_effect = ValueError("always fails")

        analyzer = SpecAnalyzer(provider=mock_provider, cache_dir=tmp_path / "cache")
        try:
            result = analyzer.analyze(schema)

            # Should have retried 3 times then fallen back
            assert mock_provider.generate_spec.call_count == 3
            # Fallback should produce a valid DataSpec with correct tables
            assert isinstance(result, DataSpec)
            assert result.get_table_spec("users") is not None
        finally:
            analyzer.close()

    def testheuristic_fallback_produces_valid_dataspec(self) -> None:
        """heuristic_fallback converts heuristic mappings to DataSpec."""
        schema = _simple_schema()

        result = heuristic_fallback(schema)

        assert isinstance(result, DataSpec)
        ts = result.get_table_spec("users")
        assert ts is not None
        assert "id" in ts.columns
        assert "email" in ts.columns
        assert ts.columns["email"].provider != ""


class TestPrompt:
    def test_prompt_includes_ddl_and_example(self) -> None:
        """Prompt must contain schema DDL and example output."""
        schema = _simple_schema()
        prompt = _build_spec_prompt(schema)

        assert "CREATE TABLE" in prompt
        assert "users" in prompt
        assert "email" in prompt
        # Should contain example or instruction
        assert "DataSpec" in prompt or "JSON" in prompt


class TestPrivacyEnforcement:
    """Privacy tier enforcement in SpecAnalyzer."""

    def test_local_tier_blocks_cloud_provider(self, tmp_path: Path) -> None:
        mock_provider = MagicMock()
        mock_provider.provider_locality = "cloud"

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.LOCAL,
        )
        try:
            with pytest.raises(PrivacyError):
                analyzer.analyze(_simple_schema())
            mock_provider.generate_spec.assert_not_called()
        finally:
            analyzer.close()

    def test_local_tier_allows_local_provider(self, tmp_path: Path) -> None:
        mock_spec = _mock_dataspec()
        mock_provider = MagicMock()
        mock_provider.provider_locality = "local"
        mock_provider.generate_spec.return_value = mock_spec

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.LOCAL,
        )
        try:
            result = analyzer.analyze(_simple_schema())
            mock_provider.generate_spec.assert_called_once()
            assert result.tables[0].table_name == "users"
        finally:
            analyzer.close()

    def test_redacted_tier_redacts_schema_for_cloud(self, tmp_path: Path) -> None:
        mock_spec = _mock_dataspec()
        mock_provider = MagicMock()
        mock_provider.provider_locality = "cloud"
        mock_provider.generate_spec.return_value = mock_spec

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.REDACTED,
        )
        try:
            analyzer.analyze(_simple_schema())
            # The schema passed to the provider should be redacted
            call_args = mock_provider.generate_spec.call_args
            passed_schema = call_args[0][0]
            # Table names should be hashed
            assert passed_schema.tables[0].name.startswith("tbl_")
            # Column names should be hashed
            assert passed_schema.tables[0].columns[0].name.startswith("col_")
        finally:
            analyzer.close()

    def test_redacted_tier_does_not_redact_for_local(self, tmp_path: Path) -> None:
        mock_spec = _mock_dataspec()
        mock_provider = MagicMock()
        mock_provider.provider_locality = "local"
        mock_provider.generate_spec.return_value = mock_spec

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.REDACTED,
        )
        try:
            analyzer.analyze(_simple_schema())
            call_args = mock_provider.generate_spec.call_args
            passed_schema = call_args[0][0]
            # Local provider gets original schema
            assert passed_schema.tables[0].name == "users"
        finally:
            analyzer.close()

    def test_cloud_tier_passes_full_schema(self, tmp_path: Path) -> None:
        mock_spec = _mock_dataspec()
        mock_provider = MagicMock()
        mock_provider.provider_locality = "cloud"
        mock_provider.generate_spec.return_value = mock_spec

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.CLOUD,
        )
        try:
            analyzer.analyze(_simple_schema())
            call_args = mock_provider.generate_spec.call_args
            passed_schema = call_args[0][0]
            assert passed_schema.tables[0].name == "users"
        finally:
            analyzer.close()

    def test_default_privacy_tier_is_local(self, tmp_path: Path) -> None:
        mock_provider = MagicMock()
        mock_provider.provider_locality = "cloud"

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
        )
        try:
            with pytest.raises(PrivacyError):
                analyzer.analyze(_simple_schema())
        finally:
            analyzer.close()

    def test_provider_without_locality_defaults_to_cloud(self, tmp_path: Path) -> None:
        """Providers without provider_locality are treated as cloud (safe default)."""
        mock_provider = MagicMock(spec=[])

        analyzer = SpecAnalyzer(
            provider=mock_provider,
            cache_dir=tmp_path / "cache",
            privacy_tier=PrivacyTier.LOCAL,
        )
        try:
            with pytest.raises(PrivacyError):
                analyzer.analyze(_simple_schema())
        finally:
            analyzer.close()
