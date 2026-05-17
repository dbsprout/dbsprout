"""Tests for dbsprout.spec.providers.cloud — Cloud LLM provider (mock-based)."""

from __future__ import annotations

import sys
import types
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
from dbsprout.spec.providers.base import SpecUsage
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


class _RealisticUsage:
    """Mirrors litellm's response.usage object — exact real attribute names."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _RealisticCompletion:
    """Mirrors the raw completion returned by instructor.create_with_completion."""

    def __init__(self, usage: _RealisticUsage, response_cost: float | None) -> None:
        self.usage = usage
        self._hidden_params: dict[str, object] = {}
        if response_cost is not None:
            self._hidden_params["response_cost"] = response_cost


def _install_fake_litellm_instructor(  # noqa: PLR0913 - test fixture knobs
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: DataSpec,
    usage: _RealisticUsage,
    completion_cost_value: float | None,
    completion_cost_raises: bool = False,
    response_cost: float | None = None,
) -> tuple[types.ModuleType, MagicMock]:
    """Inject fake ``litellm`` + ``instructor`` modules with the REAL surface.

    Real symbols modelled (verified against upstream docs/source):
    - ``instructor.from_litellm(litellm.completion)`` -> client
    - ``client.chat.completions.create_with_completion(...)`` ->
      ``(DataSpec, raw_completion)``
    - ``raw_completion.usage.{prompt,completion,total}_tokens``
    - ``litellm.completion_cost(completion_response=raw_completion)``
    - ``raw_completion._hidden_params["response_cost"]`` fallback
    """
    raw = _RealisticCompletion(usage, response_cost)

    completions = MagicMock()
    completions.create_with_completion = MagicMock(return_value=(spec, raw))
    client = MagicMock()
    client.chat.completions = completions

    litellm_mod = types.ModuleType("litellm")
    litellm_mod.completion = MagicMock(name="litellm.completion")  # type: ignore[attr-defined]
    cost_mock = MagicMock(name="litellm.completion_cost")
    if completion_cost_raises:
        cost_mock.side_effect = RuntimeError("model not in cost map")
    else:
        cost_mock.return_value = completion_cost_value
    litellm_mod.completion_cost = cost_mock  # type: ignore[attr-defined]

    instructor_mod = types.ModuleType("instructor")
    instructor_mod.from_litellm = MagicMock(  # type: ignore[attr-defined]
        return_value=client
    )

    monkeypatch.setitem(sys.modules, "litellm", litellm_mod)
    monkeypatch.setitem(sys.modules, "instructor", instructor_mod)
    return litellm_mod, completions.create_with_completion


class TestCloudUsageCapture:
    def test_real_tokens_and_cost_from_completion_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cloud provider surfaces real litellm token counts + completion_cost."""
        schema = _simple_schema()
        spec = _mock_dataspec(schema.schema_hash())
        usage = _RealisticUsage(prompt_tokens=321, completion_tokens=654)
        litellm_mod, create = _install_fake_litellm_instructor(
            monkeypatch,
            spec=spec,
            usage=usage,
            completion_cost_value=0.001234,
        )

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        try:
            result = provider.generate_spec(schema)
            assert result.tables[0].table_name == "users"

            captured = provider.get_last_usage()
            assert captured == SpecUsage(
                tokens_sent=321,
                tokens_received=654,
                cost_usd=0.001234,
            )
            # Real instructor symbol used (not .create()).
            create.assert_called_once()
            # Real cost helper called with the real kwarg.
            litellm_mod.completion_cost.assert_called_once()
            _, kwargs = litellm_mod.completion_cost.call_args
            assert "completion_response" in kwargs
        finally:
            provider.close()

    def test_cost_falls_back_to_hidden_params_response_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When completion_cost raises, use raw._hidden_params['response_cost']."""
        schema = _simple_schema()
        spec = _mock_dataspec(schema.schema_hash())
        usage = _RealisticUsage(prompt_tokens=10, completion_tokens=20)
        _install_fake_litellm_instructor(
            monkeypatch,
            spec=spec,
            usage=usage,
            completion_cost_value=None,
            completion_cost_raises=True,
            response_cost=0.009,
        )

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        try:
            provider.generate_spec(schema)
            captured = provider.get_last_usage()
            assert captured is not None
            assert captured.tokens_sent == 10
            assert captured.tokens_received == 20
            assert captured.cost_usd == 0.009
        finally:
            provider.close()

    def test_cost_zero_when_no_source_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Honest 0.0 cost when neither completion_cost nor hidden cost exists."""
        schema = _simple_schema()
        spec = _mock_dataspec(schema.schema_hash())
        usage = _RealisticUsage(prompt_tokens=5, completion_tokens=5)
        _install_fake_litellm_instructor(
            monkeypatch,
            spec=spec,
            usage=usage,
            completion_cost_value=None,
            completion_cost_raises=True,
            response_cost=None,
        )

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        try:
            provider.generate_spec(schema)
            captured = provider.get_last_usage()
            assert captured is not None
            assert captured.cost_usd == 0.0
            assert captured.tokens_sent == 5
            assert captured.tokens_received == 5
        finally:
            provider.close()

    def test_get_last_usage_none_on_cache_hit(self, tmp_path: Path) -> None:
        """A cache hit performs no real call → no usage surfaced."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        cache = SpecCache(cache_dir=tmp_path / "cache")
        cache.put(schema.schema_hash(), _mock_dataspec(schema.schema_hash()))
        cache.close()

        provider = CloudProvider(cache_dir=str(tmp_path / "cache"))
        try:
            provider.generate_spec(schema)
            assert provider.get_last_usage() is None
        finally:
            provider.close()
