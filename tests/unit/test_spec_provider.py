"""Tests for dbsprout.spec.providers.embedded — EmbeddedProvider (mock-based)."""

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
from dbsprout.spec.providers.embedded import (
    EmbeddedProvider,
    _build_prompt,
)

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
    def test_cache_hit_skips_llm(self, tmp_path: Path) -> None:
        """Cached spec returned without model call."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            # Pre-populate cache
            spec = _mock_dataspec(schema.schema_hash())
            cache.put(schema.schema_hash(), spec)

            provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
            result = provider.generate_spec(schema)

            assert result == spec
        finally:
            cache.close()
            provider.close()


class TestCacheMiss:
    def test_cache_miss_calls_llm(self, tmp_path: Path) -> None:
        """Model called when cache is empty."""
        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())
        mock_response = mock_spec.model_dump_json()

        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        # Mock the _run_inference method
        provider._run_inference = MagicMock(return_value=mock_response)  # type: ignore[assignment]

        result = provider.generate_spec(schema)

        provider._run_inference.assert_called_once()  # type: ignore[union-attr]
        assert result.tables[0].table_name == "users"
        provider.close()


class TestResultCached:
    def test_result_stored_in_cache(self, tmp_path: Path) -> None:
        """After LLM call, spec is cached for next time."""
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        schema = _simple_schema()
        mock_spec = _mock_dataspec(schema.schema_hash())

        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        provider._run_inference = MagicMock(  # type: ignore[assignment]
            return_value=mock_spec.model_dump_json(),
        )

        provider.generate_spec(schema)

        # Verify it's in the cache
        cache = SpecCache(cache_dir=tmp_path / "cache")
        try:
            cached = cache.get(schema.schema_hash())
            assert cached is not None
            assert cached.tables[0].table_name == "users"
        finally:
            cache.close()
            provider.close()


class TestLLMNotInstalled:
    def test_raises_clear_error(self, tmp_path: Path) -> None:
        """ImportError with install instructions when llama-cpp not available."""
        schema = _simple_schema()

        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        # Simulate llama-cpp not installed by making _run_inference raise
        provider._run_inference = MagicMock(  # type: ignore[assignment]
            side_effect=ImportError("No module named 'llama_cpp'"),
        )

        with pytest.raises(ImportError, match="llama_cpp"):
            provider.generate_spec(schema)
        provider.close()


class TestPrompt:
    def test_prompt_includes_ddl(self) -> None:
        """Prompt must contain the schema DDL."""
        schema = _simple_schema()
        prompt = _build_prompt(schema)

        assert "CREATE TABLE" in prompt
        assert "users" in prompt
        assert "email" in prompt


class TestImportHfHubDownload:
    def test_import_error_gives_clear_message(self) -> None:
        """Missing huggingface-hub gives clear install instructions."""
        import sys  # noqa: PLC0415

        from dbsprout.spec.providers.embedded import (  # noqa: PLC0415
            _import_hf_hub_download,
        )

        # Temporarily remove huggingface_hub if present
        original = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pip install dbsprout"):
                _import_hf_hub_download()
        finally:
            if original is not None:
                sys.modules["huggingface_hub"] = original
            else:
                sys.modules.pop("huggingface_hub", None)


class TestEnsureLlm:
    def test_llama_cpp_not_installed_error(self, tmp_path: Path) -> None:
        """Missing llama-cpp-python gives clear install instructions."""
        import sys  # noqa: PLC0415

        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        provider._download_model = MagicMock(return_value=tmp_path / "model.gguf")  # type: ignore[assignment]

        original = sys.modules.get("llama_cpp")
        sys.modules["llama_cpp"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pip install dbsprout"):
                provider._ensure_llm()
        finally:
            if original is not None:
                sys.modules["llama_cpp"] = original
            else:
                sys.modules.pop("llama_cpp", None)
            provider.close()


class TestLoRAHotSwap:
    """S-067: additive LoRA hot-swap support on the embedded provider."""

    def test_default_lora_path_none_behaviour_unchanged(self, tmp_path: Path) -> None:
        """No lora_path -> identical S-025 behaviour, no ModelLoader created."""
        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        assert provider.lora_path is None
        assert provider._loader is None
        provider.close()

    def test_ctor_accepts_lora_path(self, tmp_path: Path) -> None:
        adapter = tmp_path / "myschema.gguf"
        adapter.write_bytes(b"\x00")
        provider = EmbeddedProvider(cache_dir=tmp_path / "cache", lora_path=adapter)
        assert provider.lora_path == adapter
        provider.close()

    def test_set_lora_swaps_via_loader(self, tmp_path: Path) -> None:
        """set_lora() routes the next inference load through ModelLoader."""
        from unittest.mock import MagicMock, patch  # noqa: PLC0415

        adapter = tmp_path / "myschema.gguf"
        adapter.write_bytes(b"\x00")
        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        provider._download_model = MagicMock(  # type: ignore[assignment]
            return_value=tmp_path / "base.gguf"
        )

        with patch("dbsprout.train.loader.ModelLoader") as loader_cls:
            handle = MagicMock(name="LlamaHandle")
            loader_cls.return_value.load.return_value = MagicMock()
            loader_cls.return_value.get_handle.return_value = handle

            provider.set_lora(adapter)
            assert provider.lora_path == adapter
            llm = provider._ensure_llm()

            loader_cls.return_value.load.assert_called_once()
            _, kwargs = loader_cls.return_value.load.call_args
            assert kwargs["lora_path"] == adapter
            assert llm is handle
        provider.close()

    def test_set_lora_none_clears_adapter(self, tmp_path: Path) -> None:
        adapter = tmp_path / "myschema.gguf"
        adapter.write_bytes(b"\x00")
        provider = EmbeddedProvider(cache_dir=tmp_path / "cache", lora_path=adapter)
        provider.set_lora(None)
        assert provider.lora_path is None
        provider.close()

    def test_set_lora_resets_cached_llm(self, tmp_path: Path) -> None:
        """Swapping adapters drops the previously cached handle (hot-swap)."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        provider = EmbeddedProvider(cache_dir=tmp_path / "cache")
        provider._llm = MagicMock(name="stale")
        provider.set_lora(tmp_path / "new.gguf")
        assert provider._llm is None
        provider.close()
