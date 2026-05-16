"""Embedded LLM provider — local inference via llama-cpp-python.

Runs Qwen2.5-1.5B (or another GGUF model) locally with GBNF grammar
constraints for guaranteed valid JSON output. No API keys needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import DataSpec

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_DEFAULT_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
_DEFAULT_N_CTX = 4096
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_MAX_TOKENS = 4096


class EmbeddedProvider:
    """Spec provider using local llama-cpp-python inference.

    Flow:
    1. Check SpecCache for cached spec
    2. If miss: build prompt from schema DDL
    3. Run inference with GBNF grammar constraint
    4. Parse response as DataSpec
    5. Store in cache and return
    """

    provider_locality: str = "local"

    def __init__(
        self,
        cache_dir: Path | str = ".dbsprout/cache",
        model_repo: str = _DEFAULT_MODEL_REPO,
        model_file: str = _DEFAULT_MODEL_FILE,
        lora_path: Path | str | None = None,
    ) -> None:
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        self._cache = SpecCache(cache_dir=cache_dir)
        self._model_repo = model_repo
        self._model_file = model_file
        self._llm: Any = None
        # S-067 LoRA hot-swap: when a ``lora_path`` is in use the ``Llama``
        # handle is owned by a ``ModelLoader`` (LRU cache + restart-free swap).
        # When ``lora_path`` is ``None`` the loader is never created and the
        # original S-025 direct-``Llama`` path is preserved unchanged.
        self._lora_path: Path | None = Path(lora_path) if lora_path is not None else None
        self._loader: Any = None

    @property
    def lora_path(self) -> Path | None:
        """The active LoRA adapter path, or ``None`` for the base model."""
        return self._lora_path

    def set_lora(self, lora_path: Path | str | None) -> None:
        """Hot-swap the LoRA adapter used for subsequent inference.

        Sets the adapter and drops the cached ``Llama`` handle so the next
        :meth:`_run_inference` reloads via the :class:`ModelLoader` (which
        unloads the previous handle before constructing the new one — no
        process restart). Passing ``None`` reverts to the base model.
        """
        self._lora_path = Path(lora_path) if lora_path is not None else None
        # Force the next _ensure_llm() to (re)load through the loader.
        self._llm = None

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from a database schema.

        Checks cache first. On miss, runs LLM inference with GBNF
        grammar constraints and caches the result.
        """
        from dbsprout.spec.models import DataSpec as _DataSpec  # noqa: PLC0415

        schema_hash = schema.schema_hash()

        # Check cache
        cached = self._cache.get(schema_hash)
        if cached is not None:
            logger.info("Spec cache hit for hash %s", schema_hash)
            return cached

        # Cache miss — run inference
        logger.info("Spec cache miss for hash %s — running LLM inference", schema_hash)
        prompt = _build_prompt(schema)
        raw_json = self._run_inference(prompt)
        spec = _DataSpec.model_validate_json(raw_json)

        # Update with correct schema hash and store
        spec = spec.model_copy(update={"schema_hash": schema_hash})
        self._cache.put(schema_hash, spec)

        return spec

    def _run_inference(self, prompt: str) -> str:  # pragma: no cover
        """Run LLM inference with GBNF grammar constraint.

        Returns raw JSON string. Requires llama-cpp-python.
        """
        llm = self._ensure_llm()

        from dbsprout.spec.grammar import generate_dataspec_grammar  # noqa: PLC0415

        grammar_str = generate_dataspec_grammar()

        try:
            from llama_cpp import LlamaGrammar  # noqa: PLC0415
        except ImportError:
            msg = (
                "llama-cpp-python is required for embedded LLM inference. "
                "Install it with: pip install dbsprout[llm]"
            )
            raise ImportError(msg) from None

        grammar = LlamaGrammar.from_string(grammar_str)

        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            grammar=grammar,
            temperature=_DEFAULT_TEMPERATURE,
            max_tokens=_DEFAULT_MAX_TOKENS,
        )

        content: str = response["choices"][0]["message"]["content"]
        return content

    def _ensure_llm(self) -> Any:
        """Lazy-load the LLM model. Requires llama-cpp-python.

        When a LoRA adapter is active (S-067) the ``Llama`` handle is loaded
        and cached by a :class:`~dbsprout.train.loader.ModelLoader`, enabling
        restart-free hot-swap between adapters. Without an adapter the original
        S-025 direct-``Llama`` construction path is used unchanged.
        """
        if self._llm is not None:
            return self._llm

        model_path = self._download_model()

        if self._lora_path is not None:
            self._llm = self._load_via_loader(model_path)
            return self._llm

        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError:
            msg = (
                "llama-cpp-python is required for embedded LLM inference. "
                "Install it with: pip install dbsprout[llm]"
            )
            raise ImportError(msg) from None

        logger.info("Loading model from %s", model_path)
        self._llm = Llama(  # pragma: no cover - real model load, never in CI
            model_path=str(model_path),
            n_ctx=_DEFAULT_N_CTX,
            verbose=False,
        )
        return self._llm

    def _load_via_loader(self, model_path: Path) -> Any:
        """Load (or hot-swap) the model+adapter through the ModelLoader.

        Threads the same ``n_ctx`` as the S-025 base path so the LoRA path
        does not overflow the spec prompt, and uses the handle returned by
        ``load()`` directly (no redundant second ``get_handle`` lookup). A
        slow swap (>= the loader's budget) is surfaced as a WARNING so the
        ``<2s`` AC is observable at runtime.
        """
        from dbsprout.train.loader import (  # noqa: PLC0415
            _MAX_SWAP_SECONDS,
            ModelLoader,
        )

        if self._loader is None:
            self._loader = ModelLoader()
        loaded = self._loader.load(model_path, lora_path=self._lora_path, n_ctx=_DEFAULT_N_CTX)
        if loaded.swap_seconds >= _MAX_SWAP_SECONDS:
            logger.warning(
                "LoRA hot-swap took %.2fs (>= %.1fs budget) for adapter %s",
                loaded.swap_seconds,
                _MAX_SWAP_SECONDS,
                self._lora_path,
            )
        return loaded.handle

    def _download_model(self) -> Path:  # pragma: no cover
        """Download the model from Hugging Face if not cached."""
        import pathlib  # noqa: PLC0415

        hf_download = _import_hf_hub_download()
        model_path = hf_download(
            repo_id=self._model_repo,
            filename=self._model_file,
            cache_dir=str(pathlib.Path.home() / ".cache" / "dbsprout" / "models"),
        )
        return pathlib.Path(model_path)

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()


def _import_hf_hub_download() -> Any:
    """Import hf_hub_download with a clear error on missing dependency."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        msg = (
            "huggingface-hub is required for model download. "
            "Install it with: pip install dbsprout[llm]"
        )
        raise ImportError(msg) from None
    return hf_hub_download


_SYSTEM_PROMPT = """You are a database seed data specification generator.
Given a SQL schema (CREATE TABLE statements), produce a JSON DataSpec
that describes how to generate realistic seed data for each table.

For each column, choose an appropriate data provider and parameters.
Common providers: mimesis.Person.email, mimesis.Person.full_name,
mimesis.Address.city, numpy.integers, builtin.autoincrement, builtin.uuid4.

Output valid JSON matching the DataSpec schema exactly."""


def _build_prompt(schema: DatabaseSchema) -> str:
    """Build the LLM prompt from a database schema."""
    ddl = schema.to_ddl()
    return (
        f"Generate a DataSpec JSON for the following database schema:\n\n"
        f"```sql\n{ddl}\n```\n\n"
        f"Include a GeneratorConfig for every column in every table. "
        f"Use appropriate providers for realistic data generation."
    )
