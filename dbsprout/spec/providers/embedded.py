"""Embedded LLM provider — local inference via llama-cpp-python.

Runs Qwen2.5-1.5B (or another GGUF model) locally with GBNF grammar
constraints for guaranteed valid JSON output. No API keys needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

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
    ) -> None:
        from dbsprout.spec.cache import SpecCache  # noqa: PLC0415

        self._cache = SpecCache(cache_dir=cache_dir)
        self._model_repo = model_repo
        self._model_file = model_file
        self._llm: Any = None

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
                "Install with: pip install dbsprout[llm]"
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

    def _ensure_llm(self) -> Any:  # pragma: no cover
        """Lazy-load the LLM model. Requires llama-cpp-python."""
        if self._llm is not None:
            return self._llm

        model_path = self._download_model()

        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError:
            msg = (
                "llama-cpp-python is required for embedded LLM inference. "
                "Install with: pip install dbsprout[llm]"
            )
            raise ImportError(msg) from None

        logger.info("Loading model from %s", model_path)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=_DEFAULT_N_CTX,
            verbose=False,
        )
        return self._llm

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
            "Install with: pip install dbsprout[llm]"
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
