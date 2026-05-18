"""Ollama provider — spec generation via locally-hosted Ollama models.

Uses LiteLLM with ``ollama/`` model prefix to route to a local
Ollama instance. Includes health check for early error detection.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "ollama/llama3.2"
_DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider:
    """Spec provider using a local Ollama instance via LiteLLM."""

    provider_locality: str = "local"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str | None = None,
        cache_dir: str = ".dbsprout/cache",
    ) -> None:
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", _DEFAULT_HOST)
        self._cache = SpecCache(cache_dir=cache_dir)

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from schema using local Ollama.

        Checks health, then cache, then calls LLM.
        """
        schema_hash = schema.schema_hash()

        cached = self._cache.get(schema_hash)
        if cached is not None:
            logger.info("Spec cache hit for hash %s", schema_hash)
            return cached

        self._check_health()

        logger.info("Calling Ollama (%s) for spec generation", self._model)
        spec = self._call_llm(schema)
        spec = spec.model_copy(update={"schema_hash": schema_hash})
        self._cache.put(schema_hash, spec)
        return spec

    def _check_health(self) -> None:
        """Verify Ollama is running and reachable."""
        url = f"{self._host}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5):  # noqa: S310  # nosec B310
                pass
        except Exception as exc:
            msg = f"Ollama not reachable at {self._host}. Start Ollama with: ollama serve"
            raise ConnectionError(msg) from exc

    def _call_llm(self, schema: DatabaseSchema) -> DataSpec:  # pragma: no cover
        """Call Ollama via LiteLLM and parse response as DataSpec."""
        try:
            import litellm  # noqa: PLC0415
        except ImportError:
            msg = (
                "litellm is required for Ollama provider. "
                "Install it with: pip install dbsprout[cloud]"
            )
            raise ImportError(msg) from None

        os.environ["OLLAMA_API_BASE"] = self._host

        prompt = _build_ollama_prompt(schema)
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        raw_json = response.choices[0].message.content
        return DataSpec.model_validate_json(raw_json)

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()


_SYSTEM_PROMPT = (
    "You are a database seed data specification generator. "
    "Given SQL DDL, produce valid JSON matching the DataSpec schema. "
    "Output ONLY the JSON, no explanation."
)


def _build_ollama_prompt(schema: DatabaseSchema) -> str:
    """Build the LLM prompt from a database schema."""
    ddl = schema.to_ddl()
    return (
        "Generate a DataSpec JSON for the following database schema.\n\n"
        f"```sql\n{ddl}\n```\n\n"
        "For each column, choose an appropriate provider "
        "(mimesis.Person.email, numpy.integers, builtin.autoincrement, etc.).\n"
        "Return valid JSON only."
    )
