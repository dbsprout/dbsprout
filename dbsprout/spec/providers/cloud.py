"""Cloud LLM provider — spec generation via LiteLLM + Instructor.

Uses LiteLLM for provider-agnostic API calls (OpenAI, Anthropic,
Google, etc.) and Instructor for Pydantic-validated structured output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_MAX_RETRIES = 3


class CloudProvider:
    """Spec provider using cloud LLM APIs via LiteLLM + Instructor."""

    provider_locality: str = "cloud"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        cache_dir: str = ".dbsprout/cache",
    ) -> None:
        self._model = model
        self._cache = SpecCache(cache_dir=cache_dir)

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from a schema using cloud LLM.

        Checks cache first. On miss, calls LLM API with Instructor
        for Pydantic-validated output.
        """
        schema_hash = schema.schema_hash()

        cached = self._cache.get(schema_hash)
        if cached is not None:
            logger.info("Spec cache hit for hash %s", schema_hash)
            return cached

        logger.info("Spec cache miss — calling %s", self._model)
        spec = self._call_llm(schema)
        spec = spec.model_copy(update={"schema_hash": schema_hash})
        self._cache.put(schema_hash, spec)
        return spec

    def _call_llm(self, schema: DatabaseSchema) -> DataSpec:  # pragma: no cover
        """Call cloud LLM via LiteLLM + Instructor.

        Requires ``litellm`` and ``instructor`` packages.
        """
        try:
            import instructor  # type: ignore[import-not-found]  # noqa: PLC0415
            import litellm  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError:
            msg = (
                "litellm and instructor are required for cloud LLM. "
                "Install with: pip install dbsprout[cloud]"
            )
            raise ImportError(msg) from None

        client = instructor.from_litellm(litellm.completion)
        prompt = _build_cloud_prompt(schema)

        result: DataSpec = client.chat.completions.create(
            model=self._model,
            response_model=DataSpec,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_retries=_MAX_RETRIES,
        )
        return result

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()


_SYSTEM_PROMPT = (
    "You are a database seed data specification generator. "
    "Given SQL DDL, produce a DataSpec JSON describing how to generate "
    "realistic seed data for each table and column."
)


def _build_cloud_prompt(schema: DatabaseSchema) -> str:
    """Build the LLM prompt from a database schema."""
    ddl = schema.to_ddl()
    return (
        "Generate a DataSpec JSON for the following database schema.\n\n"
        f"```sql\n{ddl}\n```\n\n"
        "For each column, choose an appropriate provider "
        "(mimesis.Person.email, numpy.integers, builtin.autoincrement, etc.) "
        "and set distribution parameters where appropriate.\n\n"
        "Return a valid DataSpec with all tables and columns."
    )
