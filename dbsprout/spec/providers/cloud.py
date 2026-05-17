"""Cloud LLM provider — spec generation via LiteLLM + Instructor.

Uses LiteLLM for provider-agnostic API calls (OpenAI, Anthropic,
Google, etc.) and Instructor for Pydantic-validated structured output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec
from dbsprout.spec.providers.base import SpecUsage

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
        self._last_usage: SpecUsage | None = None

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from a schema using cloud LLM.

        Checks cache first. On miss, calls LLM API with Instructor
        for Pydantic-validated output. A cache hit performs no real
        call, so :meth:`get_last_usage` is reset to ``None``.
        """
        self._last_usage = None
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

    def get_last_usage(self) -> SpecUsage | None:
        """Token/cost accounting for the most recent real API call.

        Returns ``None`` after a cache hit or before any real call
        (S-080a). Cloud usage comes from the litellm completion usage
        object and ``litellm.completion_cost``.
        """
        return self._last_usage

    def _call_llm(self, schema: DatabaseSchema) -> DataSpec:
        """Call cloud LLM via LiteLLM + Instructor.

        Requires ``litellm`` and ``instructor`` packages. Captures the
        real token usage + cost from the raw completion (S-080a).
        """
        try:
            import instructor  # noqa: PLC0415
            import litellm  # noqa: PLC0415
        except ImportError:
            msg = (
                "litellm and instructor are required for cloud LLM. "
                "Install it with: pip install dbsprout[cloud]"
            )
            raise ImportError(msg) from None

        client = instructor.from_litellm(litellm.completion)
        prompt = _build_cloud_prompt(schema)

        result: DataSpec
        raw: Any
        result, raw = client.chat.completions.create_with_completion(
            model=self._model,
            response_model=DataSpec,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_retries=_MAX_RETRIES,
        )
        self._last_usage = _usage_from_completion(litellm, raw)
        return result

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()


def _usage_from_completion(litellm: Any, raw: Any) -> SpecUsage:
    """Build :class:`SpecUsage` from a litellm raw completion.

    Real symbols (verified against litellm/instructor upstream):

    * ``raw.usage.prompt_tokens`` / ``.completion_tokens``
    * ``litellm.completion_cost(completion_response=raw)``
    * ``raw._hidden_params["response_cost"]`` (fallback)

    Any missing/unparseable value degrades to ``0`` — never fabricated.
    """
    usage = getattr(raw, "usage", None)
    tokens_sent = _as_int(getattr(usage, "prompt_tokens", 0))
    tokens_received = _as_int(getattr(usage, "completion_tokens", 0))
    return SpecUsage(
        tokens_sent=tokens_sent,
        tokens_received=tokens_received,
        cost_usd=_cost_from_completion(litellm, raw),
    )


def _cost_from_completion(litellm: Any, raw: Any) -> float:
    """Resolve USD cost: ``completion_cost`` first, then hidden params, else 0."""
    try:
        cost = litellm.completion_cost(completion_response=raw)
        if cost is not None:
            return float(cost)
    except Exception as exc:  # cost is best-effort telemetry — never fatal
        logger.debug("litellm.completion_cost failed (%s); trying hidden params", exc)

    hidden = getattr(raw, "_hidden_params", None)
    if isinstance(hidden, dict):
        response_cost = hidden.get("response_cost")
        if response_cost is not None:
            try:
                return float(response_cost)
            except (TypeError, ValueError):
                logger.debug("response_cost not a float (%r)", response_cost)
    return 0.0


def _as_int(value: Any) -> int:
    """Coerce a token count to ``int``; ``0`` if not coercible (honest)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
