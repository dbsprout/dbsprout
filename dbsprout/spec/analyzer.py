"""LLM spec analyzer — orchestrates schema DDL → DataSpec JSON pipeline.

Manages cache, provider calls, retry logic, privacy enforcement,
and heuristic fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.providers.base import SpecProvider

from dbsprout.privacy.enforcer import PrivacyEnforcer, PrivacyTier
from dbsprout.privacy.redactor import de_redact_spec, redact_schema
from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class SpecAnalyzer:
    """Orchestrates the spec generation pipeline.

    Flow: privacy check → cache check → provider call → validate → retry → fallback.
    """

    def __init__(
        self,
        provider: SpecProvider,
        cache_dir: Path | str = ".dbsprout/cache",
        privacy_tier: PrivacyTier = PrivacyTier.LOCAL,
    ) -> None:
        self._provider = provider
        self._cache = SpecCache(cache_dir=cache_dir)
        self._privacy_tier = privacy_tier
        self._enforcer = PrivacyEnforcer()

    def analyze(self, schema: DatabaseSchema) -> DataSpec:
        """Analyze a schema and produce a DataSpec.

        1. Validate privacy tier against provider locality
        2. Check cache (by schema_hash)
        3. On miss: redact schema if ``redacted`` tier, then call provider
        4. On total failure: heuristic fallback
        5. Cache and return
        """
        # Privacy enforcement — blocks cloud providers under local tier
        provider_locality: str = getattr(self._provider, "provider_locality", "cloud")
        self._enforcer.validate_provider(
            provider_locality=provider_locality,
            tier=self._privacy_tier,
        )

        schema_hash = schema.schema_hash()

        # Cache check
        cached = self._cache.get(schema_hash)
        if cached is not None:
            logger.info("Spec cache hit for hash %s", schema_hash)
            return cached

        # Redact schema for cloud providers under redacted tier
        provider_schema = schema
        redaction_map = None
        if self._privacy_tier == PrivacyTier.REDACTED and provider_locality == "cloud":
            logger.info("Redacting schema for redacted tier before cloud call")
            provider_schema, redaction_map = redact_schema(schema)

        # Provider call with retry
        spec = self._call_with_retry(provider_schema)

        # De-redact spec if schema was redacted
        if redaction_map is not None:
            spec = de_redact_spec(spec, redaction_map)

        # Cache result
        spec = spec.model_copy(update={"schema_hash": schema_hash})
        self._cache.put(schema_hash, spec)
        return spec

    def _call_with_retry(self, schema: DatabaseSchema) -> DataSpec:
        """Call provider with retry logic, falling back to heuristics."""
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info("Spec generation attempt %d/%d", attempt, _MAX_RETRIES)
                return self._provider.generate_spec(schema)
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                logger.warning(
                    "Spec generation attempt %d failed: %s",
                    attempt,
                    exc,
                )
                last_error = exc

        # All retries exhausted — heuristic fallback
        logger.warning(
            "All %d spec generation attempts failed (last error: %s). "
            "Falling back to heuristic mapping.",
            _MAX_RETRIES,
            last_error,
        )
        return heuristic_fallback(schema)

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()


def _build_spec_prompt(schema: DatabaseSchema) -> str:
    """Build the LLM prompt from a database schema.

    Includes schema DDL, provider examples, and output format instructions.
    """
    ddl = schema.to_ddl()
    return (
        "Generate a DataSpec JSON for the following database schema.\n\n"
        "## Schema DDL\n\n"
        f"```sql\n{ddl}\n```\n\n"
        "## Instructions\n\n"
        "For each column in each table, produce a GeneratorConfig with:\n"
        "- `provider`: data generator (e.g., mimesis.Person.email, "
        "mimesis.Person.full_name, numpy.integers, builtin.autoincrement, "
        "builtin.uuid4)\n"
        "- `distribution`: for numeric columns (uniform, normal, zipf)\n"
        "- `min_value`/`max_value`: for numeric ranges\n"
        "- `unique`: true for UNIQUE columns\n"
        "- `nullable_rate`: fraction of rows that should be NULL (0.0-1.0)\n\n"
        "## Example\n\n"
        "For a table `products(id INT PK, name VARCHAR, price DECIMAL)`:\n"
        "```json\n"
        "{\n"
        '  "table_name": "products",\n'
        '  "columns": {\n'
        '    "id": {"provider": "builtin.autoincrement"},\n'
        '    "name": {"provider": "mimesis.Text.word"},\n'
        '    "price": {"provider": "numpy.uniform", '
        '"min_value": 1.0, "max_value": 999.99}\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "Output the complete DataSpec JSON with all tables."
    )


def heuristic_fallback(schema: DatabaseSchema) -> DataSpec:
    """Convert heuristic mappings to a DataSpec as a fallback.

    Uses Sprint 2's ``map_columns`` to produce GeneratorMapping objects,
    then translates each to a GeneratorConfig for the DataSpec.
    """
    from dbsprout.spec.heuristics import map_columns  # noqa: PLC0415

    all_mappings = map_columns(schema)
    table_specs: list[TableSpec] = []

    for table in schema.tables:
        mappings = all_mappings.get(table.name, {})
        columns: dict[str, GeneratorConfig] = {}

        for col in table.columns:
            mapping = mappings.get(col.name)
            if mapping is not None:
                columns[col.name] = GeneratorConfig(
                    provider=f"{mapping.provider}.{mapping.generator_name}",
                    params=dict(mapping.params),
                )
            else:
                columns[col.name] = GeneratorConfig(provider="builtin.default")

        table_specs.append(
            TableSpec(table_name=table.name, columns=columns),
        )

    return DataSpec(
        tables=table_specs,
        schema_hash=schema.schema_hash(),
        model_used="heuristic_fallback",
    )
