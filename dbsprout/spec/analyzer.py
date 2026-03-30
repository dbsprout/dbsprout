"""LLM spec analyzer — orchestrates schema DDL → DataSpec JSON pipeline.

Manages cache, provider calls, retry logic, and heuristic fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.providers.base import SpecProvider

from dbsprout.spec.cache import SpecCache
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class SpecAnalyzer:
    """Orchestrates the spec generation pipeline.

    Flow: cache check → provider call → validate → retry → fallback.
    """

    def __init__(
        self,
        provider: SpecProvider,
        cache_dir: Path | str = ".dbsprout/cache",
    ) -> None:
        self._provider = provider
        self._cache = SpecCache(cache_dir=cache_dir)

    def analyze(self, schema: DatabaseSchema) -> DataSpec:
        """Analyze a schema and produce a DataSpec.

        1. Check cache (by schema_hash)
        2. On miss: call provider with retry
        3. On total failure: heuristic fallback
        4. Cache and return
        """
        schema_hash = schema.schema_hash()

        # Cache check
        cached = self._cache.get(schema_hash)
        if cached is not None:
            logger.info("Spec cache hit for hash %s", schema_hash)
            return cached

        # Provider call with retry
        spec = self._call_with_retry(schema)

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
