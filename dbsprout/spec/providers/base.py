"""SpecProvider protocol — interface for all spec generation providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import DataSpec


@runtime_checkable
class SpecProvider(Protocol):  # pragma: no cover
    """Protocol for spec generation providers (embedded, cloud, etc.)."""

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from a database schema.

        Implementations may use LLM inference, heuristics, or other
        strategies. Results should be cached via SpecCache.
        """
        ...
