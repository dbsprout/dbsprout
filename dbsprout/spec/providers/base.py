"""SpecProvider protocol — interface for all spec generation providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.spec.models import DataSpec


class SpecUsage(BaseModel):
    """Token/cost accounting for a single spec-generation LLM call (S-080a).

    Surfaced as an opt-in side-channel: a provider that performs a real LLM
    call records the measured usage so the state layer can persist honest
    telemetry. Defaults are deliberately zero — a provider that cannot
    measure a value leaves it at ``0`` rather than fabricating a number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens_sent: int = 0
    tokens_received: int = 0
    cost_usd: float = 0.0


@runtime_checkable
class SpecProvider(Protocol):  # pragma: no cover
    """Protocol for spec generation providers (embedded, cloud, etc.).

    Intentionally unchanged across S-080a: ``generate_spec`` is the only
    required member so every existing provider, the plugin registry, and
    ``isinstance(obj, SpecProvider)`` checks keep working untouched. Token
    accounting is an opt-in capability — see :class:`UsageReportingProvider`.
    """

    def generate_spec(self, schema: DatabaseSchema) -> DataSpec:
        """Generate a DataSpec from a database schema.

        Implementations may use LLM inference, heuristics, or other
        strategies. Results should be cached via SpecCache.
        """
        ...


@runtime_checkable
class UsageReportingProvider(Protocol):  # pragma: no cover
    """Opt-in capability: a provider that can report its last LLM usage.

    Additive to :class:`SpecProvider`. Implementations that cannot measure
    usage (heuristic fallback, plugin providers predating S-080a) simply do
    not implement this; consumers detect it with ``isinstance(provider,
    UsageReportingProvider)`` or read it defensively, e.g.
    ``getattr(provider, "get_last_usage", lambda: None)()``.
    """

    def get_last_usage(self) -> SpecUsage | None:
        """Token/cost accounting for the most recent real LLM call.

        Returns ``None`` when the last call was served from cache or no
        real LLM call has happened yet.
        """
        ...


def read_usage(provider: object) -> SpecUsage | None:
    """Defensively read a provider's last usage without breaking back-compat.

    Returns the provider's :class:`SpecUsage` if it exposes a callable
    ``get_last_usage``; otherwise ``None``. Never raises for a provider that
    predates S-080a.
    """
    getter = getattr(provider, "get_last_usage", None)
    if not callable(getter):
        return None
    usage = getter()
    return usage if isinstance(usage, SpecUsage) else None
