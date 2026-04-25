"""Protocols that DBSprout plugins must satisfy.

Three new Protocols are defined here; two are re-exported from their
original modules so that the external plugin API lives in a single
package (``dbsprout.plugins``).

All Protocols are ``@runtime_checkable`` so the registry can do an
``isinstance`` smoke test at registration. This check covers attribute
names only, not signatures — signature mismatches will surface at first
call with a normal ``TypeError``.

**Class-level attributes required.** The registry's ``isinstance`` check
only inspects class attributes for class-valued entry points. Plugin
authors must declare Protocol attributes (``suffixes``, ``format``)
at the class level, not inside ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.spec.providers.base import SpecProvider

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema


@runtime_checkable
class SchemaParser(Protocol):
    """Parses a schema source into a :class:`DatabaseSchema`."""

    suffixes: tuple[str, ...]

    def can_parse(self, text: str) -> bool: ...

    def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema: ...


@runtime_checkable
class GenerationEngine(Protocol):
    """Produces rows for a single table.

    The concrete ``generate_table`` signature varies across in-tree
    engines (``HeuristicEngine`` takes a column→generator mapping;
    ``SpecDrivenEngine`` takes a per-table :class:`DataSpec`). Like
    :class:`OutputWriter`, this Protocol declares only the method name
    so the registry can validate plugin shape at registration; callers
    are responsible for invoking the engine with the kwargs it expects.
    """

    def generate_table(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...


@runtime_checkable
class OutputWriter(Protocol):
    """Writes generated rows for a full schema.

    The concrete ``write`` signature varies across writers (e.g. the CSV
    writer accepts ``dialect`` while Parquet does not). Plugins are
    expected to accept the arguments they care about and ignore the
    rest; this Protocol intentionally declares only the attributes and
    methods every writer shares.
    """

    format: str

    def write(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "GenerationEngine",
    "MigrationParser",
    "OutputWriter",
    "SchemaParser",
    "SpecProvider",
]
