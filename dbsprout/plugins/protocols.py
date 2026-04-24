"""Protocols that DBSprout plugins must satisfy.

Three new Protocols are defined here; two are re-exported from their
original modules so that the external plugin API lives in a single
package (``dbsprout.plugins``).

All Protocols are ``@runtime_checkable`` so the registry can do an
``isinstance`` smoke test at registration. This check covers attribute
names only, not signatures — signature mismatches will surface at first
call with a normal ``TypeError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.spec.providers.base import SpecProvider

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema, TableSchema
    from dbsprout.spec.models import DataSpec


@runtime_checkable
class SchemaParser(Protocol):
    """Parses a schema source into a :class:`DatabaseSchema`."""

    suffixes: tuple[str, ...]

    def can_parse(self, text: str) -> bool: ...

    def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema: ...


@runtime_checkable
class GenerationEngine(Protocol):
    """Produces rows for a single table."""

    def generate_table(
        self,
        table: TableSchema,
        *,
        rows: int,
        spec: DataSpec | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class OutputWriter(Protocol):
    """Writes generated rows for a full schema."""

    format: str

    def write(
        self,
        rows: dict[str, list[dict[str, Any]]],
        *,
        schema: DatabaseSchema,
        output_dir: Path,
        dialect: str,
    ) -> None: ...


__all__ = [
    "GenerationEngine",
    "MigrationParser",
    "OutputWriter",
    "SchemaParser",
    "SpecProvider",
]
