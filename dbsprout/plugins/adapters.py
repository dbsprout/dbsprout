"""Adapters that present function-style parsers as ``SchemaParser`` objects.

The five in-tree file-format parsers (DDL, DBML, Mermaid, PlantUML, Prisma)
are module-level functions. Rather than refactor them into classes, we wrap
each one in a thin adapter so the plugin registry always returns an object
with the same shape.

Note: ``parse_django_models`` is intentionally excluded. Its signature is
``parse_django_models(app_labels: list[str] | None = None)`` — it performs
live Django app introspection and does not accept a ``(text, *, source_file)``
call, so it cannot be wrapped by ``_FnParserAdapter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dbsprout.schema.parsers.dbml import parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl
from dbsprout.schema.parsers.mermaid import parse_mermaid
from dbsprout.schema.parsers.plantuml import parse_plantuml
from dbsprout.schema.parsers.prisma import parse_prisma

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbsprout.schema.models import DatabaseSchema


class _FnParserAdapter:
    """Wrap a ``parse_*`` function so it satisfies ``SchemaParser``."""

    def __init__(
        self,
        fn: Callable[..., DatabaseSchema],
        *,
        suffixes: tuple[str, ...],
    ) -> None:
        self._fn = fn
        self.suffixes = suffixes

    def can_parse(self, text: str) -> bool:  # noqa: ARG002
        return True

    def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema:
        return self._fn(text, source_file=source_file)


ddl_parser = _FnParserAdapter(parse_ddl, suffixes=(".sql",))
dbml_parser = _FnParserAdapter(parse_dbml, suffixes=(".dbml",))
mermaid_parser = _FnParserAdapter(parse_mermaid, suffixes=(".mermaid", ".mmd"))
plantuml_parser = _FnParserAdapter(parse_plantuml, suffixes=(".puml", ".plantuml", ".pu"))
prisma_parser = _FnParserAdapter(parse_prisma, suffixes=(".prisma",))

__all__ = [
    "dbml_parser",
    "ddl_parser",
    "mermaid_parser",
    "plantuml_parser",
    "prisma_parser",
]
