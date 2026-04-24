"""Schema parsers — DDL, DBML, Mermaid, PlantUML, Prisma file format parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl
from dbsprout.schema.parsers.django import parse_django_models
from dbsprout.schema.parsers.mermaid import can_parse_mermaid, parse_mermaid
from dbsprout.schema.parsers.plantuml import can_parse_plantuml, parse_plantuml
from dbsprout.schema.parsers.prisma import can_parse_prisma, parse_prisma

if TYPE_CHECKING:
    from pathlib import Path

    from dbsprout.schema.models import DatabaseSchema

_MAX_SCHEMA_BYTES = 10 * 1024 * 1024  # 10 MB


def parse_schema_file(path: Path) -> DatabaseSchema:
    """Parse a schema file, dispatching on suffix.

    Supports ``.sql`` (DDL), ``.dbml``, ``.mermaid``/``.mmd``,
    ``.puml``/``.plantuml``/``.pu``, ``.prisma``. Unknown suffixes fall
    back to SQL DDL.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: file exceeds 10 MB or a parser fails.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    if path.stat().st_size > _MAX_SCHEMA_BYTES:
        msg = f"File too large (>10 MB): {path}"
        raise ValueError(msg)

    text = path.read_text(encoding="utf-8")
    source = str(path)
    suffix = path.suffix.lower()

    # Registry-first: a plugin whose ``suffixes`` tuple contains this
    # file suffix wins. The fallback below keeps editable-install / CI
    # paths working when entry-point metadata is unavailable.
    from dbsprout.plugins.registry import get_registry  # noqa: PLC0415

    for info in get_registry().list("dbsprout.parsers"):
        obj = info.obj
        if obj is None:
            continue
        suffixes = getattr(obj, "suffixes", ())
        if suffix not in suffixes:
            continue
        if not obj.can_parse(text):
            continue
        return cast("DatabaseSchema", obj.parse(text, source_file=source))

    if suffix == ".dbml":
        return parse_dbml(text, source_file=source)
    if suffix in (".mermaid", ".mmd"):
        return parse_mermaid(text, source_file=source)
    if suffix in (".puml", ".plantuml", ".pu"):
        return parse_plantuml(text, source_file=source)
    if suffix == ".prisma":
        return parse_prisma(text, source_file=source)
    return parse_ddl(text, source_file=source)


__all__ = [
    "can_parse_dbml",
    "can_parse_mermaid",
    "can_parse_plantuml",
    "can_parse_prisma",
    "parse_dbml",
    "parse_ddl",
    "parse_django_models",
    "parse_mermaid",
    "parse_plantuml",
    "parse_prisma",
    "parse_schema_file",
]
