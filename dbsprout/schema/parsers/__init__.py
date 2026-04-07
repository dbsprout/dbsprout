"""Schema parsers — DDL, DBML, Mermaid, PlantUML, Prisma file format parsers."""

from __future__ import annotations

from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl
from dbsprout.schema.parsers.django import parse_django_models
from dbsprout.schema.parsers.mermaid import can_parse_mermaid, parse_mermaid
from dbsprout.schema.parsers.plantuml import can_parse_plantuml, parse_plantuml
from dbsprout.schema.parsers.prisma import can_parse_prisma, parse_prisma

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
]
