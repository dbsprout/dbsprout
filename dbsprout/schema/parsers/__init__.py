"""Schema parsers — DDL, DBML, Prisma file format parsers."""

from __future__ import annotations

from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl
from dbsprout.schema.parsers.prisma import can_parse_prisma, parse_prisma

__all__ = [
    "can_parse_dbml",
    "can_parse_prisma",
    "parse_dbml",
    "parse_ddl",
    "parse_prisma",
]
