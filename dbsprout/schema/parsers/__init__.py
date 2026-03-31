"""Schema parsers — DDL, DBML, and other file format parsers."""

from __future__ import annotations

from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl

__all__ = ["can_parse_dbml", "parse_dbml", "parse_ddl"]
