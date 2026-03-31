"""Schema parsers — DDL, DBML, and other file format parsers."""

from __future__ import annotations

from dbsprout.schema.parsers.dbml import can_parse_dbml, parse_dbml
from dbsprout.schema.parsers.ddl import parse_ddl
from dbsprout.schema.parsers.mermaid import can_parse_mermaid, parse_mermaid

__all__ = ["can_parse_dbml", "can_parse_mermaid", "parse_dbml", "parse_ddl", "parse_mermaid"]
