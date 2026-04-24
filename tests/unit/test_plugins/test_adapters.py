"""Unit tests for dbsprout.plugins.adapters."""

from __future__ import annotations

from dbsprout.plugins.adapters import dbml_parser, ddl_parser
from dbsprout.plugins.protocols import SchemaParser


def test_dbml_adapter_satisfies_protocol() -> None:
    assert isinstance(dbml_parser, SchemaParser)


def test_dbml_adapter_suffixes() -> None:
    assert dbml_parser.suffixes == (".dbml",)


def test_ddl_adapter_parses_simple_create_table() -> None:
    schema = ddl_parser.parse("CREATE TABLE users (id INTEGER PRIMARY KEY);", source_file=None)
    assert any(t.name == "users" for t in schema.tables)


def test_ddl_adapter_suffixes() -> None:
    assert ddl_parser.suffixes == (".sql",)
