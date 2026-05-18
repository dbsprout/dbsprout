from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from dbsprout_plugin_yaml import YamlParser  # noqa: E402

from dbsprout.plugins.protocols import SchemaParser  # noqa: E402
from dbsprout.schema.models import ColumnType  # noqa: E402

SCHEMA_YAML = """
dialect: postgresql
tables:
  orgs:
    columns:
      id: {type: integer, primary_key: true}
      name: {type: varchar}
    primary_key: [id]
  users:
    columns:
      id: {type: integer, primary_key: true}
      email: {type: varchar, unique: true, nullable: false}
      org_id: {type: integer, nullable: false}
    primary_key: [id]
    foreign_keys:
      - columns: [org_id]
        ref_table: orgs
        ref_columns: [id]
"""


def test_yaml_parser_satisfies_real_protocol() -> None:
    assert isinstance(YamlParser(), SchemaParser)


def test_parse_builds_full_schema() -> None:
    schema = YamlParser().parse(SCHEMA_YAML, source_file="schema.yaml")
    assert schema.dialect == "postgresql"
    assert sorted(schema.table_names()) == ["orgs", "users"]
    users = schema.get_table("users")
    assert users is not None
    assert users.primary_key == ["id"]
    email = users.get_column("email")
    assert email is not None
    assert email.data_type == ColumnType.VARCHAR
    assert email.unique is True
    assert email.nullable is False
    assert users.foreign_keys[0].ref_table == "orgs"
    assert users.foreign_keys[0].columns == ["org_id"]


def test_unknown_type_falls_back_to_unknown() -> None:
    schema = YamlParser().parse(
        "tables:\n  t:\n    columns:\n      c: {type: wat}\n", source_file=None
    )
    table = schema.get_table("t")
    assert table is not None
    col = table.get_column("c")
    assert col is not None
    assert col.data_type == ColumnType.UNKNOWN


def test_can_parse_accepts_schema_yaml() -> None:
    assert YamlParser().can_parse(SCHEMA_YAML) is True


def test_can_parse_rejects_non_schema_yaml() -> None:
    assert YamlParser().can_parse("just: a value\n") is False
    assert YamlParser().can_parse(": : not yaml : :\n[") is False


def test_parse_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        YamlParser().parse("- a\n- b\n", source_file=None)
