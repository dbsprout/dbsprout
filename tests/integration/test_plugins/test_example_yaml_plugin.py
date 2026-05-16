"""CLI-first walking-skeleton test for the example YAML plugin.

Loads the example plugin through the *real* ``PluginRegistry`` (the same
discovery code path real entry points use) and drives it end-to-end via
``parse_schema_file``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

EXAMPLE_PKG = Path(__file__).resolve().parents[3] / "examples" / "plugin-yaml-parser"
if str(EXAMPLE_PKG) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_PKG))

from dbsprout_plugin_yaml import YamlParser, yaml_parser  # noqa: E402

from dbsprout.plugins.protocols import SchemaParser  # noqa: E402
from dbsprout.plugins.registry import get_registry  # noqa: E402
from dbsprout.schema.parsers import parse_schema_file  # noqa: E402

SCHEMA_YAML = """
dialect: postgresql
tables:
  orgs:
    columns:
      id: {type: integer, primary_key: true}
    primary_key: [id]
"""


@dataclass
class _FakeEP:
    name: str
    group: str
    value: str = "dbsprout_plugin_yaml:yaml_parser"

    def load(self) -> Any:
        # Mirrors the real entry point ``dbsprout_plugin_yaml:yaml_parser``
        # (a module-level instance, like the in-tree adapter parsers).
        return yaml_parser


@contextmanager
def _patched_eps(eps: dict[str, list[_FakeEP]]) -> Iterator[None]:
    def fake(*, group: str | None = None, **_: Any) -> list[_FakeEP]:
        if group is None:
            return [e for v in eps.values() for e in v]
        return list(eps.get(group, []))

    with mock.patch("dbsprout.plugins.discovery.entry_points", side_effect=fake):
        yield


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    get_registry.cache_clear()
    yield
    get_registry.cache_clear()


@pytest.mark.integration
def test_example_satisfies_real_protocol() -> None:
    assert isinstance(YamlParser(), SchemaParser)
    assert isinstance(yaml_parser, SchemaParser)


@pytest.mark.integration
def test_example_discovered_through_real_registry() -> None:
    with _patched_eps({"dbsprout.parsers": [_FakeEP(name="yaml", group="dbsprout.parsers")]}):
        reg = get_registry()
        infos = reg.list("dbsprout.parsers")
        assert any(i.name == "yaml" and i.status == "loaded" for i in infos)
        obj = reg.get("dbsprout.parsers", "yaml")
    assert obj is yaml_parser
    assert isinstance(obj, YamlParser)


@pytest.mark.integration
def test_example_drives_parse_schema_file(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(SCHEMA_YAML, encoding="utf-8")
    with _patched_eps({"dbsprout.parsers": [_FakeEP(name="yaml", group="dbsprout.parsers")]}):
        schema = parse_schema_file(schema_file)
    assert schema.table_names() == ["orgs"]
    assert schema.dialect == "postgresql"
