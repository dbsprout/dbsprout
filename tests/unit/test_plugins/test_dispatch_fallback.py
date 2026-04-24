"""Tests: registry-first dispatch in parse_schema_file + hard-wired fallback."""

from __future__ import annotations

from typing import ClassVar

import pytest

from dbsprout.plugins.registry import get_registry
from dbsprout.schema.models import DatabaseSchema
from dbsprout.schema.parsers import parse_schema_file


class _ParserCounts:
    """Parser adapter that records how many times its ``parse`` is called."""

    suffixes = (".fake",)
    calls = 0

    def can_parse(self, text: str) -> bool:
        return True

    def parse(self, text: str, *, source_file: str | None = None) -> DatabaseSchema:
        type(self).calls += 1
        return DatabaseSchema(tables=[], dialect="postgresql")


@pytest.fixture(autouse=True)
def _reset():
    get_registry.cache_clear()
    _ParserCounts.calls = 0
    yield
    get_registry.cache_clear()


def test_registry_parser_is_used_for_new_suffix(tmp_path, make_ep, patched_eps):
    with patched_eps(
        {"dbsprout.parsers": [make_ep(name="fake", group="dbsprout.parsers", obj=_ParserCounts())]}
    ):
        fake = tmp_path / "schema.fake"
        fake.write_text("ignored")
        parse_schema_file(fake)
    assert _ParserCounts.calls == 1


def test_registry_empty_falls_back_to_hardwired(tmp_path, patched_eps):
    # Empty registry — built-in SQL DDL fallback still works.
    with patched_eps({}):
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY);")
        schema = parse_schema_file(sql_file)
    assert any(t.name == "users" for t in schema.tables)


def test_registry_writer_is_used(make_ep, patched_eps):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    class FakeWriter:
        format = "fake"
        written: ClassVar[dict] = {}

        def write(self, rows, *, schema, output_dir, dialect):
            FakeWriter.written = {"rows": rows, "dialect": dialect}

    with patched_eps(
        {"dbsprout.outputs": [make_ep(name="fake", group="dbsprout.outputs", obj=FakeWriter())]}
    ):
        writer = _resolve_writer("fake")
    assert writer.__class__.__name__ == "FakeWriter"


def test_registry_engine_is_used(make_ep, patched_eps):
    from dbsprout.cli.commands.generate import _resolve_engine  # noqa: PLC0415

    class FakeEngine:
        def generate_table(self, table, *, rows, spec=None):
            return []

    with patched_eps(
        {
            "dbsprout.generators": [
                make_ep(name="fake", group="dbsprout.generators", obj=FakeEngine())
            ]
        }
    ):
        engine = _resolve_engine("fake", seed=42)
    assert engine.__class__.__name__ == "FakeEngine"
