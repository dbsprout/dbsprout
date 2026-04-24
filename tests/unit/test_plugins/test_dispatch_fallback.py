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


# ---------------------------------------------------------------------------
# Fallback branch coverage: empty registry → hard-wired parsers/writers/engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "suffix_text"),
    [
        ("schema.dbml", "Table users {\n  id int [pk]\n}\n"),
        ("schema.mermaid", "erDiagram\n  USERS {\n    int id PK\n  }\n"),
        ("schema.puml", "@startuml\nentity users {\n  *id : INTEGER <<PK>>\n}\n@enduml\n"),
        ("schema.prisma", "model users {\n  id Int @id\n}\n"),
    ],
)
def test_fallback_parser_per_suffix(tmp_path, patched_eps, filename, suffix_text):
    """Empty registry → each suffix routes to its hard-wired parser fallback."""
    with patched_eps({}):
        f = tmp_path / filename
        f.write_text(suffix_text)
        schema = parse_schema_file(f)
    assert schema is not None


def test_fallback_parser_skips_registry_entry_with_none_obj(tmp_path, make_ep, patched_eps):
    """Registry entries whose .obj is None are skipped (continue branch)."""
    from dbsprout.plugins.registry import PluginInfo, get_registry  # noqa: PLC0415

    with patched_eps({}):
        get_registry.cache_clear()
        reg = get_registry()
        reg._by_key[("dbsprout.parsers", "__phantom__")] = PluginInfo(
            group="dbsprout.parsers",
            name="__phantom__",
            module="test",
            status="error",
            error="test",
            obj=None,
        )
        f = tmp_path / "schema.sql"
        f.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY);")
        schema = parse_schema_file(f)
    assert any(t.name == "users" for t in schema.tables)


def test_fallback_writer_sql(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    with patched_eps({}):
        writer = _resolve_writer("sql")
    assert writer is not None
    assert writer.__class__.__name__ == "SQLWriter"


def test_fallback_writer_csv(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    with patched_eps({}):
        writer = _resolve_writer("csv")
    assert writer.__class__.__name__ == "CSVWriter"


@pytest.mark.parametrize("fmt", ["json", "jsonl"])
def test_fallback_writer_json(patched_eps, fmt):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    with patched_eps({}):
        writer = _resolve_writer(fmt)
    assert writer.__class__.__name__ == "JSONWriter"


def test_fallback_writer_unknown_returns_none(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    with patched_eps({}):
        writer = _resolve_writer("nonexistent-format")
    assert writer is None


def test_fallback_engine_heuristic(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_engine  # noqa: PLC0415

    with patched_eps({}):
        eng = _resolve_engine("heuristic", seed=42)
    assert eng.__class__.__name__ == "HeuristicEngine"


def test_fallback_engine_spec_driven(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_engine  # noqa: PLC0415

    with patched_eps({}):
        eng = _resolve_engine("spec_driven", seed=42)
    assert eng.__class__.__name__ == "SpecDrivenEngine"


def test_fallback_engine_unknown_returns_none(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_engine  # noqa: PLC0415

    with patched_eps({}):
        eng = _resolve_engine("nonexistent-engine", seed=42)
    assert eng is None


def test_fallback_writer_parquet(patched_eps):
    from dbsprout.cli.commands.generate import _resolve_writer  # noqa: PLC0415

    with patched_eps({}):
        writer = _resolve_writer("parquet")
    assert writer.__class__.__name__ == "ParquetWriter"
