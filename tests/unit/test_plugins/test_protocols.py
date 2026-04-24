from pathlib import Path

from dbsprout.plugins.protocols import (
    GenerationEngine,
    MigrationParser,
    OutputWriter,
    SchemaParser,
    SpecProvider,
)


class _GoodParser:
    suffixes = (".foo",)

    def can_parse(self, text: str) -> bool:
        return True

    def parse(self, text: str, *, source_file: str | None = None):
        return object()


class _BadParserMissingMethod:
    suffixes = (".foo",)


def test_schema_parser_isinstance_positive():
    assert isinstance(_GoodParser(), SchemaParser)


def test_schema_parser_isinstance_negative():
    assert not isinstance(_BadParserMissingMethod(), SchemaParser)


class _GoodWriter:
    format = "csv"

    def write(self, rows, *, schema, output_dir: Path, dialect: str) -> None: ...


def test_output_writer_isinstance_positive():
    assert isinstance(_GoodWriter(), OutputWriter)


def test_re_exported_spec_provider_identity():
    from dbsprout.spec.providers.base import SpecProvider as Original  # noqa: PLC0415

    assert SpecProvider is Original


def test_re_exported_migration_parser_identity():
    from dbsprout.migrate.parsers import MigrationParser as Original  # noqa: PLC0415

    assert MigrationParser is Original


class _GoodEngine:
    def generate_table(self, table, *, rows: int, spec=None):
        return []


def test_generation_engine_isinstance_positive():
    assert isinstance(_GoodEngine(), GenerationEngine)
