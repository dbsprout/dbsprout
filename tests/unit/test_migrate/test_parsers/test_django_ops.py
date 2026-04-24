from __future__ import annotations

import pytest

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.migrate.parsers.django import DjangoMigrationParser


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(DjangoMigrationParser(), MigrationParser)

    def test_frozen_dataclass(self) -> None:
        parser = DjangoMigrationParser()
        with pytest.raises((AttributeError, TypeError)):
            parser.foo = "bar"  # type: ignore[attr-defined]
