from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.parsers import MigrationParser
from dbsprout.migrate.parsers.django import DjangoMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import build_django_project

if TYPE_CHECKING:
    from pathlib import Path


EMPTY_MIG = (
    "from django.db import migrations\n\n"
    "class Migration(migrations.Migration):\n"
    "    dependencies = []\n"
    "    operations = []\n"
)


def _mig(body: str) -> str:
    return (
        "from django.db import migrations, models\n\n"
        "class Migration(migrations.Migration):\n"
        "    dependencies = []\n"
        f"    operations = [\n{textwrap.indent(body, '        ')}\n    ]\n"
    )


class TestScaffold:
    def test_implements_protocol(self) -> None:
        assert isinstance(DjangoMigrationParser(), MigrationParser)

    def test_frozen_dataclass(self) -> None:
        parser = DjangoMigrationParser()
        with pytest.raises((AttributeError, TypeError)):
            parser.foo = "bar"  # type: ignore[attr-defined]


class TestEndToEndEmpty:
    def test_empty_ops_returns_empty_list(self, tmp_path: Path) -> None:
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", EMPTY_MIG)]})
        assert DjangoMigrationParser().detect_changes(root) == []
