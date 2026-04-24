from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from dbsprout.migrate.models import SchemaChangeType
from dbsprout.migrate.parsers import MigrationParser
from dbsprout.migrate.parsers.django import DjangoMigrationParser
from tests.unit.test_migrate.test_parsers.conftest import assert_change, build_django_project

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


class TestCreateModel:
    def test_basic_create_model(self, tmp_path: Path) -> None:
        body = (
            "migrations.CreateModel(\n"
            "    name='Post',\n"
            "    fields=[\n"
            "        ('id', models.AutoField(primary_key=True)),\n"
            "        ('title', models.CharField(max_length=200)),\n"
            "    ],\n"
            "),"
        )
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", _mig(body))]})
        changes = DjangoMigrationParser().detect_changes(root)
        assert len(changes) == 1
        assert_change(changes[0], change_type=SchemaChangeType.TABLE_ADDED, table_name="blog_post")
        fields = changes[0].detail["fields"]
        names = [f["name"] for f in fields]
        assert names == ["id", "title"]

    def test_create_model_honours_db_table(self, tmp_path: Path) -> None:
        body = (
            "migrations.CreateModel(\n"
            "    name='Post',\n"
            "    fields=[('id', models.AutoField(primary_key=True))],\n"
            "    options={'db_table': 'my_posts'},\n"
            "),"
        )
        root = build_django_project(tmp_path, apps={"blog": [("0001_initial", _mig(body))]})
        changes = DjangoMigrationParser().detect_changes(root)
        assert changes[0].change_type is SchemaChangeType.TABLE_ADDED
        assert changes[0].table_name == "my_posts"
        assert changes[0].detail["db_table"] == "my_posts"


class TestDeleteModel:
    def test_delete_model(self, tmp_path: Path) -> None:
        body1 = "migrations.CreateModel(name='Post', fields=[('id', models.AutoField())]),\n"
        body2 = "migrations.DeleteModel(name='Post'),"
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(body1)),
                    ("0002_delete", _mig(body2)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        assert changes[-1].change_type is SchemaChangeType.TABLE_REMOVED
        assert changes[-1].table_name == "blog_post"
