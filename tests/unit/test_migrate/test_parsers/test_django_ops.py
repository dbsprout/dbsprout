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


class TestAddField:
    def test_add_plain_column(self, tmp_path: Path) -> None:
        create = "migrations.CreateModel(name='Post', fields=[('id', models.AutoField())]),"
        add = (
            "migrations.AddField(model_name='Post', name='title',"
            " field=models.CharField(max_length=200)),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_add_title", _mig(add)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        added = [c for c in changes if c.change_type is SchemaChangeType.COLUMN_ADDED]
        assert len(added) == 1
        assert added[0].table_name == "blog_post"
        assert added[0].column_name == "title"
        assert "CharField" in added[0].detail["django_type"]
        assert added[0].detail["nullable"] is False

    def test_add_foreign_key_emits_both_changes(self, tmp_path: Path) -> None:
        create_user = "migrations.CreateModel(name='User', fields=[('id', models.AutoField())]),"
        create_post = "migrations.CreateModel(name='Post', fields=[('id', models.AutoField())]),"
        add_fk = (
            "migrations.AddField("
            "model_name='Post', name='author', "
            "field=models.ForeignKey('accounts.User', on_delete=models.CASCADE)"
            "),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "accounts": [("0001_initial", _mig(create_user))],
                "blog": [
                    (
                        "0001_initial",
                        _mig(create_post).replace(
                            "dependencies = []",
                            "dependencies = [('accounts', '0001_initial')]",
                        ),
                    ),
                    (
                        "0002_add_author",
                        _mig(add_fk).replace(
                            "dependencies = []",
                            "dependencies = [('blog', '0001_initial'),"
                            " ('accounts', '0001_initial')]",
                        ),
                    ),
                ],
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.COLUMN_ADDED in kinds
        assert SchemaChangeType.FOREIGN_KEY_ADDED in kinds
        fk = next(c for c in changes if c.change_type is SchemaChangeType.FOREIGN_KEY_ADDED)
        assert fk.detail["ref_table"] == "accounts_user"

    def test_add_m2m_emits_through_table(self, tmp_path: Path) -> None:
        create_tag = "migrations.CreateModel(name='Tag', fields=[('id', models.AutoField())]),"
        create_post = "migrations.CreateModel(name='Post', fields=[('id', models.AutoField())]),"
        add_m2m = (
            "migrations.AddField("
            "model_name='Post', name='tags', "
            "field=models.ManyToManyField('Tag')"
            "),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create_tag + "\n" + create_post)),
                    (
                        "0002_tags",
                        _mig(add_m2m).replace(
                            "dependencies = []",
                            "dependencies = [('blog', '0001_initial')]",
                        ),
                    ),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        through = [
            c
            for c in changes
            if c.change_type is SchemaChangeType.TABLE_ADDED and c.table_name == "blog_post_tags"
        ]
        assert len(through) == 1
        fks = through[0].detail["foreign_keys"]
        ref_tables = sorted(fk["ref_table"] for fk in fks)
        assert ref_tables == ["blog_post", "blog_tag"]


class TestRemoveField:
    def test_remove_plain_column(self, tmp_path: Path) -> None:
        create = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()), ('subtitle', models.CharField(max_length=100))"
            "]),"
        )
        remove = "migrations.RemoveField(model_name='Post', name='subtitle'),"
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_drop", _mig(remove)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        removed = [c for c in changes if c.change_type is SchemaChangeType.COLUMN_REMOVED]
        assert len(removed) == 1
        assert removed[0].table_name == "blog_post"
        assert removed[0].column_name == "subtitle"

    def test_remove_fk_emits_fk_removed(self, tmp_path: Path) -> None:
        create_user = "migrations.CreateModel(name='User', fields=[('id', models.AutoField())]),"
        create_post = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()),"
            "('author', models.ForeignKey('accounts.User', on_delete=models.CASCADE))"
            "]),"
        )
        remove = "migrations.RemoveField(model_name='Post', name='author'),"
        root = build_django_project(
            tmp_path,
            apps={
                "accounts": [("0001_initial", _mig(create_user))],
                "blog": [
                    (
                        "0001_initial",
                        _mig(create_post).replace(
                            "dependencies = []",
                            "dependencies = [('accounts', '0001_initial')]",
                        ),
                    ),
                    (
                        "0002_drop",
                        _mig(remove).replace(
                            "dependencies = []",
                            "dependencies = [('blog', '0001_initial')]",
                        ),
                    ),
                ],
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        kinds = [c.change_type for c in changes]
        assert SchemaChangeType.FOREIGN_KEY_REMOVED in kinds


class TestAlterField:
    def test_alter_type_only(self, tmp_path: Path) -> None:
        create = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()), ('n', models.IntegerField())]),"
        )
        alter = (
            "migrations.AlterField(model_name='Post', name='n', field=models.BigIntegerField()),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_alter", _mig(alter)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        altered = [c for c in changes if c.change_type is SchemaChangeType.COLUMN_TYPE_CHANGED]
        assert len(altered) == 1
        assert "BigIntegerField" in altered[0].new_value

    def test_alter_multi_dimension_emits_multiple(self, tmp_path: Path) -> None:
        create = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()), ('n', models.IntegerField())]),"
        )
        alter = (
            "migrations.AlterField(model_name='Post', name='n',"
            " field=models.BigIntegerField(null=True, default=0)),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_alter", _mig(alter)),
                ]
            },
        )
        kinds = [c.change_type for c in DjangoMigrationParser().detect_changes(root)]
        assert SchemaChangeType.COLUMN_TYPE_CHANGED in kinds
        assert SchemaChangeType.COLUMN_NULLABILITY_CHANGED in kinds
        assert SchemaChangeType.COLUMN_DEFAULT_CHANGED in kinds

    def test_alter_without_prior_state_emits_single_trigger(self, tmp_path: Path) -> None:
        alter = (
            "migrations.AlterField(model_name='Post', name='n', field=models.BigIntegerField()),"
        )
        root = build_django_project(tmp_path, apps={"blog": [("0001_alter", _mig(alter))]})
        changes = DjangoMigrationParser().detect_changes(root)
        assert len(changes) == 1
        assert changes[0].change_type is SchemaChangeType.COLUMN_TYPE_CHANGED


class TestRenames:
    def test_rename_field(self, tmp_path: Path) -> None:
        create = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()), ('old', models.CharField())]),"
        )
        rename = "migrations.RenameField(model_name='Post', old_name='old', new_name='new'),"
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_rename", _mig(rename)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        pair = [c for c in changes if c.detail and c.detail.get("rename_of")]
        assert len(pair) == 2
        assert {c.change_type for c in pair} == {
            SchemaChangeType.COLUMN_REMOVED,
            SchemaChangeType.COLUMN_ADDED,
        }
        assert pair[0].detail["rename_of"] == {"old": "old", "new": "new"}

    def test_rename_model(self, tmp_path: Path) -> None:
        create = "migrations.CreateModel(name='OldPost', fields=[('id', models.AutoField())]),"
        rename = "migrations.RenameModel(old_name='OldPost', new_name='Post'),"
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_rename", _mig(rename)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        pair = [c for c in changes if c.detail and c.detail.get("rename_of")]
        assert len(pair) == 2
        assert {c.change_type for c in pair} == {
            SchemaChangeType.TABLE_REMOVED,
            SchemaChangeType.TABLE_ADDED,
        }
        assert {c.table_name for c in pair} == {"blog_oldpost", "blog_post"}


class TestIndexes:
    def test_add_index(self, tmp_path: Path) -> None:
        create = (
            "migrations.CreateModel(name='Post', fields=["
            "('id', models.AutoField()), ('title', models.CharField())]),"
        )
        add_idx = (
            "migrations.AddIndex(model_name='Post',"
            " index=models.Index(fields=['title'], name='idx_post_title')),"
        )
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_idx", _mig(add_idx)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        idx = [c for c in changes if c.change_type is SchemaChangeType.INDEX_ADDED]
        assert len(idx) == 1
        assert idx[0].detail["cols"] == ["title"]
        assert idx[0].detail["index_name"] == "idx_post_title"

    def test_remove_index(self, tmp_path: Path) -> None:
        create = "migrations.CreateModel(name='Post', fields=[('id', models.AutoField())]),"
        rm_idx = "migrations.RemoveIndex(model_name='Post', name='idx_post_title'),"
        root = build_django_project(
            tmp_path,
            apps={
                "blog": [
                    ("0001_initial", _mig(create)),
                    ("0002_rm", _mig(rm_idx)),
                ]
            },
        )
        changes = DjangoMigrationParser().detect_changes(root)
        idx = [c for c in changes if c.change_type is SchemaChangeType.INDEX_REMOVED]
        assert len(idx) == 1
        assert idx[0].detail["index_name"] == "idx_post_title"
