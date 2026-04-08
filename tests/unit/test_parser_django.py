"""Tests for dbsprout.schema.parsers.django — Django model introspection parser."""

from __future__ import annotations

import builtins
import re
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.schema.models import ColumnType
from dbsprout.schema.parsers.django import (
    _field_to_column,
    _fk_to_foreign_key,
    _m2m_junction_table,
    _model_to_table,
    parse_django_models,
)

_SENTINEL = object()


def _mock_field(  # noqa: PLR0913
    internal_type: str = "CharField",
    column: str = "col",
    name: str = "col",
    primary_key: bool = False,
    null: bool = False,
    unique: bool = False,
    max_length: int | None = None,
    choices: list[tuple[str, str]] | None = None,
    default: Any = _SENTINEL,
    has_default: bool = False,
) -> MagicMock:
    field = MagicMock()
    field.get_internal_type.return_value = internal_type
    field.column = column
    field.name = name
    field.primary_key = primary_key
    field.null = null
    field.unique = unique
    field.max_length = max_length
    field.choices = choices
    field.default = default
    field.has_default.return_value = has_default
    field.related_model = None
    return field


def _mock_fk_field(  # noqa: PLR0913
    column: str = "user_id",
    ref_table: str = "auth_user",
    ref_column: str = "id",
    on_delete_name: str = "CASCADE",
    one_to_one: bool = False,
    ref_pk_type: str = "AutoField",
) -> MagicMock:
    field = MagicMock()
    field.get_internal_type.return_value = "OneToOneField" if one_to_one else "ForeignKey"
    field.column = column
    field.name = column.replace("_id", "")
    field.null = False
    field.unique = one_to_one
    field.primary_key = False
    field.max_length = None
    field.choices = None
    field.has_default.return_value = False
    field.related_model = MagicMock()
    field.related_model._meta.db_table = ref_table
    field.related_model._meta.pk.column = ref_column
    field.related_model._meta.pk.get_internal_type.return_value = ref_pk_type
    field.remote_field = MagicMock()
    field.remote_field.on_delete = MagicMock(__name__=on_delete_name)
    return field


def _mock_m2m_field(
    m2m_table: str = "app_user_groups",
    source_col: str = "user_id",
    target_col: str = "group_id",
    source_table: str = "app_user",
    target_table: str = "app_group",
) -> MagicMock:
    field = MagicMock()
    field.m2m_db_table.return_value = m2m_table
    field.m2m_column_name.return_value = source_col
    field.m2m_reverse_name.return_value = target_col
    field.model._meta.db_table = source_table
    field.model._meta.pk.column = "id"
    field.model._meta.pk.get_internal_type.return_value = "AutoField"
    field.related_model._meta.db_table = target_table
    field.related_model._meta.pk.column = "id"
    field.related_model._meta.pk.get_internal_type.return_value = "AutoField"
    field.remote_field.through._meta.auto_created = True
    return field


class TestFieldToColumn:
    """Test _field_to_column for all scalar Django field types."""

    def test_autofield(self) -> None:
        field = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        col = _field_to_column(field)
        assert col.data_type == ColumnType.INTEGER
        assert col.primary_key is True
        assert col.autoincrement is True

    def test_bigautofield(self) -> None:
        field = _mock_field(internal_type="BigAutoField", column="id", primary_key=True)
        col = _field_to_column(field)
        assert col.data_type == ColumnType.BIGINT
        assert col.primary_key is True
        assert col.autoincrement is True

    def test_integerfield(self) -> None:
        field = _mock_field(internal_type="IntegerField", column="age")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.INTEGER

    def test_bigintegerfield(self) -> None:
        field = _mock_field(internal_type="BigIntegerField", column="big_num")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.BIGINT

    def test_smallintegerfield(self) -> None:
        field = _mock_field(internal_type="SmallIntegerField", column="small_num")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.SMALLINT

    def test_floatfield(self) -> None:
        field = _mock_field(internal_type="FloatField", column="score")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.FLOAT

    def test_decimalfield(self) -> None:
        field = _mock_field(internal_type="DecimalField", column="price")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.DECIMAL

    def test_booleanfield(self) -> None:
        field = _mock_field(internal_type="BooleanField", column="is_active")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.BOOLEAN

    def test_charfield_with_max_length(self) -> None:
        field = _mock_field(internal_type="CharField", column="title", max_length=100)
        col = _field_to_column(field)
        assert col.data_type == ColumnType.VARCHAR
        assert col.max_length == 100

    def test_textfield(self) -> None:
        field = _mock_field(internal_type="TextField", column="body")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.TEXT

    def test_datefield(self) -> None:
        field = _mock_field(internal_type="DateField", column="birth_date")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.DATE

    def test_datetimefield(self) -> None:
        field = _mock_field(internal_type="DateTimeField", column="created_at")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.DATETIME

    def test_uuidfield(self) -> None:
        field = _mock_field(internal_type="UUIDField", column="uuid")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.UUID

    def test_jsonfield(self) -> None:
        field = _mock_field(internal_type="JSONField", column="data")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.JSON

    def test_binaryfield(self) -> None:
        field = _mock_field(internal_type="BinaryField", column="blob")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.BINARY

    def test_unknown_field(self) -> None:
        field = _mock_field(internal_type="CustomField", column="custom")
        col = _field_to_column(field)
        assert col.data_type == ColumnType.UNKNOWN

    def test_nullable_field(self) -> None:
        field = _mock_field(internal_type="CharField", column="bio", null=True)
        col = _field_to_column(field)
        assert col.nullable is True

    def test_unique_field(self) -> None:
        field = _mock_field(internal_type="CharField", column="slug", unique=True)
        col = _field_to_column(field)
        assert col.unique is True

    def test_choices_mapped_to_enum(self) -> None:
        field = _mock_field(
            internal_type="CharField",
            column="status",
            choices=[("A", "Alpha"), ("B", "Beta")],
        )
        col = _field_to_column(field)
        assert col.data_type == ColumnType.ENUM
        assert col.enum_values == ["A", "B"]

    def test_default_value(self) -> None:
        field = _mock_field(
            internal_type="CharField",
            column="greeting",
            has_default=True,
            default="hello",
        )
        col = _field_to_column(field)
        assert col.default == "hello"


class TestFKMapping:
    """Test _fk_to_foreign_key for FK / OneToOne fields."""

    def test_fk_produces_foreign_key_schema(self) -> None:
        field = _mock_fk_field(
            column="user_id",
            ref_table="auth_user",
            ref_column="id",
            on_delete_name="CASCADE",
        )
        fk = _fk_to_foreign_key(field)
        assert fk.columns == ["user_id"]
        assert fk.ref_table == "auth_user"
        assert fk.ref_columns == ["id"]
        assert fk.on_delete == "CASCADE"

    def test_fk_on_delete_set_null(self) -> None:
        field = _mock_fk_field(on_delete_name="SET_NULL")
        fk = _fk_to_foreign_key(field)
        assert fk.on_delete == "SET NULL"

    def test_one_to_one_produces_unique_column(self) -> None:
        field = _mock_fk_field(
            column="profile_id",
            ref_table="auth_profile",
            ref_column="id",
            one_to_one=True,
        )
        col = _field_to_column(field)
        assert col.unique is True

        fk = _fk_to_foreign_key(field)
        assert fk.columns == ["profile_id"]
        assert fk.ref_table == "auth_profile"


class TestM2MJunctionTable:
    """Test _m2m_junction_table for M2M fields."""

    def test_m2m_creates_junction_table(self) -> None:
        field = _mock_m2m_field()
        table = _m2m_junction_table(field)
        assert table is not None
        assert table.name == "app_user_groups"
        col_names = [c.name for c in table.columns]
        assert "user_id" in col_names
        assert "group_id" in col_names

    def test_m2m_junction_has_correct_fks(self) -> None:
        field = _mock_m2m_field()
        table = _m2m_junction_table(field)
        assert table is not None
        assert len(table.foreign_keys) == 2
        ref_tables = {fk.ref_table for fk in table.foreign_keys}
        assert "app_user" in ref_tables
        assert "app_group" in ref_tables

    def test_m2m_explicit_through_skipped(self) -> None:
        field = _mock_m2m_field()
        field.remote_field.through._meta.auto_created = False
        table = _m2m_junction_table(field)
        assert table is None


def _mock_model(  # noqa: PLR0913
    db_table: str = "app_user",
    abstract: bool = False,
    proxy: bool = False,
    fields: list[MagicMock] | None = None,
    m2m_fields: list[MagicMock] | None = None,
    pk_field: MagicMock | None = None,
    unique_together: tuple[tuple[str, ...], ...] = (),
) -> MagicMock:
    """Build a mock Django model with _meta populated."""
    model = MagicMock()
    model._meta.db_table = db_table
    model._meta.abstract = abstract
    model._meta.proxy = proxy
    model._meta.local_fields = fields or []
    model._meta.local_many_to_many = m2m_fields or []
    model._meta.unique_together = unique_together
    # Set PK
    if pk_field is None and fields:
        for f in fields:
            if f.primary_key:
                pk_field = f
                break
    model._meta.pk = pk_field
    return model


class TestModelToTable:
    """Test _model_to_table for converting Django models to TableSchema."""

    def test_basic_model_produces_table_schema(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        name_field = _mock_field(
            internal_type="CharField",
            column="name",
            max_length=100,
        )
        model = _mock_model(fields=[pk, name_field])
        result = _model_to_table(model)
        assert result is not None
        table, junctions = result
        assert table.name == "app_user"
        assert len(table.columns) == 2
        col_names = [c.name for c in table.columns]
        assert "id" in col_names
        assert "name" in col_names
        assert junctions == []

    def test_table_name_from_db_table(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        model = _mock_model(db_table="custom_table", fields=[pk])
        result = _model_to_table(model)
        assert result is not None
        table, _ = result
        assert table.name == "custom_table"

    def test_primary_key_extracted(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        other = _mock_field(internal_type="CharField", column="name", max_length=50)
        model = _mock_model(fields=[pk, other])
        result = _model_to_table(model)
        assert result is not None
        table, _ = result
        assert table.primary_key == ["id"]

    def test_fk_field_produces_foreign_key(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        fk = _mock_fk_field(column="org_id", ref_table="app_org", ref_column="id")
        model = _mock_model(fields=[pk, fk])
        result = _model_to_table(model)
        assert result is not None
        table, _ = result
        assert len(table.foreign_keys) == 1
        assert table.foreign_keys[0].ref_table == "app_org"
        assert table.foreign_keys[0].columns == ["org_id"]

    def test_m2m_produces_junction_tables(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        m2m = _mock_m2m_field(
            m2m_table="app_user_groups",
            source_col="user_id",
            target_col="group_id",
        )
        model = _mock_model(fields=[pk], m2m_fields=[m2m])
        result = _model_to_table(model)
        assert result is not None
        _, junctions = result
        assert len(junctions) == 1
        assert junctions[0].name == "app_user_groups"

    def test_unique_together(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        col_a = _mock_field(internal_type="CharField", column="col_a", max_length=50)
        col_b = _mock_field(internal_type="CharField", column="col_b", max_length=50)
        model = _mock_model(
            fields=[pk, col_a, col_b],
            unique_together=(("col_a", "col_b"),),
        )
        result = _model_to_table(model)
        assert result is not None
        table, _ = result
        assert len(table.indexes) == 1
        idx = table.indexes[0]
        assert idx.unique is True
        assert idx.columns == ["col_a", "col_b"]


class TestModelFiltering:
    """Test that abstract and proxy models are filtered out."""

    def test_abstract_model_returns_none(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        model = _mock_model(abstract=True, fields=[pk])
        result = _model_to_table(model)
        assert result is None

    def test_proxy_model_returns_none(self) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        model = _mock_model(proxy=True, fields=[pk])
        result = _model_to_table(model)
        assert result is None


@pytest.fixture
def django_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set DJANGO_SETTINGS_MODULE for parse_django_models tests."""
    monkeypatch.setenv("DJANGO_SETTINGS_MODULE", "myproject.settings")


def _make_concrete_model(
    db_table: str = "app_user",
    app_label: str = "myapp",
) -> MagicMock:
    """Build a concrete mock model with a PK field for orchestrator tests."""
    pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
    name_field = _mock_field(
        internal_type="CharField",
        column="name",
        max_length=100,
    )
    model = _mock_model(db_table=db_table, fields=[pk, name_field])
    model._meta.app_label = app_label
    return model


@pytest.mark.usefixtures("django_env")
class TestParseDjangoModels:
    """Test parse_django_models orchestrator."""

    def test_returns_database_schema(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_django = MagicMock()
        mock_apps_module = MagicMock()
        model_a = _make_concrete_model(db_table="app_user")
        model_b = _make_concrete_model(db_table="app_post", app_label="blog")
        mock_apps_module.apps.get_models.return_value = [model_a, model_b]

        monkeypatch.setitem(sys.modules, "django", mock_django)
        monkeypatch.setitem(sys.modules, "django.apps", mock_apps_module)

        result = parse_django_models()
        assert result.source == "django"
        assert len(result.tables) == 2
        table_names = {t.name for t in result.tables}
        assert "app_user" in table_names
        assert "app_post" in table_names

    def test_includes_junction_tables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        m2m = _mock_m2m_field(m2m_table="app_user_tags")
        model = _mock_model(db_table="app_user", fields=[pk], m2m_fields=[m2m])
        model._meta.app_label = "myapp"

        mock_django = MagicMock()
        mock_apps_module = MagicMock()
        mock_apps_module.apps.get_models.return_value = [model]

        monkeypatch.setitem(sys.modules, "django", mock_django)
        monkeypatch.setitem(sys.modules, "django.apps", mock_apps_module)

        result = parse_django_models()
        table_names = {t.name for t in result.tables}
        assert "app_user" in table_names
        assert "app_user_tags" in table_names

    def test_skips_abstract_and_proxy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        concrete = _make_concrete_model(db_table="app_user")

        pk = _mock_field(internal_type="AutoField", column="id", primary_key=True)
        abstract_model = _mock_model(abstract=True, fields=[pk])
        abstract_model._meta.app_label = "myapp"

        proxy_model = _mock_model(proxy=True, fields=[pk])
        proxy_model._meta.app_label = "myapp"

        mock_django = MagicMock()
        mock_apps_module = MagicMock()
        mock_apps_module.apps.get_models.return_value = [
            concrete,
            abstract_model,
            proxy_model,
        ]

        monkeypatch.setitem(sys.modules, "django", mock_django)
        monkeypatch.setitem(sys.modules, "django.apps", mock_apps_module)

        result = parse_django_models()
        assert len(result.tables) == 1
        assert result.tables[0].name == "app_user"

    def test_app_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_a = _make_concrete_model(db_table="app_user", app_label="myapp")
        model_b = _make_concrete_model(db_table="blog_post", app_label="blog")
        model_c = _make_concrete_model(db_table="admin_log", app_label="admin")

        mock_django = MagicMock()
        mock_apps_module = MagicMock()
        mock_apps_module.apps.get_models.return_value = [
            model_a,
            model_b,
            model_c,
        ]

        monkeypatch.setitem(sys.modules, "django", mock_django)
        monkeypatch.setitem(sys.modules, "django.apps", mock_apps_module)

        result = parse_django_models(app_labels=["myapp"])
        assert len(result.tables) == 1
        assert result.tables[0].name == "app_user"

    def test_raises_without_settings_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
        with pytest.raises(RuntimeError, match="DJANGO_SETTINGS_MODULE"):
            parse_django_models()

    def test_raises_when_django_not_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delitem(sys.modules, "django", raising=False)
        monkeypatch.delitem(sys.modules, "django.apps", raising=False)

        original_import = builtins.__import__

        def _block_django(
            name: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if name == "django":
                raise ImportError("No module named 'django'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_django)
        with pytest.raises(RuntimeError, match="Django is not installed"):
            parse_django_models()

    def test_empty_models_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_django = MagicMock()
        mock_apps_module = MagicMock()
        mock_apps_module.apps.get_models.return_value = []

        monkeypatch.setitem(sys.modules, "django", mock_django)
        monkeypatch.setitem(sys.modules, "django.apps", mock_apps_module)

        with pytest.raises(ValueError, match="No Django models found"):
            parse_django_models()


# ── CLI integration tests for --django / --django-apps ──────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for assertion matching."""
    return _ANSI_RE.sub("", text)


class TestInitDjangoCLI:
    """Test CLI wiring for --django and --django-apps flags."""

    def test_django_and_db_mutual_exclusive(self) -> None:
        result = _runner.invoke(app, ["init", "--django", "--db", "sqlite:///test.db"])
        assert result.exit_code == 1
        out = _strip_ansi(result.output).lower()
        assert "only one" in out or "not both" in out

    def test_django_and_file_mutual_exclusive(self) -> None:
        result = _runner.invoke(app, ["init", "--django", "--file", "test.sql"])
        assert result.exit_code == 1
        out = _strip_ansi(result.output).lower()
        assert "only one" in out or "not both" in out

    @patch("dbsprout.schema.parsers.django.parse_django_models")
    def test_django_error_displayed(self, mock_parse: MagicMock) -> None:
        mock_parse.side_effect = RuntimeError("DJANGO_SETTINGS_MODULE required")
        result = _runner.invoke(app, ["init", "--django"])
        assert result.exit_code == 1
        out = _strip_ansi(result.output)
        assert "DJANGO_SETTINGS_MODULE" in out

    def test_help_shows_django_flags(self) -> None:
        result = _runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.output)
        assert "--django" in out
        assert "--django-apps" in out
