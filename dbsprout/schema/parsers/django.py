"""Django model introspection parser — reads _meta API to produce DatabaseSchema."""

from __future__ import annotations

import os
from typing import Any

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)

_DJANGO_TYPE_MAP: dict[str, ColumnType] = {
    "AutoField": ColumnType.INTEGER,
    "BigAutoField": ColumnType.BIGINT,
    "SmallAutoField": ColumnType.SMALLINT,
    "IntegerField": ColumnType.INTEGER,
    "BigIntegerField": ColumnType.BIGINT,
    "SmallIntegerField": ColumnType.SMALLINT,
    "PositiveIntegerField": ColumnType.INTEGER,
    "PositiveBigIntegerField": ColumnType.BIGINT,
    "PositiveSmallIntegerField": ColumnType.SMALLINT,
    "FloatField": ColumnType.FLOAT,
    "DecimalField": ColumnType.DECIMAL,
    "BooleanField": ColumnType.BOOLEAN,
    "NullBooleanField": ColumnType.BOOLEAN,
    "CharField": ColumnType.VARCHAR,
    "SlugField": ColumnType.VARCHAR,
    "URLField": ColumnType.VARCHAR,
    "EmailField": ColumnType.VARCHAR,
    "FilePathField": ColumnType.VARCHAR,
    "FileField": ColumnType.VARCHAR,
    "ImageField": ColumnType.VARCHAR,
    "TextField": ColumnType.TEXT,
    "DateField": ColumnType.DATE,
    "DateTimeField": ColumnType.DATETIME,
    "TimeField": ColumnType.TIME,
    "DurationField": ColumnType.BIGINT,
    "UUIDField": ColumnType.UUID,
    "JSONField": ColumnType.JSON,
    "BinaryField": ColumnType.BINARY,
    "IPAddressField": ColumnType.VARCHAR,
    "GenericIPAddressField": ColumnType.VARCHAR,
}

_AUTO_FIELDS: frozenset[str] = frozenset({"AutoField", "BigAutoField", "SmallAutoField"})


def _field_to_column(field: Any) -> ColumnSchema:
    """Convert a Django field (or mock) to a ColumnSchema.

    Reads ``get_internal_type()``, ``column``, ``null``, ``unique``,
    ``primary_key``, ``max_length``, ``choices``, and ``has_default()``
    to build the schema representation.
    """
    internal_type: str = field.get_internal_type()

    data_type = _DJANGO_TYPE_MAP.get(internal_type, ColumnType.UNKNOWN)

    # FK / OneToOne — derive column type from referenced PK
    if internal_type in ("ForeignKey", "OneToOneField") and field.related_model is not None:
        data_type = _fk_column_type(field)

    enum_values: list[str] | None = None
    if field.choices:
        data_type = ColumnType.ENUM
        enum_values = [str(c[0]) for c in field.choices]

    autoincrement = internal_type in _AUTO_FIELDS

    default: str | None = None
    if field.has_default():
        default = str(field.default)

    return ColumnSchema(
        name=field.column,
        data_type=data_type,
        raw_type=internal_type,
        nullable=bool(field.null),
        primary_key=bool(field.primary_key),
        unique=bool(field.unique),
        autoincrement=autoincrement,
        default=default,
        max_length=field.max_length,
        enum_values=enum_values,
    )


def _fk_column_type(field: Any) -> ColumnType:
    """Derive column type from the referenced model's primary key type."""
    ref_internal: str = field.related_model._meta.pk.get_internal_type()
    return _DJANGO_TYPE_MAP.get(ref_internal, ColumnType.INTEGER)


def _fk_to_foreign_key(field: Any) -> ForeignKeySchema:
    """Convert a Django ForeignKey / OneToOneField to a ForeignKeySchema."""
    ref_table: str = field.related_model._meta.db_table
    ref_column: str = field.related_model._meta.pk.column
    on_delete: str = getattr(
        field.remote_field.on_delete, "__name__", str(field.remote_field.on_delete)
    )

    return ForeignKeySchema(
        columns=[field.column],
        ref_table=ref_table,
        ref_columns=[ref_column],
        on_delete=on_delete,
    )


def _m2m_junction_table(field: Any) -> TableSchema | None:
    """Build a junction TableSchema for an auto-created M2M relationship.

    Returns ``None`` when the through model is explicitly defined
    (``auto_created=False``), since the user manages that table themselves.
    """
    if not field.remote_field.through._meta.auto_created:
        return None

    table_name: str = field.m2m_db_table()
    source_col: str = field.m2m_column_name()
    target_col: str = field.m2m_reverse_name()
    source_table: str = field.model._meta.db_table
    target_table: str = field.related_model._meta.db_table

    # Derive FK column types from referenced PKs (e.g., BigAutoField → BIGINT)
    source_pk_type = _DJANGO_TYPE_MAP.get(
        field.model._meta.pk.get_internal_type(), ColumnType.INTEGER
    )
    target_pk_type = _DJANGO_TYPE_MAP.get(
        field.related_model._meta.pk.get_internal_type(), ColumnType.INTEGER
    )

    id_column = ColumnSchema(
        name="id",
        data_type=ColumnType.BIGINT,
        raw_type="BigAutoField",
        nullable=False,
        primary_key=True,
        autoincrement=True,
    )

    source_fk_column = ColumnSchema(
        name=source_col,
        data_type=source_pk_type,
        raw_type="ForeignKey",
        nullable=False,
    )

    target_fk_column = ColumnSchema(
        name=target_col,
        data_type=target_pk_type,
        raw_type="ForeignKey",
        nullable=False,
    )

    source_fk = ForeignKeySchema(
        columns=[source_col],
        ref_table=source_table,
        ref_columns=[field.model._meta.pk.column],
    )

    target_fk = ForeignKeySchema(
        columns=[target_col],
        ref_table=target_table,
        ref_columns=[field.related_model._meta.pk.column],
    )

    return TableSchema(
        name=table_name,
        columns=[id_column, source_fk_column, target_fk_column],
        primary_key=["id"],
        foreign_keys=[source_fk, target_fk],
    )


def _model_to_table(model: Any) -> tuple[TableSchema, list[TableSchema]] | None:
    """Convert a Django model to a TableSchema and associated junction tables.

    Returns ``None`` for abstract or proxy models.
    """
    if model._meta.abstract or model._meta.proxy:
        return None

    columns: list[ColumnSchema] = []
    foreign_keys: list[ForeignKeySchema] = []

    for field in model._meta.local_fields:
        columns.append(_field_to_column(field))
        if field.related_model is not None:
            foreign_keys.append(_fk_to_foreign_key(field))

    junction_tables: list[TableSchema] = []
    for m2m_field in model._meta.local_many_to_many:
        junction = _m2m_junction_table(m2m_field)
        if junction is not None:
            junction_tables.append(junction)

    primary_key = [col.name for col in columns if col.primary_key]

    indexes = [
        IndexSchema(
            name=f"unique_{'_'.join(cols)}",
            columns=list(cols),
            unique=True,
        )
        for cols in model._meta.unique_together
    ]

    table = TableSchema(
        name=model._meta.db_table,
        columns=columns,
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        indexes=indexes,
    )

    return (table, junction_tables)


def parse_django_models(
    app_labels: list[str] | None = None,
) -> DatabaseSchema:
    """Introspect Django models and produce a unified DatabaseSchema.

    Requires ``DJANGO_SETTINGS_MODULE`` to be set. Calls ``django.setup()``
    and iterates ``apps.get_models()`` to build the schema.

    Parameters
    ----------
    app_labels:
        Optional list of Django app labels to filter on. When ``None``,
        all installed models are included.

    Raises
    ------
    RuntimeError
        If ``DJANGO_SETTINGS_MODULE`` is not set or Django is not installed.
    ValueError
        If no concrete models are found.
    """
    settings_module = os.environ.get("DJANGO_SETTINGS_MODULE")
    if not settings_module:
        msg = (
            "DJANGO_SETTINGS_MODULE environment variable is required. "
            "Set it to your settings module (e.g., 'myapp.settings')."
        )
        raise RuntimeError(msg)

    try:
        import django  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        msg = "Django is not installed. Install it with: pip install django"
        raise RuntimeError(msg) from exc

    django.setup()

    from django.apps import apps  # type: ignore[import-not-found]  # noqa: PLC0415

    all_models: list[Any] = list(apps.get_models())

    if app_labels is not None:
        all_models = [m for m in all_models if m._meta.app_label in app_labels]

    tables: list[TableSchema] = []
    for model in all_models:
        result = _model_to_table(model)
        if result is None:
            continue
        main_table, junction_tables = result
        tables.append(main_table)
        tables.extend(junction_tables)

    if not tables:
        msg = "No Django models found."
        raise ValueError(msg)

    return DatabaseSchema(
        tables=tables,
        dialect="postgresql",
        source="django",
    )
