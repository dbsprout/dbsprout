"""MongoDB schema-inference parser — samples documents to a DatabaseSchema.

MongoDB has no fixed schema, so structure is *inferred* by sampling
documents. This module wraps the battle-tested ``pymongo-schema`` library
(``extract_pymongo_client_schema``) and translates its nested per-field
dictionary into the unified, immutable :class:`DatabaseSchema` model so the
rest of the pipeline (spec → generation → output) is unchanged.

Connection-based, not file-based: it takes a ``mongodb://`` /
``mongodb+srv://`` URI and connects via pymongo, mirroring
:func:`dbsprout.schema.introspect.introspect` and
:func:`dbsprout.schema.parsers.django.parse_django_models`.

Requires the optional ``[mongo]`` extra (``pip install dbsprout[mongo]``)
which installs ``pymongo`` and ``pymongo-schema``.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote, urlsplit

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

_ID_FIELD = "_id"
_OID_RAW_TYPE = "oid"
_MONGO_SCHEMES = ("mongodb://", "mongodb+srv://")
_DIALECT = "mongodb"
_SERVER_SELECTION_TIMEOUT_MS = 5000

_MISSING_DEP_MSG = (
    "MongoDB schema inference requires the optional 'mongo' extra. "
    "Install it with: pip install dbsprout[mongo]"
)

# ── MongoDB type-string → ColumnType ─────────────────────────────────────
# Type strings come from pymongo_schema.mongo_sql_types.PYMONGO_TYPE_TO_TYPE_STRING
# plus 'number' (a least-common-parent emitted by pymongo-schema for mixed
# integer/float fields).
_MONGO_TYPE_MAP: dict[str, ColumnType] = {
    "string": ColumnType.VARCHAR,
    "integer": ColumnType.INTEGER,
    "biginteger": ColumnType.BIGINT,
    "float": ColumnType.FLOAT,
    "number": ColumnType.FLOAT,
    "boolean": ColumnType.BOOLEAN,
    "date": ColumnType.DATETIME,
    "timestamp": ColumnType.DATETIME,
    "oid": ColumnType.VARCHAR,
    "dbref": ColumnType.VARCHAR,
    "OBJECT": ColumnType.JSON,
    "ARRAY": ColumnType.ARRAY,
    "null": ColumnType.UNKNOWN,
    "unknown": ColumnType.UNKNOWN,
}


def _mongo_type_to_column_type(mongo_type: str) -> ColumnType:
    """Map a pymongo-schema type string to a unified :class:`ColumnType`.

    Unrecognized type strings fall back to :attr:`ColumnType.UNKNOWN` so a
    new BSON type never crashes inference.
    """
    return _MONGO_TYPE_MAP.get(mongo_type, ColumnType.UNKNOWN)


# ── Field → column translation (flatten embedded docs) ───────────────────


def _raw_type(field_schema: dict[str, Any]) -> str:
    """Human-readable raw type string preserved on the column."""
    type_str = str(field_schema.get("type", "unknown"))
    if type_str == "ARRAY":
        return f"array<{field_schema.get('array_type', 'unknown')}>"
    return type_str


def _object_to_columns(
    object_schema: dict[str, Any],
    *,
    prefix: str = "",
    parent_nullable: bool = False,
) -> list[ColumnSchema]:
    """Flatten a pymongo-schema ``object`` dict into ColumnSchema list.

    - Scalar / array fields → one column each.
    - ``OBJECT`` fields → a JSON column for the parent *and* recursive
      dot-notation child columns (``address`` + ``address.city``).
    - ``nullable`` is ``True`` when the field is absent from at least one
      sampled document (``prop_in_object < 1.0``) or any ancestor object
      was itself partial.
    """
    columns: list[ColumnSchema] = []
    for field_name, field_schema in object_schema.items():
        col_name = f"{prefix}{field_name}"
        type_str = str(field_schema.get("type", "unknown"))
        field_partial = float(field_schema.get("prop_in_object", 1.0)) < 1.0
        nullable = parent_nullable or field_partial

        columns.append(
            ColumnSchema(
                name=col_name,
                data_type=_mongo_type_to_column_type(type_str),
                raw_type=_raw_type(field_schema),
                nullable=nullable,
            )
        )

        if type_str == "OBJECT" and isinstance(field_schema.get("object"), dict):
            columns.extend(
                _object_to_columns(
                    field_schema["object"],
                    prefix=f"{col_name}.",
                    parent_nullable=nullable,
                )
            )
    return columns


# ── Collection → TableSchema ─────────────────────────────────────────────


def _id_column(field_schema: dict[str, Any] | None) -> ColumnSchema:
    """Build the ``_id`` primary-key column (never nullable)."""
    type_str = str(field_schema.get("type", "oid")) if field_schema else "oid"
    return ColumnSchema(
        name=_ID_FIELD,
        data_type=_mongo_type_to_column_type(type_str),
        raw_type=type_str if type_str != "ARRAY" else "array",
        nullable=False,
        primary_key=True,
    )


def _collection_to_table(name: str, collection_schema: dict[str, Any]) -> TableSchema:
    """Convert one pymongo-schema collection schema to a TableSchema.

    ``_id`` is always the primary key. It is injected if it never appeared
    in the sample (e.g. an empty collection) so every table has a PK.
    """
    obj: dict[str, Any] = dict(collection_schema.get("object", {}))
    id_field_schema = obj.pop(_ID_FIELD, None)

    columns: list[ColumnSchema] = [_id_column(id_field_schema)]
    columns.extend(_object_to_columns(obj))

    return TableSchema(
        name=name,
        columns=columns,
        primary_key=[_ID_FIELD],
        row_count_hint=int(collection_schema.get("count", 0)) or None,
    )


# ── Foreign-key inference ────────────────────────────────────────────────


def _ref_stem(column_name: str) -> str | None:
    """Extract the referenced-collection stem from ``<stem>_id``/``<stem>Id``.

    Returns ``None`` for ``_id`` itself or names without a recognized
    id-suffix.
    """
    if column_name == _ID_FIELD:
        return None
    if column_name.endswith("_id") and len(column_name) > 3:
        return column_name[:-3]
    if column_name.endswith("Id") and len(column_name) > 2:
        return column_name[:-2]
    return None


def _infer_foreign_keys(tables: list[TableSchema]) -> list[TableSchema]:
    """Infer FKs from ObjectId ``*_id``/``*Id`` fields to other collections.

    Conservative: only ``oid``-typed columns whose stem (singular or with a
    trailing ``s``) matches an existing collection/table name become a
    :class:`ForeignKeySchema` referencing that table's ``_id``.
    """
    table_names = {t.name for t in tables}
    result: list[TableSchema] = []
    for table in tables:
        fks: list[ForeignKeySchema] = list(table.foreign_keys)
        for col in table.columns:
            if col.name == _ID_FIELD or col.raw_type != _OID_RAW_TYPE:
                continue
            stem = _ref_stem(col.name)
            if stem is None:
                continue
            ref = _match_collection(stem, table_names)
            if ref is None:
                continue
            fks.append(
                ForeignKeySchema(
                    columns=[col.name],
                    ref_table=ref,
                    ref_columns=[_ID_FIELD],
                )
            )
        result.append(table.model_copy(update={"foreign_keys": fks}))
    return result


def _match_collection(stem: str, table_names: set[str]) -> str | None:
    """Match an FK stem against collection names (exact or pluralized)."""
    for candidate in (stem, f"{stem}s"):
        if candidate in table_names:
            return candidate
    return None


# ── Connection + orchestration ───────────────────────────────────────────


def is_mongo_url(url: str) -> bool:
    """True for ``mongodb://`` / ``mongodb+srv://`` connection strings."""
    return url.startswith(_MONGO_SCHEMES)


def _sanitize_uri(uri: str) -> str:
    """Return the URI with any password redacted (never log credentials)."""
    parts = urlsplit(uri)
    if parts.password is None:
        return uri
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = f"{parts.username}:***@" if parts.username else "***@"
    return f"{parts.scheme}://{userinfo}{host}{parts.path}"


def _database_from_uri(uri: str) -> str | None:
    """Extract the default database name from the URI path (no DNS)."""
    path = urlsplit(uri).path.lstrip("/")
    # Strip any collection segment; the DB is the first path component.
    db = unquote(path.split("/", 1)[0]) if path else ""
    return db or None


def _make_client(uri: str) -> Any:
    """Create a pymongo client with a bounded server-selection timeout.

    Isolated so tests can monkeypatch it with a ``mongomock`` client and
    no live MongoDB is ever required.
    """
    try:
        from pymongo import MongoClient  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ValueError(_MISSING_DEP_MSG) from exc
    return MongoClient(uri, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS)


def _load_extractor() -> Any:
    """Import ``extract_pymongo_client_schema`` or raise a clear error."""
    try:
        from pymongo_schema.extract import (  # noqa: PLC0415
            extract_pymongo_client_schema,
        )
    except ImportError as exc:
        raise ValueError(_MISSING_DEP_MSG) from exc
    return extract_pymongo_client_schema


def infer_mongo_schema(
    uri: str,
    *,
    sample_size: int = 1000,
    source: str | None = None,
) -> DatabaseSchema:
    """Infer a :class:`DatabaseSchema` by sampling MongoDB documents.

    Connects to ``uri``, samples up to ``sample_size`` documents per
    collection (``sample_size=0`` scans the full collection), and
    translates the inferred structure into the unified model. Embedded
    documents flatten to dot-notation columns; ObjectId ``*_id`` fields
    referencing other collections become foreign keys.

    Raises:
        ValueError: the ``[mongo]`` extra is not installed, or the URI is
            not a MongoDB connection string.
    """
    if not is_mongo_url(uri):
        msg = f"Not a MongoDB connection string: {_sanitize_uri(uri)!r}"
        raise ValueError(msg)

    extract = mongo_extract_fn()
    client = _make_client(uri)
    try:
        db_name = _database_from_uri(uri)
        db_names = [db_name] if db_name else None
        client_schema: dict[str, Any] = extract(
            client, database_names=db_names, sample_size=sample_size
        )
    finally:
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()

    tables = _client_schema_to_tables(client_schema, db_name)
    tables = _infer_foreign_keys(tables)
    return DatabaseSchema(
        tables=tables,
        dialect=_DIALECT,
        source=source or _sanitize_uri(uri),
    )


# Indirection so tests can monkeypatch the extractor symbol on this module.
extract_pymongo_client_schema: Any = None


def mongo_extract_fn() -> Any:
    """Resolve the schema extractor (patched symbol wins for tests)."""
    if extract_pymongo_client_schema is not None:
        return extract_pymongo_client_schema
    return _load_extractor()


def _client_schema_to_tables(
    client_schema: dict[str, Any], db_name: str | None
) -> list[TableSchema]:
    """Translate the nested client schema dict into TableSchema list."""
    if db_name is not None and db_name in client_schema:
        databases = {db_name: client_schema[db_name]}
    else:
        databases = client_schema

    tables: list[TableSchema] = []
    for db_schema in databases.values():
        for coll_name, coll_schema in db_schema.items():
            tables.append(_collection_to_table(coll_name, coll_schema))
    return tables
