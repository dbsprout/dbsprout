"""Tests for dbsprout.schema.parsers.mongodb — MongoDB schema inference.

These tests require the optional ``[mongo]`` extra plus ``mongomock``.
They are skipped automatically when those packages are absent (the CI
matrix is ``dev data stats`` and does not install the ``mongo`` extra).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo_schema", reason="pymongo-schema absent ([mongo] extra)")
mongomock = pytest.importorskip("mongomock", reason="mongomock absent (dev extra)")

from bson import ObjectId  # noqa: E402

from dbsprout.cli.commands import init as init_cmd  # noqa: E402
from dbsprout.schema.models import (  # noqa: E402
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.schema.parsers import mongodb as mongo_mod  # noqa: E402
from dbsprout.schema.parsers.mongodb import (  # noqa: E402
    _collection_to_table,
    _infer_foreign_keys,
    _mongo_type_to_column_type,
    _object_to_columns,
    infer_mongo_schema,
    is_mongo_url,
)


def _field(
    type_str: str,
    *,
    prop: float = 1.0,
    array_type: str | None = None,
    obj: dict | None = None,
) -> dict:
    """Build a minimal pymongo-schema field-schema dict for tests."""
    fs: dict = {"type": type_str, "prop_in_object": prop, "count": 1}
    if array_type is not None:
        fs["array_type"] = array_type
    if obj is not None:
        fs["object"] = obj
    return fs


class TestTypeMapping:
    @pytest.mark.parametrize(
        ("mongo_type", "expected"),
        [
            ("string", ColumnType.VARCHAR),
            ("integer", ColumnType.INTEGER),
            ("biginteger", ColumnType.BIGINT),
            ("float", ColumnType.FLOAT),
            ("number", ColumnType.FLOAT),
            ("boolean", ColumnType.BOOLEAN),
            ("date", ColumnType.DATETIME),
            ("timestamp", ColumnType.DATETIME),
            ("oid", ColumnType.VARCHAR),
            ("dbref", ColumnType.VARCHAR),
            ("OBJECT", ColumnType.JSON),
            ("ARRAY", ColumnType.ARRAY),
            ("null", ColumnType.UNKNOWN),
            ("unknown", ColumnType.UNKNOWN),
        ],
    )
    def test_known_types(self, mongo_type: str, expected: ColumnType) -> None:
        assert _mongo_type_to_column_type(mongo_type) is expected

    def test_unmapped_type_falls_back_to_unknown(self) -> None:
        assert _mongo_type_to_column_type("decimal128") is ColumnType.UNKNOWN


class TestObjectToColumns:
    def test_scalar_fields(self) -> None:
        obj = {"name": _field("string"), "age": _field("integer")}
        cols = _object_to_columns(obj)
        by_name = {c.name: c for c in cols}
        assert by_name["name"].data_type is ColumnType.VARCHAR
        assert by_name["age"].data_type is ColumnType.INTEGER
        assert by_name["name"].raw_type == "string"

    def test_partial_field_is_nullable(self) -> None:
        obj = {"always": _field("string", prop=1.0), "sometimes": _field("integer", prop=0.5)}
        by_name = {c.name: c for c in _object_to_columns(obj)}
        assert by_name["always"].nullable is False
        assert by_name["sometimes"].nullable is True

    def test_array_field(self) -> None:
        obj = {"tags": _field("ARRAY", array_type="string")}
        col = _object_to_columns(obj)[0]
        assert col.name == "tags"
        assert col.data_type is ColumnType.ARRAY
        assert col.raw_type == "array<string>"

    def test_embedded_object_emits_parent_json_and_dot_children(self) -> None:
        obj = {
            "address": _field(
                "OBJECT",
                obj={"city": _field("string"), "zip": _field("integer", prop=0.5)},
            )
        }
        by_name = {c.name: c for c in _object_to_columns(obj)}
        assert by_name["address"].data_type is ColumnType.JSON
        assert by_name["address.city"].data_type is ColumnType.VARCHAR
        assert by_name["address.zip"].data_type is ColumnType.INTEGER
        assert by_name["address.zip"].nullable is True

    def test_child_nullable_when_parent_partial(self) -> None:
        obj = {
            "meta": _field(
                "OBJECT",
                prop=0.5,
                obj={"k": _field("string", prop=1.0)},
            )
        }
        by_name = {c.name: c for c in _object_to_columns(obj)}
        # Parent present in only half the docs → child must be nullable too.
        assert by_name["meta"].nullable is True
        assert by_name["meta.k"].nullable is True

    def test_nested_object_two_levels(self) -> None:
        obj = {
            "a": _field(
                "OBJECT",
                obj={"b": _field("OBJECT", obj={"c": _field("string")})},
            )
        }
        names = {c.name for c in _object_to_columns(obj)}
        assert {"a", "a.b", "a.b.c"} <= names


class TestCollectionToTable:
    def test_basic_table_with_id_pk(self) -> None:
        cs = {
            "count": 2,
            "object": {
                "_id": _field("oid"),
                "name": _field("string"),
            },
        }
        table = _collection_to_table("users", cs)
        assert table.name == "users"
        assert table.primary_key == ["_id"]
        id_col = table.get_column("_id")
        assert id_col is not None
        assert id_col.primary_key is True
        assert id_col.nullable is False
        assert id_col.data_type is ColumnType.VARCHAR

    def test_empty_collection_still_has_id_column(self) -> None:
        cs: dict = {"count": 0, "object": {}}
        table = _collection_to_table("empty", cs)
        assert [c.name for c in table.columns] == ["_id"]
        assert table.primary_key == ["_id"]

    def test_id_injected_when_absent_from_sample(self) -> None:
        cs = {"count": 1, "object": {"name": _field("string")}}
        table = _collection_to_table("things", cs)
        names = [c.name for c in table.columns]
        assert "_id" in names
        assert names[0] == "_id"
        id_col = table.get_column("_id")
        assert id_col is not None
        assert id_col.primary_key is True

    def test_id_never_nullable_even_if_sample_partial(self) -> None:
        cs = {"count": 2, "object": {"_id": _field("oid", prop=0.5)}}
        table = _collection_to_table("c", cs)
        id_col = table.get_column("_id")
        assert id_col is not None
        assert id_col.nullable is False


class TestForeignKeyInference:
    def test_snake_case_oid_ref_inferred(self) -> None:
        users = _collection_to_table("users", {"object": {"_id": _field("oid")}})
        orders = _collection_to_table(
            "orders",
            {"object": {"_id": _field("oid"), "user_id": _field("oid")}},
        )
        result = _infer_foreign_keys([users, orders])
        orders_t = next(t for t in result if t.name == "orders")
        assert len(orders_t.foreign_keys) == 1
        fk = orders_t.foreign_keys[0]
        assert fk.columns == ["user_id"]
        assert fk.ref_table == "users"
        assert fk.ref_columns == ["_id"]

    def test_singular_collection_name_match(self) -> None:
        # collection 'user' (singular); field 'user_id' should still match.
        user = _collection_to_table("user", {"object": {"_id": _field("oid")}})
        post = _collection_to_table(
            "post",
            {"object": {"_id": _field("oid"), "user_id": _field("oid")}},
        )
        result = _infer_foreign_keys([user, post])
        post_t = next(t for t in result if t.name == "post")
        assert [fk.ref_table for fk in post_t.foreign_keys] == ["user"]

    def test_camel_case_oid_ref_inferred(self) -> None:
        users = _collection_to_table("users", {"object": {"_id": _field("oid")}})
        orders = _collection_to_table(
            "orders",
            {"object": {"_id": _field("oid"), "userId": _field("oid")}},
        )
        result = _infer_foreign_keys([users, orders])
        orders_t = next(t for t in result if t.name == "orders")
        assert [fk.columns for fk in orders_t.foreign_keys] == [["userId"]]

    def test_no_match_when_collection_absent(self) -> None:
        orders = _collection_to_table(
            "orders",
            {"object": {"_id": _field("oid"), "note_id": _field("oid")}},
        )
        result = _infer_foreign_keys([orders])
        assert next(t for t in result if t.name == "orders").foreign_keys == []

    def test_non_oid_id_field_not_a_fk(self) -> None:
        users = _collection_to_table("users", {"object": {"_id": _field("oid")}})
        orders = _collection_to_table(
            "orders",
            {"object": {"_id": _field("oid"), "user_id": _field("integer")}},
        )
        result = _infer_foreign_keys([users, orders])
        assert next(t for t in result if t.name == "orders").foreign_keys == []

    def test_id_field_itself_never_a_fk(self) -> None:
        # A collection named 'i' must not make '_id' a FK to itself/others.
        a = _collection_to_table("i", {"object": {"_id": _field("oid")}})
        b = _collection_to_table("b", {"object": {"_id": _field("oid")}})
        result = _infer_foreign_keys([a, b])
        assert all(t.foreign_keys == [] for t in result)


def _seed_client() -> mongomock.MongoClient:
    """A mongomock client with heterogeneous users + referencing orders."""
    client: mongomock.MongoClient = mongomock.MongoClient()
    db = client["shop"]
    user_id = ObjectId()
    db["users"].insert_many(
        [
            {
                "_id": user_id,
                "name": "Alice",
                "age": 30,
                "active": True,
                "address": {"city": "NYC", "zip": 10001},
                "tags": ["a", "b"],
            },
            {
                "_id": ObjectId(),
                "name": "Bob",
                "address": {"city": "LA"},
                "tags": ["c"],
            },
        ]
    )
    db["orders"].insert_many(
        [
            {"_id": ObjectId(), "user_id": user_id, "total": 9.99},
            {"_id": ObjectId(), "user_id": ObjectId(), "total": 1.5},
        ]
    )
    return client


class TestInferMongoSchema:
    def test_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _seed_client()
        monkeypatch.setattr(mongo_mod, "_make_client", lambda _uri: client)

        schema = infer_mongo_schema("mongodb://localhost:27017/shop")

        assert schema.dialect == "mongodb"
        names = set(schema.table_names())
        assert names == {"users", "orders"}

        users = schema.get_table("users")
        assert users is not None
        col_names = {c.name for c in users.columns}
        assert {"_id", "name", "address", "address.city", "address.zip", "tags"} <= col_names
        addr_city = users.get_column("address.city")
        assert addr_city is not None
        assert addr_city.data_type is ColumnType.VARCHAR
        tags = users.get_column("tags")
        assert tags is not None
        assert tags.data_type is ColumnType.ARRAY
        # 'age' present in only one of two docs → nullable.
        age = users.get_column("age")
        assert age is not None
        assert age.nullable is True

        orders = schema.get_table("orders")
        assert orders is not None
        assert [fk.ref_table for fk in orders.foreign_keys] == ["users"]
        assert orders.foreign_keys[0].columns == ["user_id"]

    def test_source_recorded_and_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _seed_client()
        monkeypatch.setattr(mongo_mod, "_make_client", lambda _uri: client)
        schema = infer_mongo_schema("mongodb://user:secret@host:27017/shop")
        assert schema.source is not None
        assert "secret" not in schema.source

    def test_explicit_source_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _seed_client()
        monkeypatch.setattr(mongo_mod, "_make_client", lambda _uri: client)
        schema = infer_mongo_schema("mongodb://localhost/shop", source="my-mongo")
        assert schema.source == "my-mongo"

    def test_sample_size_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _seed_client()
        monkeypatch.setattr(mongo_mod, "_make_client", lambda _uri: client)
        captured: dict = {}

        import pymongo_schema.extract as real_extract  # noqa: PLC0415

        def _spy(pymongo_client: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return real_extract.extract_pymongo_client_schema(pymongo_client, **kwargs)

        monkeypatch.setattr(mongo_mod, "extract_pymongo_client_schema", _spy)
        infer_mongo_schema("mongodb://localhost/shop", sample_size=250)
        assert captured["sample_size"] == 250

    def test_no_collections_yields_empty_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client: mongomock.MongoClient = mongomock.MongoClient()
        client["emptydb"].create_collection("placeholder")
        client["emptydb"].drop_collection("placeholder")
        monkeypatch.setattr(mongo_mod, "_make_client", lambda _uri: client)
        schema = infer_mongo_schema("mongodb://localhost/emptydb")
        assert schema.tables == []
        assert schema.dialect == "mongodb"


class TestIsMongoUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("mongodb://localhost:27017/db", True),
            ("mongodb+srv://user:pw@cluster.example.net/db", True),
            ("postgresql://localhost/db", False),
            ("sqlite:///x.db", False),
            ("mysql://h/db", False),
            ("not-a-url", False),
            ("", False),
        ],
    )
    def test_scheme_detection(self, url: str, expected: bool) -> None:
        assert is_mongo_url(url) is expected


class TestErrorPaths:
    def test_non_mongo_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="Not a MongoDB connection string"):
            infer_mongo_schema("postgresql://localhost/db")

    def test_missing_pymongo_raises_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pymongo":
                raise ImportError("No module named 'pymongo'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ValueError, match=r"dbsprout\[mongo\]"):
            infer_mongo_schema("mongodb://localhost/db")

    def test_missing_pymongo_schema_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins  # noqa: PLC0415

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pymongo_schema.extract":
                raise ImportError("No module named 'pymongo_schema'")
            return real_import(name, *args, **kwargs)

        # Patch the extractor symbol back to None so the loader path runs.
        monkeypatch.setattr(mongo_mod, "extract_pymongo_client_schema", None)
        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ValueError, match=r"dbsprout\[mongo\]"):
            infer_mongo_schema("mongodb://localhost/db")


class TestInitRouting:
    def test_init_routes_mongo_url_to_inference(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        tiny = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        _collection_to_table("users", {"object": {"_id": _field("oid")}}).columns[0]
                    ],
                    primary_key=["_id"],
                )
            ],
            dialect="mongodb",
            source="mongodb://localhost/db",
        )

        called: dict = {}

        def _fake_infer(uri: str, **kwargs: object) -> DatabaseSchema:
            called["uri"] = uri
            return tiny

        def _boom_introspect(_url: str) -> object:  # pragma: no cover - must not run
            raise AssertionError("SQLAlchemy introspect must not be called for mongodb URLs")

        monkeypatch.setattr(init_cmd, "introspect", _boom_introspect)
        monkeypatch.setattr("dbsprout.schema.parsers.mongodb.infer_mongo_schema", _fake_infer)

        # tiny schema has one table → init returns normally (no typer.Exit).
        init_cmd.init_command(
            db="mongodb://localhost:27017/db",
            file=None,
            django=False,
            django_apps=None,
            output_dir=tmp_path,  # type: ignore[arg-type]
            dry_run=False,
        )

        assert called["uri"] == "mongodb://localhost:27017/db"
        assert (tmp_path / "dbsprout.toml").exists()  # type: ignore[operator]

    def test_init_reports_inference_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        import typer  # noqa: PLC0415

        def _fail_infer(_uri: str, **_kw: object) -> DatabaseSchema:
            raise ValueError("install dbsprout[mongo]")

        monkeypatch.setattr("dbsprout.schema.parsers.mongodb.infer_mongo_schema", _fail_infer)
        with pytest.raises(typer.Exit) as exc:
            init_cmd.init_command(
                db="mongodb://localhost/db",
                file=None,
                django=False,
                django_apps=None,
                output_dir=tmp_path,  # type: ignore[arg-type]
                dry_run=False,
            )
        assert exc.value.exit_code == 1

    def test_init_warns_on_no_collections(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        import typer  # noqa: PLC0415

        empty = DatabaseSchema(tables=[], dialect="mongodb", source="mongodb://x/db")
        monkeypatch.setattr(
            "dbsprout.schema.parsers.mongodb.infer_mongo_schema",
            lambda _uri, **_kw: empty,
        )
        with pytest.raises(typer.Exit) as exc:
            init_cmd.init_command(
                db="mongodb://localhost/db",
                file=None,
                django=False,
                django_apps=None,
                output_dir=tmp_path,  # type: ignore[arg-type]
                dry_run=False,
            )
        assert exc.value.exit_code == 0
        assert (tmp_path / "dbsprout.toml").exists()  # type: ignore[operator]
