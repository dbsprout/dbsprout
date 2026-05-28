"""Tests for column-update writer (S-138).

Covers the core ``update_column`` API and the ``ColumnUpdateWriter`` plugin
wrapper. Pure unit tests against SQLite in-memory + mocks — dialect coverage
is logical (parameterised SQL is dialect-agnostic via SQLAlchemy ``text()``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from dbsprout.output.column_update import (
    ColumnUpdateError,
    ColumnUpdateWriter,
    update_column,
)
from dbsprout.output.models import ColumnUpdateResult
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


def _single_pk_schema(table_name: str = "users") -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name=table_name,
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="name",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="email",
                        data_type=ColumnType.VARCHAR,
                        nullable=True,
                    ),
                ],
                primary_key=["id"],
            )
        ],
        dialect="sqlite",
    )


def _composite_pk_schema(table_name: str = "memberships") -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name=table_name,
                columns=[
                    ColumnSchema(
                        name="org_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="user_id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                    ),
                    ColumnSchema(
                        name="role",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["org_id", "user_id"],
            )
        ],
        dialect="sqlite",
    )


def _pkless_schema(table_name: str = "audit") -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name=table_name,
                columns=[
                    ColumnSchema(
                        name="event",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=[],
            )
        ],
        dialect="sqlite",
    )


def _seed_users(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)")
        )
        conn.execute(
            sa.text("INSERT INTO users (id, name, email) VALUES (:id, :name, :email)"),
            [
                {"id": 1, "name": "alice", "email": "a@x"},
                {"id": 2, "name": "bob", "email": "b@x"},
                {"id": 3, "name": "carol", "email": "c@x"},
                {"id": 4, "name": "dave", "email": "d@x"},
                {"id": 5, "name": "eve", "email": "e@x"},
            ],
        )
        conn.commit()


def _seed_memberships(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE memberships "
                "(org_id INTEGER, user_id INTEGER, role TEXT NOT NULL, "
                "PRIMARY KEY (org_id, user_id))"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO memberships (org_id, user_id, role) VALUES (:org_id, :user_id, :role)"
            ),
            [
                {"org_id": 1, "user_id": 10, "role": "member"},
                {"org_id": 1, "user_id": 11, "role": "member"},
                {"org_id": 2, "user_id": 10, "role": "admin"},
            ],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Result + error types
# ---------------------------------------------------------------------------


class TestColumnUpdateResultType:
    def test_is_frozen_dataclass(self) -> None:
        r = ColumnUpdateResult(rows_updated=5, duration_seconds=0.1)
        with pytest.raises((AttributeError, Exception)):
            r.rows_updated = 0  # type: ignore[misc]

    def test_carries_count_and_duration(self) -> None:
        r = ColumnUpdateResult(rows_updated=3, duration_seconds=0.42)
        assert r.rows_updated == 3
        assert r.duration_seconds == pytest.approx(0.42)


class TestColumnUpdateError:
    def test_holds_code_and_table(self) -> None:
        err = ColumnUpdateError("no_primary_key", table="audit")
        assert err.code == "no_primary_key"
        assert err.table == "audit"
        assert "no_primary_key" in str(err)
        assert "audit" in str(err)

    def test_optional_detail(self) -> None:
        err = ColumnUpdateError("update_failed", table="users", detail="IntegrityError")
        assert err.detail == "IntegrityError"
        assert "IntegrityError" in str(err)


# ---------------------------------------------------------------------------
# Empty / no-op
# ---------------------------------------------------------------------------


class TestEmptyRows:
    def test_returns_zero_no_sql_emitted(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        result = update_column(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="name",
            rows=[],
        )
        assert isinstance(result, ColumnUpdateResult)
        assert result.rows_updated == 0
        assert result.duration_seconds >= 0.0
        with engine.connect() as conn:
            row = conn.execute(sa.text("SELECT name FROM users WHERE id = 1")).scalar()
        assert row == "alice"


# ---------------------------------------------------------------------------
# Single-column PK bulk path
# ---------------------------------------------------------------------------


class TestSinglePkBulkPath:
    def test_updates_only_target_rows(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        result = update_column(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="name",
            rows=[(1, "ALICE"), (2, "BOB"), (3, "CAROL")],
        )
        assert result.rows_updated == 3
        with engine.connect() as conn:
            rows = conn.execute(sa.text("SELECT id, name FROM users ORDER BY id")).fetchall()
        assert rows == [
            (1, "ALICE"),
            (2, "BOB"),
            (3, "CAROL"),
            (4, "dave"),
            (5, "eve"),
        ]

    def test_accepts_generator(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)

        def _gen() -> Iterable[tuple[int, str]]:
            yield (1, "X")
            yield (2, "Y")

        result = update_column(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="name",
            rows=_gen(),
        )
        assert result.rows_updated == 2
        with engine.connect() as conn:
            n1 = conn.execute(sa.text("SELECT name FROM users WHERE id=1")).scalar()
            n2 = conn.execute(sa.text("SELECT name FROM users WHERE id=2")).scalar()
        assert n1 == "X"
        assert n2 == "Y"

    def test_nullable_value(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        result = update_column(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="email",
            rows=[(1, None)],
        )
        assert result.rows_updated == 1
        with engine.connect() as conn:
            email = conn.execute(sa.text("SELECT email FROM users WHERE id=1")).scalar()
        assert email is None


# ---------------------------------------------------------------------------
# Composite-PK fallback
# ---------------------------------------------------------------------------


class TestCompositePkFallback:
    def test_updates_with_tuple_keys(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_memberships(engine)
        result = update_column(
            connection=engine,
            schema=_composite_pk_schema(),
            table="memberships",
            column="role",
            rows=[((1, 10), "owner"), ((2, 10), "guest")],
        )
        assert result.rows_updated == 2
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text("SELECT org_id, user_id, role FROM memberships ORDER BY org_id, user_id")
            ).fetchall()
        assert rows == [
            (1, 10, "owner"),
            (1, 11, "member"),
            (2, 10, "guest"),
        ]

    def test_composite_pk_wrong_tuple_arity_raises(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_memberships(engine)
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=_composite_pk_schema(),
                table="memberships",
                column="role",
                rows=[((1,), "owner")],
            )
        assert exc.value.code == "pk_arity_mismatch"
        assert exc.value.table == "memberships"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestPkLessGuard:
    def test_raises_no_primary_key(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(sa.text("CREATE TABLE audit (event TEXT NOT NULL)"))
            conn.commit()
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=_pkless_schema(),
                table="audit",
                column="event",
                rows=[(1, "x")],
            )
        assert exc.value.code == "no_primary_key"
        assert exc.value.table == "audit"


class TestUnknownTableGuard:
    def test_raises_unknown_table(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=_single_pk_schema(),
                table="ghosts",
                column="name",
                rows=[(1, "x")],
            )
        assert exc.value.code == "unknown_table"
        assert exc.value.table == "ghosts"


class TestUnknownColumnGuard:
    def test_raises_unknown_column(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=_single_pk_schema(),
                table="users",
                column="nonexistent",
                rows=[(1, "x")],
            )
        assert exc.value.code == "unknown_column"
        assert exc.value.table == "users"


class TestPkColumnUpdateRejected:
    def test_cannot_update_primary_key_column(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=_single_pk_schema(),
                table="users",
                column="id",
                rows=[(1, 99)],
            )
        assert exc.value.code == "pk_column_update"
        assert exc.value.table == "users"


class TestUnsafeIdentifierGuard:
    @pytest.mark.parametrize(
        ("bad_table", "bad_column"),
        [
            ("users; DROP TABLE users--", "name"),
            ("good table", "name"),
        ],
    )
    def test_unsafe_table(self, bad_table: str, bad_column: str) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        # Use a schema that *claims* the unsafe table name exists, to force
        # the writer's identifier guard (rather than the unknown-table guard).
        schema = DatabaseSchema(
            tables=[
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(
                            name="id",
                            data_type=ColumnType.INTEGER,
                            nullable=False,
                            primary_key=True,
                        ),
                        ColumnSchema(
                            name="name",
                            data_type=ColumnType.VARCHAR,
                            nullable=False,
                        ),
                    ],
                    primary_key=["id"],
                )
            ],
            dialect="sqlite",
        )
        # Try to update via the safe schema but pass an unsafe table arg.
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=engine,
                schema=schema,
                table=bad_table,
                column=bad_column,
                rows=[(1, "x")],
            )
        # Either guard (unknown_table or unsafe_identifier) is acceptable —
        # both prevent SQL injection. Verify it's one of them.
        assert exc.value.code in {"unknown_table", "unsafe_identifier"}


# ---------------------------------------------------------------------------
# Transactional behaviour
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_on_execute_failure(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)

        # Wrap the engine's connect so that the first execute call raises.
        # We hand the writer a real Connection but mock its .execute so any
        # SQL run fails and the writer's BEGIN/ROLLBACK contract activates.
        with engine.connect() as conn:
            original_execute = conn.execute
            calls = {"n": 0}

            def _bad_execute(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                raise sa.exc.IntegrityError("UPDATE", {}, Exception("boom"))

            conn.execute = _bad_execute  # type: ignore[method-assign]
            with pytest.raises(ColumnUpdateError) as exc:
                update_column(
                    connection=conn,
                    schema=_single_pk_schema(),
                    table="users",
                    column="name",
                    rows=[(1, "X")],
                )
            assert exc.value.code == "update_failed"
            assert exc.value.table == "users"
            # detail field carries the wrapped exception class name
            assert exc.value.detail is not None
            # Restore original .execute for cleanup
            conn.execute = original_execute  # type: ignore[method-assign]

        # State unchanged after rollback.
        with engine.connect() as conn2:
            name = conn2.execute(sa.text("SELECT name FROM users WHERE id=1")).scalar()
        assert name == "alice"


class TestConnectionMode:
    def test_accepts_live_connection(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        with engine.connect() as conn:
            # Caller-owned connection — writer opens its own (savepoint) txn.
            result = update_column(
                connection=conn,
                schema=_single_pk_schema(),
                table="users",
                column="name",
                rows=[(1, "PATCHED")],
            )
            assert result.rows_updated == 1
            # Within the same connection the write is visible.
            name = conn.execute(sa.text("SELECT name FROM users WHERE id=1")).scalar()
            assert name == "PATCHED"
            conn.commit()

        # And persisted across a fresh connection on the same engine.
        with engine.connect() as conn2:
            name = conn2.execute(sa.text("SELECT name FROM users WHERE id=1")).scalar()
        assert name == "PATCHED"

    def test_accepts_connection_with_existing_transaction(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        with engine.begin() as conn:
            # Caller has an active txn — writer must use begin_nested.
            result = update_column(
                connection=conn,
                schema=_single_pk_schema(),
                table="users",
                column="name",
                rows=[(1, "OUTER")],
            )
            assert result.rows_updated == 1


# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------


class TestParameterisedSafety:
    def test_malicious_value_stored_verbatim(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        payload = "'; DROP TABLE users; --"
        result = update_column(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="name",
            rows=[(1, payload)],
        )
        assert result.rows_updated == 1
        with engine.connect() as conn:
            name = conn.execute(sa.text("SELECT name FROM users WHERE id=1")).scalar()
            count = conn.execute(sa.text("SELECT COUNT(*) FROM users")).scalar()
        assert name == payload
        assert count == 5  # table not dropped


# ---------------------------------------------------------------------------
# Writer / plugin wrapper
# ---------------------------------------------------------------------------


class TestColumnUpdateWriter:
    def test_format_attr(self) -> None:
        assert ColumnUpdateWriter.format == "column_update"

    def test_delegates_to_update_column(self) -> None:
        engine = sa.create_engine("sqlite:///:memory:")
        _seed_users(engine)
        result = ColumnUpdateWriter().write(
            connection=engine,
            schema=_single_pk_schema(),
            table="users",
            column="name",
            rows=[(1, "WRITER")],
        )
        assert isinstance(result, ColumnUpdateResult)
        assert result.rows_updated == 1

    def test_refuses_positional_args(self) -> None:
        with pytest.raises(TypeError, match="keyword arguments"):
            ColumnUpdateWriter().write("engine_placeholder")  # type: ignore[call-overload]


class TestUnsafeIdentifierDirect:
    """Exercise the identifier-guard branch directly via crafted schemas.

    The public API runs ``schema.get_table`` first, so an unsafe *table*
    name normally hits the ``unknown_table`` guard. To prove the identifier
    guard itself fires, we register a schema whose PK column name is
    structurally unsafe — that bypasses the column/PK guards and routes
    straight to ``_safe_ident``.

    We can't actually build such a ``DatabaseSchema`` because Pydantic's
    identifier validator rejects path-traversal characters; instead we call
    the private ``_safe_ident`` helper directly to lock the contract.
    """

    def test_safe_ident_rejects_injection_payload(self) -> None:
        from dbsprout.output.column_update import _safe_ident  # noqa: PLC0415

        with pytest.raises(ColumnUpdateError) as exc:
            _safe_ident("users; DROP TABLE users--")
        assert exc.value.code == "unsafe_identifier"

    def test_safe_ident_rejects_whitespace(self) -> None:
        from dbsprout.output.column_update import _safe_ident  # noqa: PLC0415

        with pytest.raises(ColumnUpdateError) as exc:
            _safe_ident("good name")
        assert exc.value.code == "unsafe_identifier"

    def test_safe_ident_accepts_valid(self) -> None:
        from dbsprout.output.column_update import _safe_ident  # noqa: PLC0415

        assert _safe_ident("users") == '"users"'
        assert _safe_ident("my_table_2") == '"my_table_2"'


class TestPluginRegistration:
    def test_registered_under_outputs_group(self) -> None:
        # Force-reset the registry so the entry point is freshly discovered.
        # Imported locally because this test mutates global state and we want
        # the helpers visible only when the test runs.
        from dbsprout.plugins.registry import (  # noqa: PLC0415
            _reset_for_tests,
            get_registry,
        )

        _reset_for_tests()
        try:
            reg = get_registry()
            obj = reg.get("dbsprout.outputs", "column_update")
            assert obj is not None
            assert obj.format == "column_update"
        finally:
            _reset_for_tests()


# ---------------------------------------------------------------------------
# Engine-without-tables short-circuit (defensive coverage)
# ---------------------------------------------------------------------------


class TestEngineCreationFailure:
    def test_bad_connection_argument_raises_typed_error(self) -> None:
        # Pass something that's neither Engine nor Connection.
        sentinel = MagicMock(spec=[])  # no relevant attrs
        with pytest.raises(ColumnUpdateError) as exc:
            update_column(
                connection=sentinel,  # type: ignore[arg-type]
                schema=_single_pk_schema(),
                table="users",
                column="name",
                rows=[(1, "x")],
            )
        assert exc.value.code == "invalid_connection"
