"""Tests for dbsprout.generate.regenerate — single-table re-roll (S-128).

Covers PK stability, FK re-sampling, constraint enforcement, determinism,
purity (no web imports), and the error surface.
"""

from __future__ import annotations

import copy
import importlib
import sys
from typing import Any

import pytest

from dbsprout.generate.regenerate import _regen_columns, regenerate_table
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    IndexSchema,
    TableSchema,
)
from dbsprout.spec.models import DataSpec, GeneratorConfig, TableSpec

# ───────────────────────── helpers ─────────────────────────


def _int_col(name: str, *, nullable: bool = True, pk: bool = False) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=ColumnType.INTEGER,
        nullable=nullable,
        primary_key=pk,
    )


def _str_col(
    name: str,
    *,
    nullable: bool = True,
    unique: bool = False,
    max_length: int | None = 50,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=ColumnType.VARCHAR,
        nullable=nullable,
        unique=unique,
        max_length=max_length,
    )


def _parent_table() -> TableSchema:
    return TableSchema(
        name="departments",
        columns=[
            _int_col("id", nullable=False, pk=True),
            _str_col("name", nullable=False),
        ],
        primary_key=["id"],
    )


def _child_table(*, unique_email: bool = False) -> TableSchema:
    return TableSchema(
        name="employees",
        columns=[
            _int_col("id", nullable=False, pk=True),
            _str_col("name", nullable=False),
            _str_col("email", nullable=False, unique=unique_email),
            _int_col("dept_id", nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["dept_id"],
                ref_table="departments",
                ref_columns=["id"],
            )
        ],
    )


def _build_schema(*, unique_email: bool = False) -> DatabaseSchema:
    return DatabaseSchema(
        tables=[_parent_table(), _child_table(unique_email=unique_email)],
        dialect="sqlite",
    )


def _initial_state() -> dict[str, list[dict[str, Any]]]:
    """A plausible 'initial pass' tables_data dict."""
    parents = [{"id": i, "name": f"D{i}"} for i in (10, 20, 30)]
    children = [
        {"id": 1, "name": "Alice", "email": "alice@a.com", "dept_id": 10},
        {"id": 2, "name": "Bob", "email": "bob@a.com", "dept_id": 20},
        {"id": 3, "name": "Carol", "email": "carol@a.com", "dept_id": 30},
        {"id": 4, "name": "Dan", "email": "dan@a.com", "dept_id": 10},
        {"id": 5, "name": "Eve", "email": "eve@a.com", "dept_id": 20},
    ]
    return {"departments": parents, "employees": children}


# ───────────────────────── PK stability ─────────────────────────


class TestPKStability:
    def test_pk_values_byte_identical_after_regeneration(self) -> None:
        """Golden invariant: PK values must be preserved row-by-row."""
        schema = _build_schema()
        state = _initial_state()
        original_pks = [row["id"] for row in state["employees"]]

        new_rows = regenerate_table(schema, state, "employees", seed=99)

        new_pks = [row["id"] for row in new_rows]
        assert new_pks == original_pks, "PK list must be byte-identical"

    def test_composite_pk_preserves_all_pk_columns(self) -> None:
        table = TableSchema(
            name="enrollments",
            columns=[
                _int_col("student_id", nullable=False, pk=True),
                _int_col("course_id", nullable=False, pk=True),
                _str_col("grade", nullable=True),
            ],
            primary_key=["student_id", "course_id"],
        )
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        state = {
            "enrollments": [
                {"student_id": 1, "course_id": 100, "grade": "A"},
                {"student_id": 2, "course_id": 200, "grade": "B"},
                {"student_id": 3, "course_id": 300, "grade": "C"},
            ]
        }
        original_pk_tuples = [(r["student_id"], r["course_id"]) for r in state["enrollments"]]

        new_rows = regenerate_table(schema, state, "enrollments", seed=7)

        new_pk_tuples = [(r["student_id"], r["course_id"]) for r in new_rows]
        assert new_pk_tuples == original_pk_tuples


# ───────────────────────── FK re-sampling ─────────────────────────


class TestFKResampling:
    def test_fk_values_reference_only_parent_pks(self) -> None:
        schema = _build_schema()
        state = _initial_state()
        parent_pks = {r["id"] for r in state["departments"]}

        new_rows = regenerate_table(schema, state, "employees", seed=42)

        for row in new_rows:
            assert row["dept_id"] in parent_pks


# ───────────────────────── determinism ─────────────────────────


class TestDeterminism:
    def test_same_seed_produces_identical_output(self) -> None:
        schema = _build_schema()
        state_a = _initial_state()
        state_b = _initial_state()

        rows_a = regenerate_table(schema, state_a, "employees", seed=123)
        rows_b = regenerate_table(schema, state_b, "employees", seed=123)

        assert rows_a == rows_b

    def test_different_seeds_change_non_pk_columns(self) -> None:
        schema = _build_schema()
        state_a = _initial_state()
        state_b = _initial_state()

        rows_a = regenerate_table(schema, state_a, "employees", seed=1)
        rows_b = regenerate_table(schema, state_b, "employees", seed=2)

        # Non-PK columns should differ at least somewhere.
        names_a = [r["name"] for r in rows_a]
        names_b = [r["name"] for r in rows_b]
        assert names_a != names_b


# ───────────────────────── constraints ─────────────────────────


class TestConstraintEnforcement:
    def test_unique_column_has_no_duplicates_after_regen(self) -> None:
        schema = _build_schema(unique_email=True)
        state = _initial_state()

        new_rows = regenerate_table(schema, state, "employees", seed=11)

        emails = [r["email"] for r in new_rows]
        assert len(emails) == len(set(emails)), "UNIQUE column must have no duplicates"

    def test_not_null_columns_have_no_none_after_regen(self) -> None:
        schema = _build_schema()
        state = _initial_state()

        new_rows = regenerate_table(schema, state, "employees", seed=11)

        for row in new_rows:
            assert row["name"] is not None
            assert row["email"] is not None


# ───────────────────────── purity / API surface ─────────────────────────


class TestPurity:
    def test_does_not_mutate_input_state(self) -> None:
        schema = _build_schema()
        state = _initial_state()
        snapshot = copy.deepcopy(state)

        regenerate_table(schema, state, "employees", seed=42)

        assert state == snapshot, "regenerate_table must not mutate the state dict"

    def test_returns_new_list(self) -> None:
        schema = _build_schema()
        state = _initial_state()

        new_rows = regenerate_table(schema, state, "employees", seed=42)

        assert new_rows is not state["employees"]
        assert isinstance(new_rows, list)
        assert len(new_rows) == len(state["employees"])

    def test_module_has_no_web_or_progress_imports(self) -> None:
        """Pure API: must not import from dbsprout.web or job-manager surfaces.

        Parse the module AST so we inspect real ``import``/``from ... import``
        statements rather than substring-matching docstrings (which legitimately
        *describe* the boundary they enforce).
        """
        import ast  # noqa: PLC0415

        mod_name = "dbsprout.generate.regenerate"
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

        import dbsprout.generate.regenerate as mod  # noqa: PLC0415

        with open(mod.__file__, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())

        banned_module_prefixes = ("dbsprout.web", "dbsprout.generate.progress")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith(banned_module_prefixes), (
                    f"regenerate.py must not import from {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(banned_module_prefixes), (
                        f"regenerate.py must not import {alias.name}"
                    )

        # And the public surface must not advertise WS-progress kwargs.
        sig = mod.regenerate_table.__annotations__
        assert "progress_callback" not in sig
        assert "cancel_token" not in sig


# ───────────────────────── error surface ─────────────────────────


class TestErrorSurface:
    def test_unknown_table_raises_valueerror(self) -> None:
        schema = _build_schema()
        state = _initial_state()
        with pytest.raises(ValueError, match="unknown table"):
            regenerate_table(schema, state, "ghost", seed=1)

    def test_missing_state_entry_raises_valueerror(self) -> None:
        schema = _build_schema()
        # Drop the entry entirely.
        state: dict[str, list[dict[str, Any]]] = {"departments": _initial_state()["departments"]}
        with pytest.raises(ValueError, match="no rows in state"):
            regenerate_table(schema, state, "employees", seed=1)

    def test_empty_state_entry_raises_valueerror(self) -> None:
        schema = _build_schema()
        state = _initial_state()
        state["employees"] = []
        with pytest.raises(ValueError, match="no rows in state"):
            regenerate_table(schema, state, "employees", seed=1)


# ───────────────────────── engine selection ─────────────────────────


class TestEngineSelection:
    def test_heuristic_engine_default(self) -> None:
        """Default engine="heuristic" path runs without a DataSpec."""
        schema = _build_schema()
        state = _initial_state()

        rows = regenerate_table(schema, state, "employees", seed=42)

        assert len(rows) == len(state["employees"])

    def test_spec_engine_uses_provided_dataspec(self) -> None:
        """When engine='spec' and a DataSpec is supplied, spec-driven engine runs."""
        schema = _build_schema()
        state = _initial_state()
        spec = DataSpec(
            tables=[
                TableSpec(
                    table_name="employees",
                    row_count=len(state["employees"]),
                    columns={
                        "name": GeneratorConfig(provider="person.full_name"),
                        "email": GeneratorConfig(provider="internet.email"),
                    },
                )
            ],
            schema_hash="test",
        )

        rows = regenerate_table(schema, state, "employees", seed=42, spec=spec, engine="spec")

        assert len(rows) == len(state["employees"])
        # PK still preserved.
        assert [r["id"] for r in rows] == [r["id"] for r in state["employees"]]

    def test_spec_engine_without_dataspec_falls_back_to_heuristic(self) -> None:
        """engine='spec' without a DataSpec must still produce valid rows."""
        schema = _build_schema()
        state = _initial_state()

        rows = regenerate_table(schema, state, "employees", seed=42, engine="spec")

        assert len(rows) == len(state["employees"])


# ───────────────────────── composite UNIQUE index ─────────────────────────


class TestRegenColumnsSplice:
    """Coverage for the S-129 forward-compat partial-column path.

    The public ``regenerate_table`` always sets ``columns=None`` (whole-table
    re-roll), so the splice path is exercised here via ``_regen_columns`` to
    pin its semantics now and unblock the S-129 wave.
    """

    def test_columns_set_keeps_only_requested_columns_regenerated(self) -> None:
        schema = _build_schema()
        state = _initial_state()
        table_schema = schema.get_table("employees")
        assert table_schema is not None
        original_rows = state["employees"]

        new_rows = _regen_columns(
            schema=schema,
            state=state,
            table_schema=table_schema,
            original_rows=original_rows,
            columns={"name"},  # re-roll name only
            seed=42,
            spec=None,
            engine="heuristic",
        )

        # name column was re-rolled (at least one differs).
        # email — a non-FK, non-PK column NOT in keep_columns — is restored
        # from the originals.
        for orig, new in zip(original_rows, new_rows, strict=False):
            assert new["email"] == orig["email"]
            assert new["id"] == orig["id"]  # PK pinned


class TestPKPinNoPK:
    """A table with no primary key must still regenerate cleanly."""

    def test_no_primary_key_table_regenerates_without_error(self) -> None:
        table = TableSchema(
            name="audit_log",
            columns=[
                _str_col("message", nullable=False),
                _int_col("ts", nullable=False),
            ],
            primary_key=[],
        )
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        state = {
            "audit_log": [
                {"message": "boot", "ts": 1},
                {"message": "tick", "ts": 2},
            ]
        }

        new_rows = regenerate_table(schema, state, "audit_log", seed=3)

        assert len(new_rows) == 2


class TestCompositeUniqueIndex:
    def test_composite_unique_index_dedup_after_regen(self) -> None:
        table = TableSchema(
            name="users",
            columns=[
                _int_col("id", nullable=False, pk=True),
                _str_col("first", nullable=False),
                _str_col("last", nullable=False),
            ],
            primary_key=["id"],
            indexes=[IndexSchema(name="uq_name", columns=["first", "last"], unique=True)],
        )
        schema = DatabaseSchema(tables=[table], dialect="sqlite")
        state = {
            "users": [
                {"id": 1, "first": "A", "last": "Z"},
                {"id": 2, "first": "B", "last": "Y"},
                {"id": 3, "first": "C", "last": "X"},
            ]
        }

        new_rows = regenerate_table(schema, state, "users", seed=5)

        tuples = [(r["first"], r["last"]) for r in new_rows]
        assert len(tuples) == len(set(tuples))
