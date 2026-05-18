"""Unit tests for the Mermaid ERD builder (S-082)."""

from __future__ import annotations

import re

from dbsprout.report.erd import (
    MAX_COLUMNS_PER_ENTITY,
    _mermaid_type,
    build_erd_mermaid,
)
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)

_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
_REL_TOKENS = ("||--o{", "||--||", "}o--o{")


def _col(
    name: str,
    dtype: ColumnType = ColumnType.VARCHAR,
    *,
    pk: bool = False,
    nullable: bool = True,
    unique: bool = False,
) -> ColumnSchema:
    return ColumnSchema(
        name=name,
        data_type=dtype,
        primary_key=pk,
        nullable=nullable,
        unique=unique,
    )


def _users_orders_schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            _col("id", ColumnType.INTEGER, pk=True, nullable=False),
            _col("email", ColumnType.VARCHAR),
            _col("name", ColumnType.VARCHAR),
        ],
        primary_key=["id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            _col("id", ColumnType.INTEGER, pk=True, nullable=False),
            _col("user_id", ColumnType.INTEGER, nullable=False),
            _col("created_at", ColumnType.TIMESTAMP),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                name="fk_orders_user",
                columns=["user_id"],
                ref_table="users",
                ref_columns=["id"],
            )
        ],
    )
    return DatabaseSchema(tables=[users, orders])


class TestSkeletonAndTypeMapping:
    def test_empty_schema_returns_erdiagram_header(self) -> None:
        out = build_erd_mermaid(DatabaseSchema(tables=[]))
        assert out.splitlines()[0].strip() == "erDiagram"

    def test_every_columntype_maps_to_safe_token(self) -> None:
        for ct in ColumnType:
            token = _mermaid_type(ct)
            assert _IDENT_RE.match(token), f"{ct} -> {token!r} not safe"

    def test_varchar_maps_to_varchar(self) -> None:
        assert _mermaid_type(ColumnType.VARCHAR) == "varchar"


class TestEntities:
    def test_entities_list_columns_with_types(self) -> None:
        out = build_erd_mermaid(_users_orders_schema())
        assert "users {" in out
        assert "orders {" in out
        assert "integer id PK" in out
        assert "varchar email" in out
        assert "integer user_id FK" in out

    def test_pk_wins_over_fk_marker(self) -> None:
        # A column that is both PK and FK shows PK (single key marker).
        child = TableSchema(
            name="profile",
            columns=[_col("user_id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["user_id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        users = TableSchema(
            name="users",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[users, child]))
        assert "integer user_id PK" in out
        assert "user_id FK" not in out


class TestRelationships:
    def test_plain_fk_is_one_to_many(self) -> None:
        out = build_erd_mermaid(_users_orders_schema())
        assert "users ||--o{ orders" in out
        assert "fk_orders_user" in out

    def test_unique_fk_is_one_to_one(self) -> None:
        users = TableSchema(
            name="users",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )
        profile = TableSchema(
            name="profile",
            columns=[
                _col("id", ColumnType.INTEGER, pk=True, nullable=False),
                _col("user_id", ColumnType.INTEGER, nullable=False, unique=True),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"])
            ],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[users, profile]))
        assert "users ||--|| profile" in out

    def test_junction_table_is_many_to_many(self) -> None:
        users = TableSchema(
            name="users",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )
        roles = TableSchema(
            name="roles",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )
        user_roles = TableSchema(
            name="user_roles",
            columns=[
                _col("user_id", ColumnType.INTEGER, pk=True, nullable=False),
                _col("role_id", ColumnType.INTEGER, pk=True, nullable=False),
            ],
            primary_key=["user_id", "role_id"],
            foreign_keys=[
                ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
                ForeignKeySchema(columns=["role_id"], ref_table="roles", ref_columns=["id"]),
            ],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[users, roles, user_roles]))
        assert "}o--o{" in out
        # Parents rendered alphabetically for deterministic output.
        assert "roles }o--o{ users" in out
        assert "user_roles" in out

    def test_self_referential_fk(self) -> None:
        emp = TableSchema(
            name="employees",
            columns=[
                _col("id", ColumnType.INTEGER, pk=True, nullable=False),
                _col("manager_id", ColumnType.INTEGER),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["manager_id"],
                    ref_table="employees",
                    ref_columns=["id"],
                )
            ],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[emp]))
        assert "employees {" in out
        assert "employees ||--o{ employees" in out

    def test_composite_fk_is_one_to_many(self) -> None:
        # Multi-column FK: no single fk_col, so cardinality defaults to 1:m.
        parent = TableSchema(
            name="parent",
            columns=[
                _col("a", ColumnType.INTEGER, pk=True, nullable=False),
                _col("b", ColumnType.INTEGER, pk=True, nullable=False),
            ],
            primary_key=["a", "b"],
        )
        child = TableSchema(
            name="child",
            columns=[
                _col("id", ColumnType.INTEGER, pk=True, nullable=False),
                _col("pa", ColumnType.INTEGER),
                _col("pb", ColumnType.INTEGER),
            ],
            primary_key=["id"],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["pa", "pb"],
                    ref_table="parent",
                    ref_columns=["a", "b"],
                )
            ],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[parent, child]))
        assert "parent ||--o{ child" in out

    def test_duplicate_m2m_edge_emitted_once(self) -> None:
        # Two junction tables between the same pair → one m2m edge only.
        users = TableSchema(
            name="users",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )
        roles = TableSchema(
            name="roles",
            columns=[_col("id", ColumnType.INTEGER, pk=True, nullable=False)],
            primary_key=["id"],
        )

        def _junction(jname: str) -> TableSchema:
            return TableSchema(
                name=jname,
                columns=[
                    _col("user_id", ColumnType.INTEGER, pk=True, nullable=False),
                    _col("role_id", ColumnType.INTEGER, pk=True, nullable=False),
                ],
                primary_key=["user_id", "role_id"],
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        ref_table="users",
                        ref_columns=["id"],
                    ),
                    ForeignKeySchema(
                        columns=["role_id"],
                        ref_table="roles",
                        ref_columns=["id"],
                    ),
                ],
            )

        schema = DatabaseSchema(tables=[users, roles, _junction("ur1"), _junction("ur2")])
        out = build_erd_mermaid(schema)
        assert out.count("roles }o--o{ users") == 1


class TestSanitisationAndCap:
    def test_non_conforming_identifiers_sanitised(self) -> None:
        t = TableSchema(
            name="my table.v2",
            columns=[_col("first name", ColumnType.VARCHAR)],
        )
        out = build_erd_mermaid(DatabaseSchema(tables=[t]))
        assert "my_table_v2 {" in out
        assert "varchar first_name" in out
        assert "my table.v2" not in out

    def test_large_table_column_cap(self) -> None:
        cols = [_col(f"c{i}", ColumnType.INTEGER) for i in range(MAX_COLUMNS_PER_ENTITY + 5)]
        t = TableSchema(name="wide", columns=cols)
        out = build_erd_mermaid(DatabaseSchema(tables=[t]))
        entity_block = out.split("wide {", 1)[1].split("}", 1)[0]
        rendered_cols = [ln for ln in entity_block.splitlines() if ln.strip()]
        # cap rows + 1 sentinel row
        assert len(rendered_cols) == MAX_COLUMNS_PER_ENTITY + 1
        assert "more" in entity_block


class TestLargeSchemaWellFormed:
    def test_25_table_schema_well_formed(self) -> None:
        tables: list[TableSchema] = []
        for i in range(25):
            fks = []
            if i > 0:
                fks.append(
                    ForeignKeySchema(
                        name=f"fk_t{i}_parent",
                        columns=["parent_id"],
                        ref_table=f"t{i - 1}",
                        ref_columns=["id"],
                    )
                )
            cols = [
                _col("id", ColumnType.INTEGER, pk=True, nullable=False),
                _col("name", ColumnType.VARCHAR),
            ]
            if i > 0:
                cols.append(_col("parent_id", ColumnType.INTEGER))
            tables.append(
                TableSchema(
                    name=f"t{i}",
                    columns=cols,
                    primary_key=["id"],
                    foreign_keys=fks,
                )
            )
        out = build_erd_mermaid(DatabaseSchema(tables=tables))
        for i in range(25):
            assert f"t{i} {{" in out
        # exactly one erDiagram header
        assert out.count("erDiagram") == 1
        # entity blocks: one opener + one closer per table (ignore the
        # cardinality tokens, which legitimately contain stray braces).
        openers = [ln for ln in out.splitlines() if ln.strip().endswith("{")]
        closers = [ln for ln in out.splitlines() if ln.strip() == "}"]
        assert len(openers) == 25
        assert len(closers) == 25
        # every relationship line uses a valid cardinality token
        for line in out.splitlines():
            s = line.strip()
            if " : " in s and "--" in s:
                assert any(tok in s for tok in _REL_TOKENS), s
