"""Unit tests for SampleExtractor with all DB calls mocked."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import sqlalchemy as sa

from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    ForeignKeySchema,
    TableSchema,
)
from dbsprout.train.extractor import SampleExtractor
from dbsprout.train.models import ExtractorConfig

if TYPE_CHECKING:
    from pathlib import Path


def _two_table_schema() -> DatabaseSchema:
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True)
        ],
        primary_key=["id"],
        foreign_keys=[],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False, primary_key=True),
            ColumnSchema(name="user_id", data_type=ColumnType.INTEGER, nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(columns=["user_id"], ref_table="users", ref_columns=["id"]),
        ],
    )
    return DatabaseSchema(dialect="sqlite", tables=[users, orders])


def _patch_engine_internals(
    *,
    users_df: pl.DataFrame,
    orders_df: pl.DataFrame,
    fetched_by_pk: pl.DataFrame,
):
    """Helper: patch the orchestrator's engine-side calls.

    Returns a tuple of context managers that the test composes with ``with``.
    """
    schema = _two_table_schema()

    def fake_random(engine, *, table, n, seed, row_count):
        return users_df if table.name == "users" else orders_df

    def fake_fetch(engine, *, table, pk_column, values):
        return fetched_by_pk

    def fake_table_factory(name, _md, **_kwargs):
        mock = MagicMock()
        mock.name = name
        return mock

    return (
        patch("dbsprout.train.extractor.introspect_engine", return_value=schema),
        patch(
            "dbsprout.train.extractor._row_counts",
            return_value={"users": 100, "orders": 100},
        ),
        patch("dbsprout.train.extractor._fetch_random", side_effect=fake_random),
        patch("dbsprout.train.extractor._fetch_by_pk", side_effect=fake_fetch),
        patch("dbsprout.train.extractor.sa.create_engine"),
        patch("dbsprout.train.extractor.sa.Table", side_effect=fake_table_factory),
    )


def test_extractor_writes_per_table_parquet_and_manifest(tmp_path: Path) -> None:
    cfg = ExtractorConfig(
        sample_rows=10,
        output_dir=tmp_path,
        seed=1,
        max_per_table=10,
        quiet=True,
    )
    patches = _patch_engine_internals(
        users_df=pl.DataFrame({"id": [1, 2]}),
        orders_df=pl.DataFrame({"id": [10, 11], "user_id": [1, 2]}),
        fetched_by_pk=pl.DataFrame(schema={"id": pl.Int64}),
    )
    p_introspect, p_counts, p_random, p_fetch, p_engine, p_table = patches
    with p_introspect, p_counts, p_random, p_fetch, p_engine as create_engine_mock, p_table:
        engine = create_engine_mock.return_value
        engine.dialect.name = "sqlite"
        result = SampleExtractor().extract(source="sqlite:///:memory:", config=cfg)

    assert (tmp_path / "samples" / "users.parquet").exists()
    assert (tmp_path / "samples" / "orders.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    assert {r.table for r in result.tables} == {"users", "orders"}


def test_extractor_records_closure_additions(tmp_path: Path) -> None:
    cfg = ExtractorConfig(
        sample_rows=10,
        output_dir=tmp_path,
        seed=1,
        max_per_table=10,
        quiet=True,
    )
    patches = _patch_engine_internals(
        users_df=pl.DataFrame({"id": [1]}),
        orders_df=pl.DataFrame({"id": [10, 11], "user_id": [1, 2]}),
        fetched_by_pk=pl.DataFrame({"id": [2]}),
    )
    p_introspect, p_counts, p_random, p_fetch, p_engine, p_table = patches
    with p_introspect, p_counts, p_random, p_fetch, p_engine as create_engine_mock, p_table:
        engine = create_engine_mock.return_value
        engine.dialect.name = "sqlite"
        result = SampleExtractor().extract(source="sqlite:///:memory:", config=cfg)

    by_table = {r.table: r for r in result.tables}
    assert by_table["users"].fk_closure_added == 1


def test_extractor_disposes_engine_on_failure(tmp_path: Path) -> None:
    """Engine.dispose() must run even when an inner step raises."""
    cfg = ExtractorConfig(
        sample_rows=10,
        output_dir=tmp_path,
        seed=1,
        max_per_table=10,
        quiet=True,
    )
    schema = _two_table_schema()
    with (
        patch("dbsprout.train.extractor.introspect_engine", return_value=schema),
        patch(
            "dbsprout.train.extractor._row_counts",
            side_effect=RuntimeError("boom"),
        ),
        patch("dbsprout.train.extractor.sa.create_engine") as create_engine_mock,
    ):
        engine = create_engine_mock.return_value
        engine.dialect.name = "sqlite"
        with pytest.raises(RuntimeError, match="boom"):
            SampleExtractor().extract(source="sqlite:///:memory:", config=cfg)
        engine.dispose.assert_called_once()


def test_extractor_creates_single_engine(tmp_path: Path) -> None:
    """extract() must build exactly one engine and dispose it (no leak).

    Uses a real on-disk SQLite database and spies on ``sa.create_engine`` so
    the schema introspection step reuses the extractor's engine instead of
    opening a second pool to the same database.
    """
    db_path = tmp_path / "src.db"
    url = f"sqlite:///{db_path}"
    setup = sa.create_engine(url)
    with setup.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(sa.text("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')"))
    setup.dispose()

    created: list[MagicMock] = []
    real_create_engine = sa.create_engine

    def _spy(*args: object, **kwargs: object) -> object:
        engine = real_create_engine(*args, **kwargs)  # type: ignore[arg-type]
        created.append(engine)
        return engine

    cfg = ExtractorConfig(
        sample_rows=2,
        output_dir=tmp_path / "out",
        seed=1,
        min_per_table=1,
        max_per_table=2,
        quiet=True,
    )
    with patch("dbsprout.train.extractor.sa.create_engine", side_effect=_spy):
        result = SampleExtractor().extract(source=url, config=cfg)

    assert {r.table for r in result.tables} == {"users"}
    assert len(created) == 1, "extract() must create exactly one engine"
