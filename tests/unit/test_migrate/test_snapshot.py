"""Tests for dbsprout.migrate.snapshot — schema snapshot storage."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from dbsprout.migrate.snapshot import SnapshotInfo, SnapshotMetadata, SnapshotStore
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_schema() -> DatabaseSchema:
    """Single-table schema for snapshot tests."""
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    table = TableSchema(name="users", columns=[col], primary_key=["id"])
    return DatabaseSchema(tables=[table], dialect="sqlite")


@pytest.fixture
def second_schema() -> DatabaseSchema:
    """Different single-table schema to test distinct snapshots."""
    col = ColumnSchema(name="id", data_type=ColumnType.INTEGER, nullable=False)
    table = TableSchema(name="orders", columns=[col], primary_key=["id"])
    return DatabaseSchema(tables=[table], dialect="sqlite")


# ── SnapshotInfo model tests (AC-8) ────────────────────────────────────


class TestSnapshotInfoModel:
    def test_snapshot_info_fields(self, tmp_path: Path) -> None:
        dummy = tmp_path / "x.json"
        dummy.write_text("{}", encoding="utf-8")
        info = SnapshotInfo(
            path=dummy,
            schema_hash="abc12345deadbeef",
            timestamp=datetime(2026, 4, 8, tzinfo=timezone.utc),
            table_count=3,
        )
        assert info.table_count == 3
        assert info.schema_hash == "abc12345deadbeef"
        assert info.path == dummy

    def test_snapshot_info_is_frozen(self, tmp_path: Path) -> None:
        dummy = tmp_path / "x.json"
        dummy.write_text("{}", encoding="utf-8")
        info = SnapshotInfo(
            path=dummy,
            schema_hash="abc12345deadbeef",
            timestamp=datetime(2026, 4, 8, tzinfo=timezone.utc),
            table_count=3,
        )
        with pytest.raises(ValidationError):
            info.path = Path("/other")  # type: ignore[misc]


# ── SnapshotMetadata model tests (AC-9) ────────────────────────────────


class TestSnapshotMetadataModel:
    def test_snapshot_metadata_fields(self) -> None:
        meta = SnapshotMetadata(
            schema_hash="abc12345deadbeef",
            timestamp="2026-04-08T00:00:00Z",
            table_count=2,
            table_names=["a", "b"],
            dialect="postgresql",
        )
        assert meta.table_count == 2
        assert meta.table_names == ["a", "b"]
        assert meta.dialect == "postgresql"

    def test_snapshot_metadata_is_frozen(self) -> None:
        meta = SnapshotMetadata(
            schema_hash="abc12345deadbeef",
            timestamp="2026-04-08T00:00:00Z",
            table_count=0,
            table_names=[],
            dialect="sqlite",
        )
        with pytest.raises(ValidationError):
            meta.dialect = "pg"  # type: ignore[misc]

    def test_snapshot_metadata_dialect_nullable(self) -> None:
        meta = SnapshotMetadata(
            schema_hash="abc12345deadbeef",
            timestamp="2026-04-08T00:00:00Z",
            table_count=0,
            table_names=[],
            dialect=None,
        )
        assert meta.dialect is None


# ── SnapshotStore init tests (AC-10, AC-16) ────────────────────────────


class TestSnapshotStoreInit:
    def test_default_base_dir(self) -> None:
        store = SnapshotStore()
        assert store.base_dir == Path(".dbsprout") / "snapshots"

    def test_custom_base_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_snaps"
        store = SnapshotStore(base_dir=custom)
        assert store.base_dir == custom

    def test_save_creates_directory_on_first_call(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        snap_dir = tmp_path / "snaps"
        store = SnapshotStore(base_dir=snap_dir)
        assert not snap_dir.exists()
        store.save(minimal_schema)
        assert snap_dir.is_dir()


# ── SnapshotStore.save() tests (AC-2, AC-3, AC-11, AC-14) ──────────────


class TestSnapshotStoreSave:
    def test_save_returns_snapshot_info(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        assert isinstance(info, SnapshotInfo)
        assert info.path.exists()
        assert info.schema_hash == minimal_schema.schema_hash()
        assert info.table_count == 1

    def test_save_filename_format(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        name = info.path.name
        hash_prefix = minimal_schema.schema_hash()[:8]
        assert name.endswith(f"_{hash_prefix}.json")
        # Timestamp portion: YYYYMMDDTHHMMSSz = 16 chars
        ts_part = name.split("_")[0]
        assert len(ts_part) == 16

    def test_save_idempotent_same_hash(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info1 = store.save(minimal_schema)
        info2 = store.save(minimal_schema)
        assert info1.path == info2.path
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1

    def test_save_never_overwrites(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        mtime_before = info.path.stat().st_mtime_ns
        store.save(minimal_schema)
        assert info.path.stat().st_mtime_ns == mtime_before

    def test_save_rejects_empty_schema(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        empty = DatabaseSchema(tables=[], dialect="sqlite")
        with pytest.raises(ValueError, match=r"[Ee]mpty schema"):
            store.save(empty)

    def test_save_wrapper_json_format(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        """Saved file uses wrapper envelope: {metadata: ..., schema: ...}."""
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        raw = json.loads(info.path.read_text(encoding="utf-8"))
        assert "metadata" in raw
        assert "schema" in raw
        assert raw["metadata"]["schema_hash"] == minimal_schema.schema_hash()
        assert raw["metadata"]["table_count"] == 1
        assert raw["metadata"]["table_names"] == ["users"]
        assert raw["schema"]["tables"][0]["name"] == "users"


# ── SnapshotStore.list_snapshots() tests (AC-6, AC-15) ──────────────────


class TestSnapshotStoreListSnapshots:
    def test_list_empty(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        assert store.list_snapshots() == []

    def test_list_nonexistent_dir(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path / "does_not_exist")
        assert store.list_snapshots() == []

    def test_list_sorted_newest_first(
        self,
        tmp_path: Path,
        minimal_schema: DatabaseSchema,
        second_schema: DatabaseSchema,
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        store.save(minimal_schema)
        store.save(second_schema)
        results = store.list_snapshots()
        assert len(results) == 2
        assert results[0].timestamp >= results[1].timestamp

    def test_list_skips_tmp_files(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        store.save(minimal_schema)
        tmp_file = tmp_path / "20260101T000000Z_abcd1234.json.tmp"
        tmp_file.write_text("{}", encoding="utf-8")
        results = store.list_snapshots()
        assert len(results) == 1
        assert not any(r.path.name.endswith(".tmp") for r in results)

    def test_list_skips_corrupt_file(
        self,
        tmp_path: Path,
        minimal_schema: DatabaseSchema,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        store.save(minimal_schema)
        corrupt = tmp_path / "20260101T000000Z_deadbeef.json"
        corrupt.write_text("not valid json", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            results = store.list_snapshots()
        assert len(results) == 1
        assert any(
            "corrupt" in r.message.lower() or "skip" in r.message.lower() for r in caplog.records
        )


# ── SnapshotStore.load_latest() tests (AC-4) ────────────────────────────


class TestSnapshotStoreLoadLatest:
    def test_load_latest_none_when_empty(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        assert store.load_latest() is None

    def test_load_latest_returns_most_recent(
        self,
        tmp_path: Path,
        minimal_schema: DatabaseSchema,
        second_schema: DatabaseSchema,
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        store.save(minimal_schema)
        store.save(second_schema)
        loaded = store.load_latest()
        assert loaded is not None
        assert isinstance(loaded, DatabaseSchema)

    def test_load_latest_skips_corrupt_returns_valid(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        store.save(minimal_schema)
        corrupt = tmp_path / "29991231T235959Z_deadbeef.json"
        corrupt.write_text("{invalid}", encoding="utf-8")
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.tables[0].name == "users"


# ── SnapshotStore.load_by_hash() tests (AC-5) ───────────────────────────


class TestSnapshotStoreLoadByHash:
    def test_load_by_hash_returns_matching(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        loaded = store.load_by_hash(info.schema_hash[:8])
        assert loaded is not None
        assert loaded.table_names() == minimal_schema.table_names()

    def test_load_by_hash_returns_none_when_not_found(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        assert store.load_by_hash("00000000") is None

    def test_load_by_hash_nonexistent_dir(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path / "does_not_exist")
        assert store.load_by_hash("deadbeef") is None

    def test_load_by_hash_skips_tmp_files(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        h = minimal_schema.schema_hash()[:8]
        tmp_file = tmp_path / f"20260101T000000Z_{h}.json.tmp"
        tmp_file.write_text("{}", encoding="utf-8")
        assert store.load_by_hash(h) is None

    def test_load_by_hash_prefix_match(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        loaded = store.load_by_hash(info.schema_hash[:4])
        assert loaded is not None


# ── SnapshotStore.resolve() tests (AC-7) ────────────────────────────────


class TestSnapshotStoreResolve:
    def test_resolve_accepts_path(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        loaded = store.resolve(str(info.path))
        assert loaded is not None
        assert loaded.table_names() == minimal_schema.table_names()

    def test_resolve_accepts_hash_prefix(
        self, tmp_path: Path, minimal_schema: DatabaseSchema
    ) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        info = store.save(minimal_schema)
        loaded = store.resolve(info.schema_hash[:8])
        assert loaded is not None

    def test_resolve_returns_none_for_unknown(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        assert store.resolve("notexist") is None

    def test_resolve_returns_none_for_corrupt_path(self, tmp_path: Path) -> None:
        store = SnapshotStore(base_dir=tmp_path)
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("not valid json at all", encoding="utf-8")
        assert store.resolve(str(corrupt)) is None


# ── Legacy format backward compatibility ─────────────────────────────────


class TestLegacyFormat:
    def test_load_legacy_flat_format(self, tmp_path: Path, minimal_schema: DatabaseSchema) -> None:
        """Bare DatabaseSchema JSON (old _write_snapshot) loads correctly."""
        store = SnapshotStore(base_dir=tmp_path)
        legacy_path = tmp_path / f"20260101T000000Z_{minimal_schema.schema_hash()[:8]}.json"
        legacy_path.write_text(minimal_schema.model_dump_json(indent=2), encoding="utf-8")
        loaded = store.load_latest()
        assert loaded is not None
        assert loaded.table_names() == minimal_schema.table_names()
