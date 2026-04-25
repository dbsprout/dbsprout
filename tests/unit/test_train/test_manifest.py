"""Unit tests for manifest JSON read/write."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dbsprout.train.manifest import read_manifest, write_manifest
from dbsprout.train.models import SampleManifest, TableExtractionResult


def _sample_manifest() -> SampleManifest:
    return SampleManifest(
        dbsprout_version="0.1.0",
        extracted_at=datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
        dialect="sqlite",
        schema_hash="deadbeef",
        seed=42,
        requested_budget=100,
        effective_budget=98,
        tables=(
            TableExtractionResult(
                table="users",
                target=50,
                sampled=50,
                fk_closure_added=2,
                parquet_path=Path("samples/users.parquet"),
            ),
        ),
        fk_closure_iterations=2,
        fk_unresolved_per_table={},
        duration_seconds=0.42,
    )


def test_round_trip(tmp_path: Path) -> None:
    m = _sample_manifest()
    target = tmp_path / "manifest.json"
    write_manifest(target, m)
    loaded = read_manifest(target)
    assert loaded == m


def test_writer_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "manifest.json"
    write_manifest(target, _sample_manifest())
    assert target.exists()


def test_reader_rejects_future_version(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    raw = _sample_manifest().model_dump(mode="json")
    raw["manifest_version"] = 99
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest_version 99"):
        read_manifest(target)
