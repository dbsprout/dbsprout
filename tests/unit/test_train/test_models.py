"""Unit tests for dbsprout.train.models Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dbsprout.train.models import (
    ExtractorConfig,
    SampleAllocation,
    SampleManifest,
    SampleResult,
    TableExtractionResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_extractor_config_defaults(tmp_path: Path) -> None:
    cfg = ExtractorConfig(
        db_url="sqlite:///:memory:",
        sample_rows=100,
        output_dir=tmp_path,
    )
    assert cfg.seed == 0
    assert cfg.min_per_table == 10
    assert cfg.max_per_table is None
    assert cfg.quiet is False
    assert cfg.fk_closure_max_iterations is None


def test_extractor_config_is_frozen(tmp_path: Path) -> None:
    cfg = ExtractorConfig(
        db_url="sqlite:///:memory:",
        sample_rows=1,
        output_dir=tmp_path,
    )
    with pytest.raises(ValidationError):
        cfg.sample_rows = 999  # type: ignore[misc]


def test_extractor_config_rejects_zero_sample_rows(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ExtractorConfig(
            db_url="sqlite:///:memory:",
            sample_rows=0,
            output_dir=tmp_path,
        )


def test_sample_allocation_round_trip() -> None:
    a = SampleAllocation(
        table="users",
        row_count=1000,
        target=100,
        floor_clamped=False,
        ceiling_clamped=False,
    )
    assert a.table == "users"
    assert a.target == 100


def test_table_extraction_result_defaults(tmp_path: Path) -> None:
    r = TableExtractionResult(
        table="users",
        target=100,
        sampled=99,
        parquet_path=tmp_path / "users.parquet",
    )
    assert r.fk_closure_added == 0


def test_sample_manifest_default_version_is_one() -> None:
    m = SampleManifest(
        dbsprout_version="0.1.0",
        extracted_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
        dialect="sqlite",
        schema_hash="abc",
        seed=0,
        requested_budget=100,
        effective_budget=100,
        tables=(),
        fk_closure_iterations=0,
        fk_unresolved_per_table={},
        duration_seconds=0.1,
    )
    assert m.manifest_version == 1


def test_sample_result_holds_tuple_of_results(tmp_path: Path) -> None:
    r = SampleResult(
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        tables=(),
        duration_seconds=0.0,
    )
    assert r.tables == ()
