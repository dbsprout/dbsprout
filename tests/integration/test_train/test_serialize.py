"""Integration: S-062 extract -> S-063 serialize end-to-end on a SQLite fixture."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

from dbsprout.train.extractor import SampleExtractor
from dbsprout.train.models import ExtractorConfig, NullPolicy
from dbsprout.train.serializer import DataPreparer

if TYPE_CHECKING:
    from pathlib import Path


def _extract(sqlite_db: str, out_dir: Path) -> None:
    SampleExtractor().extract(
        source=sqlite_db,
        config=ExtractorConfig(
            sample_rows=40,
            output_dir=out_dir,
            seed=5,
            max_per_table=20,
        ),
    )


def test_extract_then_serialize_end_to_end(sqlite_db: str, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _extract(sqlite_db, run_dir)

    out = tmp_path / "data.jsonl"
    result = DataPreparer().prepare(
        input_dir=run_dir,
        output_path=out,
        seed=5,
        null_policy=NullPolicy.SKIP,
        quiet=True,
    )

    fixture_tables = {"users", "products", "orders", "order_items"}
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one serialized row"
    assert len(lines) == result.total_rows
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"text", "table"}
        assert obj["table"] in fixture_tables
        assert obj["text"].startswith(f"[{obj['table']}]")
    assert {t.table for t in result.tables} <= fixture_tables


def test_serialize_is_byte_identical_across_runs(sqlite_db: str, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _extract(sqlite_db, run_dir)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    DataPreparer().prepare(
        input_dir=run_dir, output_path=out_a, seed=9, null_policy=NullPolicy.SKIP, quiet=True
    )
    DataPreparer().prepare(
        input_dir=run_dir, output_path=out_b, seed=9, null_policy=NullPolicy.SKIP, quiet=True
    )
    assert out_a.read_bytes() == out_b.read_bytes()


def test_serialize_stable_under_different_pythonhashseed(sqlite_db: str, tmp_path: Path) -> None:
    """AC-3 hardening: per-row shuffle must not depend on PYTHONHASHSEED."""
    run_dir = tmp_path / "run"
    _extract(sqlite_db, run_dir)
    samples_dir = run_dir / "samples"

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from dbsprout.train.serializer import DataPreparer
        from dbsprout.train.models import NullPolicy
        DataPreparer().prepare(
            input_dir=Path(sys.argv[1]),
            output_path=Path(sys.argv[2]),
            seed=13,
            null_policy=NullPolicy.SKIP,
            quiet=True,
        )
        """
    )

    out1 = tmp_path / "h1.jsonl"
    out2 = tmp_path / "h2.jsonl"
    for env_seed, target in (("0", out1), ("12345", out2)):
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", script, str(run_dir), str(target)],
            check=True,
            env={"PYTHONHASHSEED": env_seed, "PATH": ""},
            cwd=str(samples_dir.parent.parent),
        )
    assert out1.read_bytes() == out2.read_bytes()
