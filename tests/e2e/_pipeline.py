"""Shared E2E pipeline harness: init --file -> generate -> validate.

Runs the real dbsprout CLI via Typer's CliRunner against an isolated
working directory, then exposes the parsed quality report, the schema,
and the generated seed rows for assertions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.quality.integrity import validate_integrity
from dbsprout.schema.models import DatabaseSchema

_runner = CliRunner()

_EXPECTED_CHECK_KINDS = {
    "fk_satisfaction",
    "pk_uniqueness",
    "unique",
    "not_null",
}


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of one init -> generate -> validate run."""

    validate_exit: int
    report: dict[str, Any]
    seed_data: dict[str, list[dict[str, Any]]]
    schema: DatabaseSchema


def _invoke(args: list[str]) -> None:
    res = _runner.invoke(app, args)
    if res.exit_code != 0:
        raise AssertionError(
            f"`dbsprout {' '.join(args)}` exited {res.exit_code}\n{res.output}"
        )


def _extract_bare_schema(work_dir: Path) -> DatabaseSchema:
    """init writes a wrapped {metadata, schema} snapshot; validate needs a
    bare DatabaseSchema JSON. Extract it to .dbsprout/schema.json."""
    snap_dir = work_dir / ".dbsprout" / "snapshots"
    snaps = sorted(snap_dir.glob("*.json"))
    if not snaps:
        raise AssertionError(f"no snapshot written under {snap_dir}")
    wrapped = json.loads(snaps[-1].read_text(encoding="utf-8"))
    bare = wrapped["schema"]
    schema_path = work_dir / ".dbsprout" / "schema.json"
    schema_path.write_text(json.dumps(bare), encoding="utf-8")
    return DatabaseSchema.model_validate(bare)


def _read_seed_data(seeds_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Generate writes NNN_<table>.json (a list of row dicts) per table."""
    data: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(seeds_dir.glob("*.json")):
        table = f.stem.split("_", 1)[1]
        data[table] = json.loads(f.read_text(encoding="utf-8"))
    return data


def run_pipeline(
    fixture: Path,
    work_dir: Path,
    *,
    rows: int = 50,
    seed: int = 42,
) -> PipelineResult:
    """Run init --file -> generate -> validate inside ``work_dir``.

    ``generate``/``validate`` resolve snapshots relative to CWD, so this
    chdirs into ``work_dir`` for the CLI calls and restores CWD after.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = work_dir / "seeds"
    report_path = work_dir / "report.json"

    prev_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        _invoke(["init", "--file", str(fixture), "--output-dir", str(work_dir)])
        schema = _extract_bare_schema(work_dir)
        _invoke(
            [
                "generate",
                "--rows",
                str(rows),
                "--seed",
                str(seed),
                "--output-format",
                "json",
                "--output-dir",
                str(seeds_dir),
            ]
        )
        res = _runner.invoke(
            app,
            [
                "validate",
                "--rows",
                str(rows),
                "--seed",
                str(seed),
                "--format",
                "json",
                "--output",
                str(report_path),
            ],
        )
    finally:
        os.chdir(prev_cwd)

    if not report_path.exists():
        raise AssertionError(
            f"validate wrote no report (exit {res.exit_code})\n{res.output}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seed_data = _read_seed_data(seeds_dir)
    return PipelineResult(
        validate_exit=res.exit_code,
        report=report,
        seed_data=seed_data,
        schema=schema,
    )


def assert_full_integrity(result: PipelineResult) -> None:
    """CLI-level: validate exited 0, every integrity check passed, and all
    four check kinds were exercised (guards against an empty check list)."""
    assert result.validate_exit == 0, result.report
    assert result.report["passed"] is True, result.report
    integrity = result.report["integrity"]
    assert integrity["passed"] is True, integrity
    checks = integrity["checks"]
    assert checks, "no integrity checks ran"
    failed = [c for c in checks if not c["passed"]]
    assert not failed, f"failed checks: {failed}"
    kinds = {c["check"] for c in checks}
    assert _EXPECTED_CHECK_KINDS <= kinds, (
        f"missing check kinds: {_EXPECTED_CHECK_KINDS - kinds}"
    )


def assert_programmatic_integrity(result: PipelineResult) -> None:
    """Defense-in-depth: re-validate the on-disk seed files directly via
    the S-022 API, independent of validate's internal regeneration."""
    report = validate_integrity(result.seed_data, result.schema)
    failed = [c for c in report.checks if not c.passed]
    assert report.passed, f"programmatic integrity failed: {failed}"
    assert report.checks, "programmatic validation ran no checks"
