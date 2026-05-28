"""``POST /api/validate`` — fidelity + detection blocks (S-134).

S-133 introduced the integrity report; S-134 extends the same endpoint with
two additional envelope keys, ``fidelity`` and ``detection``, populated from
the existing helpers in :mod:`dbsprout.quality.fidelity` and
:mod:`dbsprout.quality.detection` whenever reference rows are available on the
workspace. When reference rows are absent, both keys are present in the
envelope but ``None`` — clients never see a 500 from the no-reference path
and the JSON shape stays stable across both paths.

The graceful-degrade contract also covers the optional ``[stats]`` extra:
``validate_fidelity`` raises ``ImportError`` without ``scipy``, and
``validate_detection`` raises ``ImportError`` without ``scikit-learn``. The
router catches ``ImportError`` and degrades the affected block to ``None``.

Tests seed reference rows via ``app.state.workspace.set_reference_data`` (the
seam introduced for S-134) and synthetic data via ``set_last_result`` exactly
as the S-133 tests do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

from dbsprout.generate.orchestrator import GenerateResult
from dbsprout.schema.models import ColumnSchema, ColumnType, DatabaseSchema, TableSchema

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


# ── fixtures / helpers ─────────────────────────────────────────────────


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def _users_schema() -> DatabaseSchema:
    """Single-table schema with one numeric + one categorical column."""
    return DatabaseSchema(
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
                        name="age",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                    ),
                    ColumnSchema(
                        name="city",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            )
        ]
    )


def _synthetic_result(n: int = 40) -> GenerateResult:
    """A modest synthetic dataset large enough for the C2ST 5-fold CV (>=10/class)."""
    users = [
        {"id": i, "age": 20 + (i % 30), "city": "NYC" if i % 2 == 0 else "LA"}
        for i in range(1, n + 1)
    ]
    return GenerateResult(
        tables_data={"users": users},
        insertion_order=["users"],
        total_rows=n,
        total_tables=1,
        duration_seconds=0.0,
    )


def _reference_payload(n: int = 40) -> dict[str, list[dict[str, Any]]]:
    """Reference rows aligned to ``_users_schema`` and ``_synthetic_result``."""
    return {
        "users": [
            {"id": i, "age": 25 + (i % 25), "city": "NYC" if i % 3 == 0 else "LA"}
            for i in range(1, n + 1)
        ]
    }


def _seed(
    app: FastAPI,
    *,
    schema: DatabaseSchema | None = None,
    result: GenerateResult | None = None,
    reference: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    ws = app.state.workspace
    if schema is not None:
        ws.set_schema(schema)
    if result is not None:
        ws.set_last_result(result)
    if reference is not None:
        ws.set_reference_data(reference)


# ── envelope shape: keys always present ────────────────────────────────


def test_validate_envelope_always_has_fidelity_and_detection_keys(tmp_path: Path) -> None:
    """AC: ``fidelity`` + ``detection`` keys exist in every 200 envelope,
    even when reference rows are absent (both then ``None``)."""
    app = _make_app(tmp_path)
    _seed(app, schema=_users_schema(), result=_synthetic_result())

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "fidelity" in body
    assert "detection" in body


# ── no-reference path: both blocks ``None`` (no 500) ────────────────────


def test_validate_fidelity_null_when_no_reference(tmp_path: Path) -> None:
    """AC: no reference rows → ``body['fidelity'] is None`` (no 500)."""
    app = _make_app(tmp_path)
    _seed(app, schema=_users_schema(), result=_synthetic_result())

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    assert resp.json()["fidelity"] is None


def test_validate_detection_null_when_no_reference(tmp_path: Path) -> None:
    """AC: no reference rows → ``body['detection'] is None`` (no 500)."""
    app = _make_app(tmp_path)
    _seed(app, schema=_users_schema(), result=_synthetic_result())

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    assert resp.json()["detection"] is None


# ── populated path: scores returned from existing helpers ──────────────


@pytest.mark.skipif(
    pytest.importorskip("scipy", reason="scipy absent ([stats] extra)") is None,
    reason="scipy absent",
)
def test_validate_fidelity_block_populated_when_reference_seeded(tmp_path: Path) -> None:
    """AC: with reference seeded, ``body['fidelity']`` carries ``overall_score``,
    ``passed``, and a non-empty ``metrics`` list reused from ``validate_fidelity``."""
    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    fid = body["fidelity"]
    assert fid is not None
    assert set(fid) == {"overall_score", "passed", "metrics"}
    assert 0.0 <= fid["overall_score"] <= 1.0
    assert isinstance(fid["passed"], bool)
    assert isinstance(fid["metrics"], list)
    assert fid["metrics"], "fidelity metrics list must be non-empty when reference is seeded"
    m0 = fid["metrics"][0]
    assert {"metric", "table", "column", "score", "details"} <= set(m0)


@pytest.mark.skipif(
    pytest.importorskip("sklearn", reason="scikit-learn absent ([stats] extra)") is None,
    reason="scikit-learn absent",
)
def test_validate_detection_block_populated_when_reference_seeded(tmp_path: Path) -> None:
    """AC: with reference seeded, ``body['detection']`` carries ``overall_score``,
    ``passed``, and a ``metrics`` list reused from ``validate_detection``."""
    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    det = body["detection"]
    assert det is not None
    assert set(det) == {"overall_score", "passed", "metrics"}
    assert 0.0 <= det["overall_score"] <= 1.0
    assert isinstance(det["passed"], bool)
    assert isinstance(det["metrics"], list)
    if det["metrics"]:  # C2ST may skip tables with too few rows; assert shape if present
        m0 = det["metrics"][0]
        assert {"metric", "table", "accuracy", "details"} <= set(m0)


# ── graceful-degrade: optional [stats] extra missing ───────────────────


def test_validate_fidelity_null_on_helper_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: if the fidelity helper raises ``ImportError`` (no scipy), the route
    must still return 200 with ``body['fidelity'] is None`` — never a 500."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        msg = "scipy missing (simulated)"
        raise ImportError(msg)

    monkeypatch.setattr("dbsprout.quality.fidelity.validate_fidelity", _raise)

    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    assert resp.json()["fidelity"] is None


def test_validate_detection_null_on_helper_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: if the detection helper raises ``ImportError`` (no sklearn), the
    route must still return 200 with ``body['detection'] is None`` — never a 500."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        msg = "sklearn missing (simulated)"
        raise ImportError(msg)

    monkeypatch.setattr("dbsprout.quality.detection.validate_detection", _raise)

    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate")
    assert resp.status_code == 200, resp.text
    assert resp.json()["detection"] is None


# ── HTMX fragment renders metrics blocks ───────────────────────────────


def test_validate_htmx_omits_blocks_when_null(tmp_path: Path) -> None:
    """AC: HTMX response omits the fidelity/detection regions when reference
    rows are absent — the existing integrity panel still renders."""
    app = _make_app(tmp_path)
    _seed(app, schema=_users_schema(), result=_synthetic_result())

    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "data-fidelity-metric" not in body
    assert "data-detection-metric" not in body
    assert "Integrity report" in body  # S-133 panel still renders


@pytest.mark.skipif(
    pytest.importorskip("scipy", reason="scipy absent ([stats] extra)") is None,
    reason="scipy absent",
)
def test_validate_htmx_renders_fidelity_block(tmp_path: Path) -> None:
    """AC: HTMX response surfaces a ``data-fidelity-metric`` row per scored
    column when the fidelity block is populated."""
    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "data-fidelity-metric" in body
    assert "Fidelity" in body


@pytest.mark.skipif(
    pytest.importorskip("sklearn", reason="scikit-learn absent ([stats] extra)") is None,
    reason="scikit-learn absent",
)
def test_validate_htmx_renders_detection_block(tmp_path: Path) -> None:
    """AC: HTMX response surfaces a detection region with overall accuracy
    when the detection block is populated (per-table rows depend on enough
    rows for 5-fold CV; the overall badge is always present when populated)."""
    app = _make_app(tmp_path)
    _seed(
        app,
        schema=_users_schema(),
        result=_synthetic_result(),
        reference=_reference_payload(),
    )

    resp = TestClient(app).post("/api/validate", headers={"HX-Request": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "Detection" in body
    # ``data-detection-metric`` rows render whenever per-table metrics exist;
    # with our fixture (n=40, 2 classes after split) C2ST will emit at least one.
    assert "data-detection-metric" in body
