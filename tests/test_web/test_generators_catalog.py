"""``GET /api/generators`` — generator catalogue for the Studio picker (S-120).

The catalogue is derived from the plugin/heuristic registries and surfaces
the methods the Studio method-picker offers when a user clicks a column's
method pill. The endpoint is read-only and does not depend on workspace
state (a connect / schema-load is not required).

Shape:

```json
{
  "providers": ["builtin", "mimesis", "faker", "numpy"],
  "methods": [
    {
      "provider": "mimesis",
      "method": "email",
      "description": "Mimesis email",
      "dtypes": ["VARCHAR", "TEXT"],
      "params": []
    },
    ...
  ]
}
```

* Optional ``?dtype=VARCHAR`` filter narrows ``methods`` to those that
  apply to that column type. Unknown dtypes → ``422 INVALID_DTYPE``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("fastapi", reason="fastapi absent (pip install dbsprout[web])")

from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI


def _make_app(tmp_path: Path) -> FastAPI:
    from dbsprout.web.app import create_app  # noqa: PLC0415

    return create_app(state_db_path=tmp_path / "state.db")


def test_get_generators_returns_envelope(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("providers"), list)
    assert isinstance(payload.get("methods"), list)
    assert payload["providers"], "providers must be non-empty"
    assert payload["methods"], "methods must be non-empty"


def test_get_generators_contains_canonical_entries(tmp_path: Path) -> None:
    """Catalogue must surface well-known methods used elsewhere in tests."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    assert ("mimesis", "email") in keys
    assert ("mimesis", "first_name") in keys
    assert ("builtin", "random_int") in keys
    assert ("builtin", "uuid4") in keys


def test_get_generators_method_carries_dtypes_and_description(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    methods = resp.json()["methods"]
    email = next(m for m in methods if m["provider"] == "mimesis" and m["method"] == "email")
    assert isinstance(email.get("description"), str)
    assert email["description"]
    assert isinstance(email.get("dtypes"), list)
    assert "VARCHAR" in email["dtypes"]
    assert isinstance(email.get("params"), list)


def test_get_generators_filter_varchar_excludes_random_int(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators", params={"dtype": "VARCHAR"})
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    assert ("mimesis", "email") in keys
    assert ("builtin", "random_int") not in keys


def test_get_generators_filter_integer_excludes_email(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators", params={"dtype": "INTEGER"})
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    assert ("builtin", "random_int") in keys
    assert ("mimesis", "email") not in keys


def test_get_generators_bogus_dtype_422(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators", params={"dtype": "BOGUS"})
    assert resp.status_code == 422
    body = resp.json()
    detail = body.get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "INVALID_DTYPE"


def test_get_generators_filter_case_insensitive(tmp_path: Path) -> None:
    """``?dtype=varchar`` should work just like ``VARCHAR``."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators", params={"dtype": "varchar"})
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    assert ("mimesis", "email") in keys


def test_get_generators_covers_every_pattern(tmp_path: Path) -> None:
    """Catalogue must reflect ``PATTERNS`` — guards against drift."""
    from dbsprout.spec.patterns import PATTERNS  # noqa: PLC0415

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    for pattern in PATTERNS:
        assert (pattern.provider, pattern.generator_name) in keys


def test_get_generators_works_without_workspace_schema(tmp_path: Path) -> None:
    """Read-only catalogue must not depend on a loaded schema (S-118 contract)."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    assert resp.status_code == 200


def test_get_generators_empty_dtype_returns_full_catalogue(tmp_path: Path) -> None:
    """An empty ``?dtype=`` query string is treated as "no filter"."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        # Whitespace-only value normalizes to empty after strip.
        resp = client.get("/api/generators", params={"dtype": "   "})
    assert resp.status_code == 200
    keys = {(m["provider"], m["method"]) for m in resp.json()["methods"]}
    assert ("mimesis", "email") in keys
    assert ("builtin", "random_int") in keys


# ── S-123: human-readable descriptions for the picker tooltip ─────────────


def test_known_methods_have_human_descriptions(tmp_path: Path) -> None:
    """The API echoes the curated descriptions added in S-123.

    The Studio picker uses ``entry.description`` as the ``title=`` tooltip
    on each method button. Make sure at least the high-traffic methods
    return descriptions richer than the legacy ``"Mimesis email"`` placeholder.
    """
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/generators")
    assert resp.status_code == 200
    by_key = {(m["provider"], m["method"]): m["description"] for m in resp.json()["methods"]}
    # ``email`` description should mention "email" and be longer than the
    # legacy provider-cap placeholder.
    email_desc = by_key[("mimesis", "email")]
    assert "email" in email_desc.lower()
    assert len(email_desc) > len("Mimesis email"), email_desc
    # ``random_int`` should reference integers.
    int_desc = by_key[("builtin", "random_int")]
    assert "integer" in int_desc.lower(), int_desc
