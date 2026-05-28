"""``GET /api/generators`` — generator catalogue endpoint (S-120).

The Studio method-picker opens when a user clicks a column's method pill
and lets them swap the generator without memorising names. This router
surfaces the catalogue the picker reads from — derived from the heuristic
registries in :mod:`dbsprout.spec.catalog`, not a hand-rolled list.

Endpoint
--------

* ``GET /api/generators`` — JSON catalogue:

  ```json
  {
    "providers": ["builtin", "faker", "mimesis", "numpy"],
    "methods": [
      {
        "provider": "mimesis",
        "method": "email",
        "description": "Valid-looking email address (e.g. 'user@example.com').",
        "example": "ada@example.com",
        "dtypes": ["VARCHAR", "TEXT"],
        "params": []
      },
      ...
    ]
  }
  ```

* Optional ``?dtype=<ColumnType.name>`` query parameter filters
  ``methods`` to those that apply to that column type. Match is case-
  insensitive (``?dtype=varchar`` works the same as ``VARCHAR``).
  Unknown dtypes yield ``422`` with
  ``{"code": "INVALID_DTYPE", "message": ...}``.

The endpoint is read-only and does **not** require a loaded schema on
the workspace — the picker is browsable before connect / upload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from dbsprout.schema.models import ColumnType
from dbsprout.spec.catalog import iter_methods, providers

generators_router = APIRouter()


def _serialize_dtypes(dtypes: frozenset[ColumnType]) -> list[str]:
    """Return ``ColumnType.name`` values, sorted for deterministic output."""
    return sorted(d.name for d in dtypes)


def _resolve_dtype(value: str | None) -> ColumnType | None:
    """Map a ``?dtype=`` query value to a :class:`ColumnType` or raise 422."""
    if value is None:
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    try:
        return ColumnType[normalized]
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "INVALID_DTYPE",
                "message": (
                    f"Unknown dtype {value!r}. Expected one of: "
                    + ", ".join(sorted(c.name for c in ColumnType))
                ),
            },
        ) from exc


@generators_router.get("/api/generators", response_model=None)
async def get_generators(
    dtype: str | None = Query(default=None, description="Filter by ColumnType name"),
) -> dict[str, Any]:
    """Return the generator catalogue, optionally filtered by column dtype.

    See module docstring for the response shape and filter semantics.
    """
    requested = _resolve_dtype(dtype)
    methods: list[dict[str, Any]] = []
    for entry in iter_methods():
        if requested is not None and requested not in entry.dtypes:
            continue
        methods.append(
            {
                "provider": entry.provider,
                "method": entry.method,
                "description": entry.description,
                # S-146 — single illustrative value the Studio picker
                # surfaces under each method button.
                "example": entry.example,
                "dtypes": _serialize_dtypes(entry.dtypes),
                "params": sorted(entry.params),
            }
        )
    return {"providers": providers(), "methods": methods}


__all__ = ["generators_router"]
