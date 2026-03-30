"""Geo coherence — city/state/zip lookup for address data.

Ensures generated address columns are geographically coherent:
a city always matches its real state and zip code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dbsprout.schema.models import TableSchema

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parent.parent / "_vendor" / "geo" / "us_cities.json"

_CITY_NAMES = frozenset({"city", "town", "municipality"})
_STATE_NAMES = frozenset({"state", "state_code", "province", "region"})
_ZIP_NAMES = frozenset({"zip", "zip_code", "postal_code", "zipcode"})


class GeoLookup:
    """Lookup table for coherent US city/state/zip generation."""

    def __init__(self, data_path: Path | None = None) -> None:
        self._data_path = data_path or _DATA_PATH
        self._data: list[list[str]] | None = None

    def _load(self) -> list[list[str]]:
        if self._data is not None:
            return self._data
        with self._data_path.open(encoding="utf-8") as f:
            self._data = json.load(f)
        return self._data

    def sample(self, num_rows: int, seed: int) -> list[dict[str, str]]:
        """Sample coherent city/state/zip tuples.

        Returns list of ``{"city": ..., "state": ..., "zip": ...}`` dicts.
        """
        data = self._load()
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(data), size=num_rows)
        return [
            {"city": data[idx][0], "state": data[idx][1], "zip": data[idx][2]} for idx in indices
        ]


def detect_geo_columns(table: TableSchema) -> dict[str, str]:
    """Detect geo-related columns by name matching.

    Returns a mapping ``{"city": col_name, "state": col_name, "zip": col_name}``
    for detected columns. Only returns non-empty if ≥2 geo columns found.
    """
    result: dict[str, str] = {}
    for col in table.columns:
        lower = col.name.lower()
        if lower in _CITY_NAMES:
            result["city"] = col.name
        elif lower in _STATE_NAMES:
            result["state"] = col.name
        elif lower in _ZIP_NAMES:
            result["zip"] = col.name

    min_geo_cols = 2
    if len(result) < min_geo_cols:
        return {}
    return result


def apply_geo_coherence(
    table: TableSchema,
    rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Patch geo columns in rows with coherent city/state/zip values.

    Returns the same list (mutated in-place). If no geo columns detected,
    returns rows unchanged.
    """
    geo_cols = detect_geo_columns(table)
    if not geo_cols:
        return rows

    lookup = GeoLookup()
    samples = lookup.sample(len(rows), seed)

    for i, row in enumerate(rows):
        geo = samples[i]
        if "city" in geo_cols:
            row[geo_cols["city"]] = geo["city"]
        if "state" in geo_cols:
            row[geo_cols["state"]] = geo["state"]
        if "zip" in geo_cols:
            row[geo_cols["zip"]] = geo["zip"]

    return rows
