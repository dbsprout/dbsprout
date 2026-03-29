"""Vectorized NumPy generation for numeric-heavy column types.

Provides a fast path for generating large batches of integer, float,
boolean, datetime, date, and UUID values using NumPy's modern
``Generator`` API for reproducible, high-throughput generation.
"""

from __future__ import annotations

import uuid as _uuid_mod
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

# Set of generator names that can be vectorized
VECTORIZABLE_GENERATORS = frozenset(
    {
        "random_int",
        "random_float",
        "random_decimal",
        "random_bool",
        "random_datetime",
        "random_date",
        "uuid4",
    }
)


def generate_vectorized(
    generator_name: str,
    num_rows: int,
    *,
    seed: int,
    params: dict[str, Any],
) -> list[Any] | None:
    """Generate values using NumPy vectorization.

    Returns ``None`` if the generator name is not vectorizable,
    signaling the caller to fall back to the scalar path.
    """
    func = _DISPATCH.get(generator_name)
    if func is None:
        return None
    rng = np.random.default_rng(seed)
    return func(rng, num_rows, params)  # type: ignore[no-any-return]


# ── Vectorized generators ────────────────────────────────────────────────


def _vec_int(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[int]:
    lo = params.get("min", 0)
    hi = params.get("max", 10000)
    if lo > hi:
        lo, hi = hi, lo
    arr = rng.integers(lo, hi + 1, size=n)
    return [int(x) for x in arr]


def _vec_float(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[float]:
    lo = params.get("min", 0.0)
    hi = params.get("max", 10000.0)
    arr = rng.uniform(lo, hi, size=n)
    return [round(float(v), 2) for v in arr]


def _vec_decimal(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[float]:
    precision = params.get("precision", 10)
    scale = params.get("scale", 2)
    max_val = max(10 ** (precision - scale) - 1, 1)
    arr = rng.uniform(0, max_val, size=n)
    return [round(float(v), scale) for v in arr]


def _vec_bool(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[bool]:
    true_ratio = params.get("true_ratio", 0.5)
    arr = rng.choice([True, False], size=n, p=[true_ratio, 1 - true_ratio])
    return [bool(x) for x in arr]


def _vec_datetime(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[datetime]:  # noqa: ARG001
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, tzinfo=timezone.utc)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    arr = rng.integers(start_ts, end_ts, size=n)
    return [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in arr]


def _vec_date(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[date]:
    dts = _vec_datetime(rng, n, params)
    return [dt.date() for dt in dts]


def _vec_uuid(rng: np.random.Generator, n: int, params: dict[str, Any]) -> list[str]:  # noqa: ARG001
    raw = rng.bytes(16 * n)
    result: list[str] = []
    for i in range(n):
        b = bytearray(raw[i * 16 : (i + 1) * 16])
        # Set version to 4
        b[6] = (b[6] & 0x0F) | 0x40
        # Set variant to RFC 4122
        b[8] = (b[8] & 0x3F) | 0x80
        result.append(str(_uuid_mod.UUID(bytes=bytes(b))))
    return result


# ── Dispatch table ───────────────────────────────────────────────────────

_DISPATCH: dict[str, Any] = {
    "random_int": _vec_int,
    "random_float": _vec_float,
    "random_decimal": _vec_decimal,
    "random_bool": _vec_bool,
    "random_datetime": _vec_datetime,
    "random_date": _vec_date,
    "uuid4": _vec_uuid,
}
