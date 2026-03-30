"""Spec-driven generation engine — interprets DataSpec instructions.

Dispatches GeneratorConfig to Mimesis/NumPy providers with support
for distributions, enums, and nullable rates. Falls back to random
values for unknown providers.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from mimesis import Address, Datetime, Finance, Internet, Payment, Person, Text
from mimesis.enums import Locale

if TYPE_CHECKING:
    from numpy.random import Generator

    from dbsprout.schema.models import TableSchema
    from dbsprout.spec.models import GeneratorConfig, TableSpec

from dbsprout.generate.deterministic import column_seed

logger = logging.getLogger(__name__)


class SpecDrivenEngine:
    """Generate rows from DataSpec GeneratorConfig instructions."""

    def __init__(self, locale: str = "en", seed: int = 42) -> None:
        self._seed = seed
        loc = Locale(locale) if locale != "en" else Locale.EN
        self._person = Person(loc)
        self._address = Address(loc)
        self._finance = Finance(loc)
        self._internet = Internet()
        self._text = Text(loc)
        self._dt = Datetime(loc)
        self._payment = Payment()

    def generate_table(
        self,
        table: TableSchema,
        table_spec: TableSpec,
        num_rows: int,
    ) -> list[dict[str, Any]]:
        """Generate rows from a TableSpec.

        FK columns and autoincrement PKs are set to None.
        """
        fk_cols = {col for fk in table.foreign_keys for col in fk.columns}
        auto_pk_cols = {col.name for col in table.columns if col.primary_key and col.autoincrement}

        col_data: dict[str, list[Any]] = {}
        for col in table.columns:
            if col.name in fk_cols or col.name in auto_pk_cols:
                col_data[col.name] = [None] * num_rows
                continue

            config = table_spec.columns.get(col.name)
            col_s = column_seed(self._seed, table.name, col.name)
            col_data[col.name] = self._generate_from_config(
                config,
                num_rows,
                col_s,
            )

        # Log warnings for deferred features
        if table_spec.derived:
            logger.warning(
                "Table '%s' has %d derived columns — skipped (Sprint 5)",
                table.name,
                len(table_spec.derived),
            )
        if table_spec.correlations:
            logger.warning(
                "Table '%s' has %d correlation rules — skipped (Sprint 5)",
                table.name,
                len(table_spec.correlations),
            )

        return [{col_name: col_data[col_name][i] for col_name in col_data} for i in range(num_rows)]

    def _generate_from_config(
        self,
        config: GeneratorConfig | None,
        num_rows: int,
        col_seed: int,
    ) -> list[Any]:
        """Dispatch a GeneratorConfig to the appropriate generator."""
        rng = np.random.default_rng(col_seed)

        if config is None:
            return [None] * num_rows

        # Enum override — ignores provider
        if config.enum_values:
            values = _apply_enum(config.enum_values, num_rows, rng)
            return _apply_nullable(values, config.nullable_rate, rng)

        provider = config.provider

        values = self._dispatch_provider(provider, config, num_rows, rng)
        return _apply_nullable(values, config.nullable_rate, rng)

    def _dispatch_provider(
        self,
        provider: str,
        config: GeneratorConfig,
        num_rows: int,
        rng: Generator,
    ) -> list[Any]:
        """Route a provider string to the appropriate generator."""
        if provider == "builtin.uuid4":
            return [
                str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8))))
                for _ in range(num_rows)
            ]
        if provider in ("builtin.autoincrement", "builtin.default"):
            return [None] * num_rows
        if provider.startswith("numpy."):
            return _dispatch_numpy(config, num_rows, rng)
        if provider.startswith("mimesis."):
            return self._dispatch_mimesis(provider, num_rows)
        logger.warning("Unknown provider '%s', falling back to random string", provider)
        return [f"val_{i}" for i in range(num_rows)]

    def _dispatch_mimesis(self, provider: str, num_rows: int) -> list[Any]:
        """Dispatch a mimesis.Class.method provider string."""
        parts = provider.split(".")
        min_parts = 3
        if len(parts) < min_parts:
            return [f"val_{i}" for i in range(num_rows)]

        class_name = parts[1]
        method_name = parts[2]

        instances: dict[str, Any] = {
            "Person": self._person,
            "Address": self._address,
            "Finance": self._finance,
            "Internet": self._internet,
            "Text": self._text,
            "Datetime": self._dt,
            "Payment": self._payment,
        }

        instance = instances.get(class_name)
        if instance is None:
            return [f"val_{i}" for i in range(num_rows)]

        method = getattr(instance, method_name, None)
        if method is None or not callable(method):
            return [f"val_{i}" for i in range(num_rows)]

        return [method() for _ in range(num_rows)]


def _dispatch_numpy(
    config: GeneratorConfig,
    num_rows: int,
    rng: Generator,
) -> list[Any]:
    """Generate values using NumPy with distribution support."""
    dist = config.distribution or "uniform"
    min_val = config.min_value if config.min_value is not None else 0.0
    max_val = config.max_value if config.max_value is not None else 1000.0

    if dist == "normal":
        params = config.distribution_params
        mean = params.get("mean", (min_val + max_val) / 2)
        std = params.get("std", (max_val - min_val) / 6)
        values = rng.normal(mean, std, size=num_rows)
        values = np.clip(values, min_val, max_val)
    else:
        values = rng.uniform(min_val, max_val, size=num_rows)

    return [round(float(v), 2) for v in values]


def _apply_nullable(
    values: list[Any],
    rate: float,
    rng: Generator,
) -> list[Any]:
    """Apply nullable rate: set fraction of values to None."""
    if rate <= 0.0:
        return values
    mask = rng.random(len(values)) < rate
    return [None if mask[i] else v for i, v in enumerate(values)]


def _apply_enum(
    enum_values: list[str],
    num_rows: int,
    rng: Generator,
) -> list[Any]:
    """Generate values by sampling from enum list."""
    indices = rng.integers(0, len(enum_values), size=num_rows)
    return [enum_values[idx] for idx in indices]
