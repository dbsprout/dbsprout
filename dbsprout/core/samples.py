"""Bundled sample schemas for the web Workbench "instant try" flow (P1a).

Demo DDL files live in ``dbsprout/_vendor/samples/`` and ship in the wheel.
``list_samples`` parses each once (cheap) for table counts; ``load_sample``
returns a unified :class:`DatabaseSchema` via the same parser path the upload
endpoint uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING

from dbsprout.schema.parsers import parse_schema_file

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema

_SAMPLES_PKG = "dbsprout._vendor.samples"

_MANIFEST: dict[str, tuple[str, str, str]] = {
    "ecommerce": (
        "ecommerce.sql",
        "E-commerce",
        "Storefront: users, products, orders, line items, inventory.",
    ),
    "saas": (
        "saas.sql",
        "SaaS",
        "Multi-tenant SaaS: accounts, users, subscriptions, usage.",
    ),
}


@dataclass(frozen=True)
class SampleInfo:
    """A bundled sample schema, summarised for the picker."""

    name: str
    title: str
    description: str
    dialect: str
    table_count: int


@cache
def _load(name: str) -> DatabaseSchema:
    try:
        filename = _MANIFEST[name][0]
    except KeyError as exc:
        raise KeyError(name) from exc
    with resources.as_file(resources.files(_SAMPLES_PKG) / filename) as path:
        return parse_schema_file(path)


def load_sample(name: str) -> DatabaseSchema:
    """Return the :class:`DatabaseSchema` for bundled sample *name* (KeyError if unknown)."""
    return _load(name)


def list_samples() -> list[SampleInfo]:
    """List bundled samples with derived table_count + dialect."""
    out: list[SampleInfo] = []
    for name, (_, title, description) in _MANIFEST.items():
        schema = _load(name)
        out.append(
            SampleInfo(
                name=name,
                title=title,
                description=description,
                dialect=schema.dialect or "unknown",
                table_count=len(schema.tables),
            )
        )
    return out
