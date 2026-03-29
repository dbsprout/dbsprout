"""Heuristic column-to-generator mapping.

Maps column names to data generators using regex patterns and type
analysis. This is the zero-LLM path — works offline with no AI deps.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from dbsprout.schema.models import ColumnType
from dbsprout.spec.models import GeneratorMapping
from dbsprout.spec.patterns import PATTERNS

if TYPE_CHECKING:
    from dbsprout.schema.models import ColumnSchema, DatabaseSchema

# ── Type-based fallback generators ───────────────────────────────────────

_TYPE_FALLBACKS: dict[ColumnType, tuple[str, str]] = {
    ColumnType.INTEGER: ("random_int", "builtin"),
    ColumnType.BIGINT: ("random_int", "builtin"),
    ColumnType.SMALLINT: ("random_int", "builtin"),
    ColumnType.FLOAT: ("random_float", "builtin"),
    ColumnType.DECIMAL: ("random_decimal", "builtin"),
    ColumnType.BOOLEAN: ("random_bool", "builtin"),
    ColumnType.VARCHAR: ("random_string", "builtin"),
    ColumnType.TEXT: ("random_text", "builtin"),
    ColumnType.DATE: ("random_date", "builtin"),
    ColumnType.DATETIME: ("random_datetime", "builtin"),
    ColumnType.TIMESTAMP: ("random_datetime", "builtin"),
    ColumnType.TIME: ("random_time", "builtin"),
    ColumnType.UUID: ("uuid4", "builtin"),
    ColumnType.JSON: ("random_json", "builtin"),
    ColumnType.BINARY: ("random_bytes", "builtin"),
    ColumnType.ENUM: ("random_choice", "builtin"),
    ColumnType.ARRAY: ("random_list", "builtin"),
    ColumnType.UNKNOWN: ("random_string", "builtin"),
}

_CAMEL_SPLIT_RE = re.compile(r"([a-z])([A-Z])")


# ── Public API ───────────────────────────────────────────────────────────


def map_columns(
    schema: DatabaseSchema,
) -> dict[str, dict[str, GeneratorMapping]]:
    """Map all columns in a schema to generator configurations.

    Returns a nested dict: ``{table_name: {column_name: GeneratorMapping}}``.
    """
    result: dict[str, dict[str, GeneratorMapping]] = {}
    for table in schema.tables:
        table_mappings: dict[str, GeneratorMapping] = {}
        for col in table.columns:
            table_mappings[col.name] = _match_column(col)
        result[table.name] = table_mappings
    return result


# ── Column matching pipeline ─────────────────────────────────────────────


def _match_column(col: ColumnSchema) -> GeneratorMapping:
    """Match a single column to a generator via pattern or type fallback."""
    params = _build_params(col)

    # 1. Exact regex on full column name
    full_name = col.name.lower()
    for pattern in PATTERNS:
        if re.fullmatch(pattern.regex, full_name):
            return GeneratorMapping(
                generator_name=pattern.generator_name,
                provider=pattern.provider,
                confidence=pattern.base_confidence,
                params={**pattern.params, **params},
            )

    # 2. Regex on tokenized/normalized name
    normalized = _normalize_name(col.name)
    if normalized != full_name:
        for pattern in PATTERNS:
            if re.fullmatch(pattern.regex, normalized):
                return GeneratorMapping(
                    generator_name=pattern.generator_name,
                    provider=pattern.provider,
                    confidence=pattern.base_confidence * 0.9,
                    params={**pattern.params, **params},
                )

    # 3. Type-based fallback
    return _type_fallback(col, params)


def _type_fallback(col: ColumnSchema, params: dict[str, Any]) -> GeneratorMapping:
    """Fall back to type-appropriate random generator."""
    gen_name, provider = _TYPE_FALLBACKS.get(col.data_type, ("random_string", "builtin"))

    # Special case: ENUM with values → random_choice
    if col.data_type is ColumnType.ENUM and col.enum_values:
        params = {**params, "enum_values": col.enum_values}
        gen_name = "random_choice"

    return GeneratorMapping(
        generator_name=gen_name,
        provider=provider,
        confidence=0.5,
        params=params,
    )


# ── Token extraction ─────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Normalize column name: split camelCase/PascalCase, join with underscores."""
    # camelCase → camel_case
    split = _CAMEL_SPLIT_RE.sub(r"\1_\2", name)
    # Replace hyphens with underscores, lowercase
    return split.replace("-", "_").lower()


def _build_params(col: ColumnSchema) -> dict[str, Any]:
    """Extract generator params from column metadata."""
    params: dict[str, Any] = {}
    if col.max_length is not None:
        params["max_length"] = col.max_length
    if col.precision is not None:
        params["precision"] = col.precision
    if col.scale is not None:
        params["scale"] = col.scale
    if col.enum_values:
        params["enum_values"] = col.enum_values
    return params
