"""Data-preview view-model builder (S-084).

Turns *real* generated rows into a template-ready view-model for the HTML
report's "Data Preview" section. The generated rows come from
``GenerateResult.tables_data`` (``dict[str, list[dict[str, Any]]]`` —
``dbsprout/generate/orchestrator.py``); the shared state DB stores telemetry
only, so preview data is supplied at report time.

The builder is pure (no I/O) and returns plain ``dict``/``list`` structures
so the Jinja2 template stays dumb and ``autoescape`` keeps every cell
XSS-safe. Cells are classified into four kinds so the template can style
them: ``plain``, ``null``, ``trunc`` (long text), ``fk`` (anchor to parent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema, TableSchema

#: Maximum sample rows shown per table.
PREVIEW_ROW_LIMIT = 10

#: Text values longer than this are truncated (with a full-value tooltip).
TRUNCATE_LIMIT = 50


def _anchor(table_name: str) -> str:
    """Stable, same-page anchor slug for a table preview section."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in table_name.lower())
    return f"table-{slug.strip('-') or 'unnamed'}"


def _column_view(table: TableSchema) -> list[dict[str, Any]]:
    fk_cols = {col for fk in table.foreign_keys for col in fk.columns}
    out: list[dict[str, Any]] = []
    for col in table.columns:
        badges: list[str] = []
        if col.primary_key:
            badges.append("PK")
        if col.name in fk_cols:
            badges.append("FK")
        if col.unique:
            badges.append("UNIQUE")
        if not col.nullable:
            badges.append("NOT NULL")
        out.append(
            {
                "name": col.name,
                "type": col.data_type.value,
                "badges": badges,
            }
        )
    return out


def _fk_target(table: TableSchema) -> dict[str, tuple[str, str]]:
    """Map FK column name → (ref_table, ref_anchor)."""
    mapping: dict[str, tuple[str, str]] = {}
    for fk in table.foreign_keys:
        for col in fk.columns:
            mapping[col] = (fk.ref_table, _anchor(fk.ref_table))
    return mapping


def _cell(value: Any, fk: tuple[str, str] | None) -> dict[str, Any]:
    if value is None:
        return {"kind": "null"}
    text = str(value)
    if fk is not None:
        ref_table, ref_anchor = fk
        return {
            "kind": "fk",
            "display": text,
            "ref_table": ref_table,
            "ref_anchor": ref_anchor,
        }
    if len(text) > TRUNCATE_LIMIT:
        return {
            "kind": "trunc",
            "display": text[:TRUNCATE_LIMIT] + "…",
            "full": text,
        }
    return {"kind": "plain", "display": text}


def build_table_previews(
    schema: DatabaseSchema,
    tables_data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the per-table preview view-model.

    For each table in ``schema`` (preserving schema order) returns a dict
    with ``name``, ``anchor``, ``columns`` (name/type/constraint badges),
    ``rows`` (≤ :data:`PREVIEW_ROW_LIMIT` rows of classified cells) and the
    true ``total_rows`` count. Tables absent from ``tables_data`` render
    with zero rows rather than failing.
    """
    previews: list[dict[str, Any]] = []
    for table in schema.tables:
        columns = _column_view(table)
        fk_map = _fk_target(table)
        all_rows = tables_data.get(table.name, [])
        sample = all_rows[:PREVIEW_ROW_LIMIT]
        rows: list[list[dict[str, Any]]] = [
            [_cell(row.get(col.name), fk_map.get(col.name)) for col in table.columns]
            for row in sample
        ]
        previews.append(
            {
                "name": table.name,
                "anchor": _anchor(table.name),
                "columns": columns,
                "rows": rows,
                "total_rows": len(all_rows),
            }
        )
    return previews
