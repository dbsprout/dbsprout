"""JSON Schema to GBNF grammar converter.

Converts a Pydantic model's JSON Schema into a GBNF grammar string
for llama-cpp-python constrained generation. Guarantees LLM output
is always structurally valid JSON matching the schema.
"""

from __future__ import annotations

from typing import Any

from dbsprout.spec.models import DataSpec

# ── Primitive GBNF rules (shared across all grammars) ────────────────

_PRIMITIVES = r"""
ws ::= [ \t\n]*
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
integer ::= "-"? ([0-9] | [1-9] [0-9]*)
boolean ::= "true" | "false"
null ::= "null"
value ::= string | number | boolean | null | object | array
object ::= "{" ws "}" | "{" ws string ws ":" ws value (ws "," ws string ws ":" ws value)* ws "}"
array ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
"""


def generate_dataspec_grammar() -> str:
    """Generate a GBNF grammar from the DataSpec JSON Schema.

    Convenience function that calls ``json_schema_to_gbnf`` with
    the full DataSpec schema including ``$defs``.
    """
    schema = DataSpec.model_json_schema()
    return json_schema_to_gbnf(schema, "root")


def json_schema_to_gbnf(
    schema: dict[str, Any],
    rule_name: str,
    _defs: dict[str, Any] | None = None,
) -> str:
    """Convert a JSON Schema dict to a GBNF grammar string.

    Parameters
    ----------
    schema:
        A JSON Schema dictionary (from ``model_json_schema()``).
    rule_name:
        Name for the top-level grammar rule.
    _defs:
        Internal: ``$defs`` dict for resolving ``$ref`` references.

    Returns
    -------
    str
        Complete GBNF grammar string with primitive rules included.
    """
    if _defs is None:
        _defs = schema.get("$defs", {})

    rules: list[str] = []
    _emit_rule(schema, rule_name, _defs, rules, set())

    return "\n".join(rules) + "\n" + _PRIMITIVES


def _emit_rule(
    schema: dict[str, Any],
    name: str,
    defs: dict[str, Any],
    rules: list[str],
    visited: set[str],
) -> None:
    """Recursively emit GBNF rules for a JSON Schema node."""
    if name in visited:
        return
    visited.add(name)

    # Handle $ref
    if "$ref" in schema:
        ref_path = schema["$ref"]
        ref_name = ref_path.rsplit("/", maxsplit=1)[-1]
        rules.append(f"{name} ::= {_safe_name(ref_name)}")
        if ref_name in defs:
            _emit_rule(defs[ref_name], _safe_name(ref_name), defs, rules, visited)
        return

    # Handle anyOf (nullable types, union types)
    if "anyOf" in schema:
        _emit_anyof(schema["anyOf"], name, defs, rules, visited)
        return

    # Handle enum
    if "enum" in schema:
        alternatives = " | ".join(f'"\\"{v}\\""' for v in schema["enum"])
        rules.append(f"{name} ::= {alternatives}")
        return

    schema_type = schema.get("type", "string")

    if schema_type == "object":
        _emit_object(schema, name, defs, rules, visited)
    elif schema_type == "array":
        _emit_array(schema, name, defs, rules, visited)
    else:
        _emit_primitive(schema_type, name, rules)


_PRIMITIVE_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "null": "null",
}


def _emit_primitive(schema_type: str, name: str, rules: list[str]) -> None:
    """Emit a rule for a primitive JSON type."""
    rule = _PRIMITIVE_TYPE_MAP.get(schema_type, "value")
    rules.append(f"{name} ::= {rule}")


def _emit_object(
    schema: dict[str, Any],
    name: str,
    defs: dict[str, Any],
    rules: list[str],
    visited: set[str],
) -> None:
    """Emit rules for a JSON Schema object type."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not props:
        rules.append(f"{name} ::= object")
        return

    # Build property rules
    parts: list[str] = []
    for prop_name, prop_schema in props.items():
        prop_rule = f"{name}_{_safe_name(prop_name)}"
        _emit_rule(prop_schema, prop_rule, defs, rules, visited)

        kv = f'"\\"{prop_name}\\"" ws ":" ws {prop_rule}'
        if prop_name in required:
            parts.append(kv)
        else:
            # Optional properties: include with separator or omit
            parts.append(kv)  # simplified: include all for grammar correctness

    # Join properties with comma separators
    if parts:
        body = ' ws "," ws '.join(parts)
        rules.append(f'{name} ::= "{{" ws {body} ws "}}"')
    else:
        rules.append(f"{name} ::= object")


def _emit_array(
    schema: dict[str, Any],
    name: str,
    defs: dict[str, Any],
    rules: list[str],
    visited: set[str],
) -> None:
    """Emit rules for a JSON Schema array type."""
    items_schema = schema.get("items", {})
    item_rule = f"{name}_item"
    _emit_rule(items_schema, item_rule, defs, rules, visited)

    rules.append(f'{name} ::= "[]" | "[" ws {item_rule} (ws "," ws {item_rule})* ws "]"')


def _emit_anyof(
    variants: list[dict[str, Any]],
    name: str,
    defs: dict[str, Any],
    rules: list[str],
    visited: set[str],
) -> None:
    """Emit rules for anyOf (union/nullable types)."""
    alt_names: list[str] = []
    for i, variant in enumerate(variants):
        alt_name = f"{name}_alt{i}"
        _emit_rule(variant, alt_name, defs, rules, visited)
        alt_names.append(alt_name)

    rules.append(f"{name} ::= {' | '.join(alt_names)}")


def _safe_name(name: str) -> str:
    """Make a string safe for use as a GBNF rule name."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")
