"""Tests for dbsprout.spec.grammar — JSON Schema to GBNF grammar conversion."""

from __future__ import annotations

from dbsprout.spec.grammar import generate_dataspec_grammar, json_schema_to_gbnf


class TestGenerateDataspecGrammar:
    def test_produces_nonempty_string(self) -> None:
        """Grammar output must be a non-empty string."""
        grammar = generate_dataspec_grammar()
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_contains_root_rule(self) -> None:
        """Grammar must have a root rule."""
        grammar = generate_dataspec_grammar()
        assert "root ::=" in grammar


class TestGBNFFromSchema:
    def test_string_type(self) -> None:
        """JSON Schema string type produces a string rule."""
        schema = {"type": "string"}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "test" in gbnf
        # Should contain a string-matching pattern
        assert '"' in gbnf

    def test_number_type(self) -> None:
        """JSON Schema number type produces a number rule."""
        schema = {"type": "number"}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "test" in gbnf

    def test_integer_type(self) -> None:
        """JSON Schema integer type produces an integer rule."""
        schema = {"type": "integer"}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "test" in gbnf

    def test_boolean_type(self) -> None:
        """JSON Schema boolean type produces true/false rule."""
        schema = {"type": "boolean"}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "true" in gbnf or "false" in gbnf

    def test_null_type(self) -> None:
        """JSON Schema null type produces null literal."""
        schema = {"type": "null"}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "null" in gbnf

    def test_object_type(self) -> None:
        """JSON Schema object with properties produces object rule."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "name" in gbnf
        assert "age" in gbnf

    def test_array_type(self) -> None:
        """JSON Schema array type produces array rule."""
        schema = {"type": "array", "items": {"type": "string"}}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "test" in gbnf

    def test_enum_type(self) -> None:
        """JSON Schema enum produces alternatives."""
        schema = {"enum": ["active", "inactive", "pending"]}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "active" in gbnf
        assert "inactive" in gbnf

    def test_nullable_anyof(self) -> None:
        """anyOf with null type produces nullable rule."""
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "null" in gbnf

    def test_ref_resolution(self) -> None:
        """$ref references are resolved from $defs."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
        }
        gbnf = json_schema_to_gbnf(schema, "root")
        assert "Item" in gbnf or "item" in gbnf

    def test_additional_properties(self) -> None:
        """additionalProperties (dict[str, T]) constrains value types."""
        schema = {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "test_val" in gbnf
        assert "integer" in gbnf

    def test_enum_with_quotes(self) -> None:
        """Enum values with quotes must be escaped."""
        schema = {"enum": ['it"s', "normal"]}
        gbnf = json_schema_to_gbnf(schema, "test")
        assert "normal" in gbnf
        # Should not produce broken GBNF
        assert "test ::=" in gbnf
