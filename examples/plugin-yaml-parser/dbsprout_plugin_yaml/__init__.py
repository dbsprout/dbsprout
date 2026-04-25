"""Minimal example parser plugin.

Registers a ``yaml`` schema parser as a demonstration of the DBSprout
entry-point plugin API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema


class YamlParser:
    suffixes: tuple[str, ...] = (".yaml", ".yml")

    def can_parse(self, text: str) -> bool:  # noqa: ARG002
        return True

    def parse(
        self,
        text: str,
        *,
        source_file: str | None = None,  # noqa: ARG002
    ) -> DatabaseSchema:
        import yaml  # noqa: PLC0415

        from dbsprout.schema.models import DatabaseSchema  # noqa: PLC0415

        data = yaml.safe_load(text)  # noqa: F841
        return DatabaseSchema(tables=[], dialect="postgresql")
