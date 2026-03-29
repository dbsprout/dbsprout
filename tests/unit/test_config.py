"""Tests for dbsprout.config — TOML config system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from dbsprout.config.loader import load_config
from dbsprout.config.models import (
    DBSproutConfig,
    GenerationConfig,
    SchemaConfig,
    TableOverride,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── TOML template matching S-009 init output ─────────────────────────────

VALID_TOML = """\
[schema]
dialect = "sqlite"
source = "sqlite:///myapp.db"
snapshot = ".dbsprout/snapshots/abc123.json"

[generation]
default_rows = 100
seed = 42
output_format = "sql"
output_dir = "./seeds"
"""

TOML_WITH_OVERRIDES = """\
[schema]
dialect = "postgresql"
source = "postgresql://user@localhost/db"

[generation]
default_rows = 200
seed = 99

[tables.users]
rows = 50

[tables.logs]
exclude = true
"""


# ── Model defaults ───────────────────────────────────────────────────────


class TestDefaultConfig:
    def test_all_defaults_sensible(self) -> None:
        cfg = DBSproutConfig()
        assert cfg.schema_.dialect is None
        assert cfg.schema_.source is None
        assert cfg.generation.default_rows == 100
        assert cfg.generation.seed == 42
        assert cfg.generation.engine == "heuristic"
        assert cfg.generation.output_format == "sql"
        assert cfg.generation.output_dir == "./seeds"
        assert cfg.tables == {}

    def test_frozen(self) -> None:
        cfg = DBSproutConfig()
        with pytest.raises(ValidationError):
            cfg.tables = {}  # type: ignore[misc]


class TestSchemaConfig:
    def test_fields(self) -> None:
        sc = SchemaConfig(dialect="sqlite", source="sqlite:///x.db", snapshot="snap.json")
        assert sc.dialect == "sqlite"
        assert sc.source == "sqlite:///x.db"
        assert sc.snapshot == "snap.json"

    def test_defaults(self) -> None:
        sc = SchemaConfig()
        assert sc.dialect is None
        assert sc.source is None
        assert sc.snapshot is None


class TestGenerationConfig:
    def test_fields(self) -> None:
        gc = GenerationConfig(
            default_rows=500, seed=7, engine="spec", output_format="csv", output_dir="/out"
        )
        assert gc.default_rows == 500
        assert gc.seed == 7
        assert gc.engine == "spec"

    def test_rows_validation_zero(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(default_rows=0)

    def test_rows_validation_negative(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(default_rows=-1)


class TestTableOverride:
    def test_defaults(self) -> None:
        to = TableOverride()
        assert to.rows is None
        assert to.exclude is False

    def test_rows_override(self) -> None:
        to = TableOverride(rows=50)
        assert to.rows == 50

    def test_rows_validation_negative(self) -> None:
        with pytest.raises(ValidationError):
            TableOverride(rows=-1)

    def test_rows_validation_zero(self) -> None:
        with pytest.raises(ValidationError):
            TableOverride(rows=0)


class TestExtraFieldsRejected:
    def test_unknown_top_level(self) -> None:
        with pytest.raises(ValidationError):
            DBSproutConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_unknown_generation_field(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfig(defualt_rows=100)  # type: ignore[call-arg]


# ── Loader ───────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_from_toml_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text(VALID_TOML)
        cfg = load_config(p)
        assert cfg.schema_.dialect == "sqlite"
        assert cfg.schema_.source == "sqlite:///myapp.db"
        assert cfg.generation.default_rows == 100
        assert cfg.generation.seed == 42

    def test_defaults_when_missing(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.generation.default_rows == 100
        assert cfg.tables == {}

    def test_none_path(self) -> None:
        cfg = load_config(None)
        assert cfg.generation.default_rows == 100

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.toml"
        p.write_text("")
        cfg = load_config(p)
        assert cfg.generation.default_rows == 100

    def test_with_table_overrides(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text(TOML_WITH_OVERRIDES)
        cfg = load_config(p)
        assert "users" in cfg.tables
        assert cfg.tables["users"].rows == 50
        assert "logs" in cfg.tables
        assert cfg.tables["logs"].exclude is True

    def test_invalid_type(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.toml"
        p.write_text('[generation]\ndefault_rows = "fifty"\n')
        with pytest.raises(ValidationError):
            load_config(p)

    def test_init_toml_round_trips(self, tmp_path: Path) -> None:
        """The exact TOML template from S-009 init loads correctly."""
        p = tmp_path / "dbsprout.toml"
        p.write_text(VALID_TOML)
        cfg = load_config(p)
        assert cfg.schema_.snapshot == ".dbsprout/snapshots/abc123.json"
        assert cfg.generation.output_format == "sql"
        assert cfg.generation.output_dir == "./seeds"
