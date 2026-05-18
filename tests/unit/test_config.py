"""Tests for dbsprout.config — TOML config system."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dbsprout.config.loader import load_config
from dbsprout.config.models import (
    DBSproutConfig,
    GenerationConfig,
    LLMConfig,
    PrivacyConfig,
    ReportConfig,
    SchemaConfig,
    TableOverride,
)
from dbsprout.train.config import TrainConfig

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

    def test_train_section_defaults(self) -> None:
        cfg = DBSproutConfig()
        assert isinstance(cfg.train, TrainConfig)
        assert cfg.train == TrainConfig()
        assert cfg.train.epochs == 3
        assert cfg.train.completion_only_loss is True


class TestTrainSection:
    def test_train_overrides_round_trip(self, tmp_path: Path) -> None:
        toml = """\
[train]
epochs = 5
learning_rate = 0.001
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.0
batch_size = 4
base_model = "some/model"
"""
        path = tmp_path / "dbsprout.toml"
        path.write_text(toml)
        cfg = load_config(path)
        assert cfg.train.epochs == 5
        assert cfg.train.learning_rate == pytest.approx(0.001)
        assert cfg.train.lora_rank == 8
        assert cfg.train.base_model == "some/model"

    def test_train_section_rejects_unknown_key(self, tmp_path: Path) -> None:
        path = tmp_path / "dbsprout.toml"
        path.write_text("[train]\nbogus = 1\n")
        with pytest.raises(ValidationError):
            load_config(path)


class TestReportConfig:
    def test_default_output_path(self) -> None:
        cfg = DBSproutConfig()
        assert isinstance(cfg.report, ReportConfig)
        assert cfg.report == ReportConfig()
        assert cfg.report.output == "./seeds/report.html"

    def test_report_section_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "dbsprout.toml"
        path.write_text('[report]\noutput = "custom/r.html"\n')
        cfg = load_config(path)
        assert cfg.report.output == "custom/r.html"

    def test_report_section_rejects_unknown_key(self, tmp_path: Path) -> None:
        path = tmp_path / "dbsprout.toml"
        path.write_text("[report]\nbogus = 1\n")
        with pytest.raises(ValidationError):
            load_config(path)

    def test_report_config_frozen(self) -> None:
        rc = ReportConfig()
        with pytest.raises(ValidationError):
            rc.output = "x.html"  # type: ignore[misc]


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

    def test_malformed_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.toml"
        p.write_text("[generation\nunterminated")
        with pytest.raises(ValueError, match="Failed to parse"):
            load_config(p)

    def test_invalid_engine(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text('[generation]\nengine = "bogus"\n')
        with pytest.raises(ValidationError):
            load_config(p)

    def test_invalid_output_format(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text('[generation]\noutput_format = "xlsx"\n')
        with pytest.raises(ValidationError):
            load_config(p)

    def test_from_toml_classmethod(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text(VALID_TOML)
        cfg = DBSproutConfig.from_toml(p)
        assert cfg.schema_.dialect == "sqlite"
        assert cfg.generation.default_rows == 100

    def test_from_toml_defaults(self) -> None:
        cfg = DBSproutConfig.from_toml(None)
        assert cfg.generation.default_rows == 100

    def test_init_toml_round_trips(self, tmp_path: Path) -> None:
        """The exact TOML template from S-009 init loads correctly."""
        p = tmp_path / "dbsprout.toml"
        p.write_text(VALID_TOML)
        cfg = load_config(p)
        assert cfg.schema_.snapshot == ".dbsprout/snapshots/abc123.json"
        assert cfg.generation.output_format == "sql"
        assert cfg.generation.output_dir == "./seeds"


class TestPrivacyConfig:
    def test_default_tier_is_local(self) -> None:
        pc = PrivacyConfig()
        assert pc.tier == "local"

    def test_tier_redacted(self) -> None:
        pc = PrivacyConfig(tier="redacted")
        assert pc.tier == "redacted"

    def test_tier_cloud(self) -> None:
        pc = PrivacyConfig(tier="cloud")
        assert pc.tier == "cloud"

    def test_rejects_invalid_tier(self) -> None:
        with pytest.raises(ValidationError):
            PrivacyConfig(tier="none")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        pc = PrivacyConfig()
        with pytest.raises(ValidationError):
            pc.tier = "cloud"  # type: ignore[misc]

    def test_in_dbsprout_config(self) -> None:
        cfg = DBSproutConfig()
        assert cfg.privacy.tier == "local"

    def test_from_toml(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text('[privacy]\ntier = "redacted"\n')
        cfg = load_config(p)
        assert cfg.privacy.tier == "redacted"


# ── [llm] section (S-067c) ───────────────────────────────────────────────


class TestLLMConfig:
    def test_default_lora_path_is_none(self) -> None:
        lc = LLMConfig()
        assert lc.lora_path is None

    def test_lora_path_accepts_path(self, tmp_path: Path) -> None:
        p = tmp_path / "a.gguf"
        lc = LLMConfig(lora_path=p)
        assert lc.lora_path == p

    def test_frozen(self) -> None:
        lc = LLMConfig()
        with pytest.raises(ValidationError):
            lc.lora_path = Path("x")  # type: ignore[misc]

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(provider="ollama")  # type: ignore[call-arg]

    def test_in_dbsprout_config_default(self) -> None:
        cfg = DBSproutConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert cfg.llm.lora_path is None


class TestLLMConfigLoader:
    def test_llm_section_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text('[llm]\nlora_path = "./adapters/x.gguf"\n')
        cfg = load_config(p)
        assert cfg.llm.lora_path == Path("./adapters/x.gguf")

    def test_no_llm_section_backward_compatible(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text(VALID_TOML)
        cfg = load_config(p)
        assert cfg.llm.lora_path is None
        assert cfg.llm == LLMConfig()

    def test_llm_unknown_key_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text("[llm]\nbogus = 1\n")
        with pytest.raises(ValidationError):
            load_config(p)

    def test_llm_bad_type_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "dbsprout.toml"
        p.write_text("[llm]\nlora_path = 123\n")
        with pytest.raises(ValidationError):
            load_config(p)
