"""S-067b: tests for the --lora flag wiring (CLI → command → orchestrator → provider)."""

from __future__ import annotations

import logging
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from dbsprout.cli.app import app
from dbsprout.cli.commands import generate as gen_mod
from dbsprout.cli.commands.generate import (
    _validate_lora_adapter_path,
    generate_command,
)
from dbsprout.config.models import DBSproutConfig
from dbsprout.errors import ModelError
from dbsprout.generate.orchestrator import GenerateResult, orchestrate
from dbsprout.schema.models import (
    ColumnSchema,
    ColumnType,
    DatabaseSchema,
    TableSchema,
)
from dbsprout.spec.analyzer import heuristic_fallback
from dbsprout.train.loader import ModelLoader

runner = CliRunner()

_SYNTHETIC_LORA = "/tmp/adapter.gguf"  # noqa: S108 — never read; mock boundary

# Rich renders --help into an ANSI box whose width depends on the terminal.
# CI has no TTY, so Rich falls back to a narrow width that wraps option names
# (splitting "--lora" across lines) and injects ANSI styling mid-token. Match
# the repo convention (per-module _strip_ansi on result.output) and force a
# wide, color-free render so option names stay intact regardless of env.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(
                        name="id",
                        data_type=ColumnType.INTEGER,
                        nullable=False,
                        primary_key=True,
                        autoincrement=True,
                    ),
                    ColumnSchema(
                        name="email",
                        data_type=ColumnType.VARCHAR,
                        nullable=False,
                    ),
                ],
                primary_key=["id"],
            ),
        ],
    )


class TestOrchestratorLoraWiring:
    def test_spec_engine_with_lora_uses_embedded_provider_via_analyzer(self) -> None:
        """engine='spec' + lora_path builds SpecAnalyzer(EmbeddedProvider(lora_path=...))."""
        schema = _schema()
        config = DBSproutConfig()
        lora = Path(_SYNTHETIC_LORA)

        fake_spec = heuristic_fallback(schema)

        with (
            patch("dbsprout.spec.providers.embedded.EmbeddedProvider") as m_provider,
            patch("dbsprout.spec.analyzer.SpecAnalyzer") as m_analyzer,
        ):
            m_analyzer.return_value.analyze.return_value = fake_spec
            result = orchestrate(
                schema, config, seed=42, default_rows=3, engine="spec", lora_path=lora
            )

        m_provider.assert_called_once_with(lora_path=lora)
        m_analyzer.assert_called_once_with(m_provider.return_value)
        m_analyzer.return_value.analyze.assert_called_once_with(schema)
        assert len(result.tables_data["users"]) == 3

    def test_spec_engine_without_lora_uses_heuristic_fallback(self) -> None:
        """No lora_path → heuristic_fallback path, EmbeddedProvider never constructed."""
        schema = _schema()
        config = DBSproutConfig()

        with patch("dbsprout.spec.providers.embedded.EmbeddedProvider") as m_provider:
            result = orchestrate(schema, config, seed=42, default_rows=3, engine="spec")

        m_provider.assert_not_called()
        assert len(result.tables_data["users"]) == 3

    def test_heuristic_engine_with_lora_ignores_provider(self) -> None:
        """lora_path only affects the spec engine; heuristic path never builds provider."""
        schema = _schema()
        config = DBSproutConfig()

        with patch("dbsprout.spec.providers.embedded.EmbeddedProvider") as m_provider:
            result = orchestrate(
                schema,
                config,
                seed=42,
                default_rows=3,
                engine="heuristic",
                lora_path=Path(_SYNTHETIC_LORA),
            )

        m_provider.assert_not_called()
        assert len(result.tables_data["users"]) == 3


class TestGenerateCommandLoraValidation:
    def test_lora_without_spec_engine_raises_model_error(self, tmp_path: Path) -> None:
        adapter = tmp_path / "a.gguf"
        adapter.write_bytes(b"x")

        with pytest.raises(ModelError) as exc:
            generate_command(engine="heuristic", lora_path=adapter)

        assert "spec" in exc.value.fix.lower()
        assert exc.value.exit_code == 1

    def test_lora_missing_path_raises_model_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.gguf"

        with pytest.raises(ModelError) as exc:
            generate_command(engine="spec", lora_path=missing)

        assert str(missing) in exc.value.why
        assert exc.value.exit_code == 1

    def test_lora_directory_path_raises_model_error(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError):
            generate_command(engine="spec", lora_path=tmp_path)

    def test_lora_valid_forwards_to_orchestrate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid --lora path is threaded into orchestrate(lora_path=...)."""
        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "a.gguf"
        adapter.write_bytes(b"x")
        snap_dir = tmp_path / ".dbsprout"
        snap_dir.mkdir()
        (snap_dir / "schema.json").write_text(_schema().model_dump_json(indent=2), encoding="utf-8")

        captured: dict[str, object] = {}

        def _fake_orchestrate(schema: object, config: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return GenerateResult(
                tables_data={"users": [{"id": 1, "email": "a@b.c"}]},
                insertion_order=["users"],
                total_rows=1,
                total_tables=1,
            )

        with patch.object(gen_mod, "orchestrate", _fake_orchestrate):
            gen_mod.generate_command(
                schema_snapshot=snap_dir / "schema.json",
                engine="spec",
                lora_path=adapter,
                output_dir=tmp_path / "seeds",
            )

        assert captured["lora_path"] == adapter


class TestLoraConfigPrecedence:
    """S-067c: --lora (CLI) overrides [llm].lora_path (config)."""

    @staticmethod
    def _setup(tmp_path: Path, lora: Path | None) -> Path:
        """Write a schema snapshot + dbsprout.toml; return the snapshot path."""
        snap_dir = tmp_path / ".dbsprout"
        snap_dir.mkdir(exist_ok=True)
        snap = snap_dir / "schema.json"
        snap.write_text(_schema().model_dump_json(indent=2), encoding="utf-8")
        cfg = tmp_path / "dbsprout.toml"
        cfg.write_text("" if lora is None else f'[llm]\nlora_path = "{lora}"\n')
        return snap

    @staticmethod
    def _capturing_orchestrate(captured: dict[str, object]):  # type: ignore[no-untyped-def]
        def _fake(schema: object, config: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return GenerateResult(
                tables_data={"users": [{"id": 1, "email": "a@b.c"}]},
                insertion_order=["users"],
                total_rows=1,
                total_tables=1,
            )

        return _fake

    def test_config_only_used_when_flag_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "cfg.gguf"
        adapter.write_bytes(b"x")
        snap = self._setup(tmp_path, adapter)
        captured: dict[str, object] = {}

        with patch.object(gen_mod, "orchestrate", self._capturing_orchestrate(captured)):
            gen_mod.generate_command(
                schema_snapshot=snap,
                config_path=tmp_path / "dbsprout.toml",
                engine="spec",
                lora_path=None,
                output_dir=tmp_path / "seeds",
            )

        assert captured["lora_path"] == adapter

    def test_flag_overrides_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg_adapter = tmp_path / "cfg.gguf"
        cfg_adapter.write_bytes(b"x")
        flag_adapter = tmp_path / "flag.gguf"
        flag_adapter.write_bytes(b"x")
        snap = self._setup(tmp_path, cfg_adapter)
        captured: dict[str, object] = {}

        with patch.object(gen_mod, "orchestrate", self._capturing_orchestrate(captured)):
            gen_mod.generate_command(
                schema_snapshot=snap,
                config_path=tmp_path / "dbsprout.toml",
                engine="spec",
                lora_path=flag_adapter,
                output_dir=tmp_path / "seeds",
            )

        assert captured["lora_path"] == flag_adapter

    def test_neither_passes_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        snap = self._setup(tmp_path, None)
        captured: dict[str, object] = {}

        with patch.object(gen_mod, "orchestrate", self._capturing_orchestrate(captured)):
            gen_mod.generate_command(
                schema_snapshot=snap,
                config_path=tmp_path / "dbsprout.toml",
                engine="heuristic",
                lora_path=None,
                output_dir=tmp_path / "seeds",
            )

        assert captured["lora_path"] is None

    def test_config_invalid_path_raises_model_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        snap = self._setup(tmp_path, tmp_path / "missing.gguf")

        with pytest.raises(ModelError) as exc:
            gen_mod.generate_command(
                schema_snapshot=snap,
                config_path=tmp_path / "dbsprout.toml",
                engine="spec",
                lora_path=None,
                output_dir=tmp_path / "seeds",
            )
        assert exc.value.exit_code == 1

    def test_config_path_with_non_spec_engine_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "cfg.gguf"
        adapter.write_bytes(b"x")
        snap = self._setup(tmp_path, adapter)

        with pytest.raises(ModelError) as exc:
            gen_mod.generate_command(
                schema_snapshot=snap,
                config_path=tmp_path / "dbsprout.toml",
                engine="heuristic",
                lora_path=None,
                output_dir=tmp_path / "seeds",
            )
        assert exc.value.exit_code == 1


class TestValidateLoraAdapterPathHelper:
    def test_missing_path_raises_model_error(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError) as exc:
            _validate_lora_adapter_path(tmp_path / "nope.gguf")
        assert "does not exist" in exc.value.what

    def test_directory_path_raises_model_error(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError) as exc:
            _validate_lora_adapter_path(tmp_path)
        assert "not a file" in exc.value.what

    def test_valid_file_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "a.gguf"
        f.write_bytes(b"x")
        assert _validate_lora_adapter_path(f) is None


class TestLoraCliEndToEnd:
    def test_lora_option_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"], env=_HELP_ENV)
        assert result.exit_code == 0
        assert "--lora" in _strip_ansi(result.output)

    def test_generate_lora_end_to_end_with_mocked_llama(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`dbsprout generate --engine spec --lora <file>` runs end to end.

        The only mock is the real heavy boundary: `llama_cpp.Llama` (consumed
        by ModelLoader._load_llama / EmbeddedProvider._ensure_llm) plus the HF
        download. Everything else is the real CLI → command → orchestrator →
        SpecAnalyzer → EmbeddedProvider → ModelLoader path.
        """
        monkeypatch.chdir(tmp_path)
        snap_dir = tmp_path / ".dbsprout"
        snap_dir.mkdir()
        (snap_dir / "schema.json").write_text(_schema().model_dump_json(indent=2), encoding="utf-8")
        adapter = tmp_path / "adapter.gguf"
        adapter.write_bytes(b"\x00")

        fake_llama = MagicMock(name="LlamaInstance")
        fake_llama.create_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tables": [{"table_name": "users", "columns": '
                            '{"id": {"provider": "builtin.autoincrement", '
                            '"params": {}}, "email": {"provider": '
                            '"mimesis.Person.email", "params": {}}}}], '
                            '"schema_hash": "x", "model_used": "lora"}'
                        )
                    }
                }
            ]
        }
        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = MagicMock(return_value=fake_llama)  # type: ignore[attr-defined]
        fake_module.LlamaGrammar = MagicMock()  # type: ignore[attr-defined]
        fake_module.LlamaGrammar.from_string.return_value = object()
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

        with patch(
            "dbsprout.spec.providers.embedded.EmbeddedProvider._download_model",
            return_value=adapter,
        ):
            result = runner.invoke(
                app,
                [
                    "generate",
                    "--engine",
                    "spec",
                    "--lora",
                    str(adapter),
                    "--rows",
                    "2",
                    "--output-dir",
                    str(tmp_path / "seeds"),
                ],
            )

        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "seeds").exists()

    def test_generate_lora_missing_path_cli_exit_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["generate", "--engine", "spec", "--lora", str(tmp_path / "nope.gguf")],
        )
        assert result.exit_code == 1


class TestLoraLoadTimeFallback:
    def test_loader_missing_adapter_falls_back_to_base_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ModelLoader.load with a non-existent adapter warns + returns base handle."""
        base = tmp_path / "base.gguf"
        base.write_bytes(b"\x00")
        missing_adapter = tmp_path / "missing.gguf"

        with patch.object(ModelLoader, "_load_llama") as m_load:
            m_load.return_value = MagicMock(return_value="HANDLE")
            with caplog.at_level(logging.WARNING):
                loaded = ModelLoader().load(base, lora_path=missing_adapter)

        assert loaded.lora_path is None
        assert loaded.handle == "HANDLE"
        assert "falling back to base model" in caplog.text
