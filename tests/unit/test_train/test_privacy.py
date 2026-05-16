"""Unit tests for dbsprout.train.privacy (S-070 training privacy)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import patch

import polars as pl
import pytest
from pydantic import ValidationError

from dbsprout.train.config import TrainConfig
from dbsprout.train.privacy import (
    RedactionStats,
    TableRedactionStats,
    TrainingRedactor,
    TrainPrivacyConfig,
    dp_sgd_guard,
)

# --- Task 2: TrainPrivacyConfig + TrainConfig wiring -----------------------


def test_train_privacy_config_defaults() -> None:
    cfg = TrainPrivacyConfig()
    assert cfg.pii_redaction is True
    assert cfg.dp_sgd is False
    assert cfg.pii_entities is None


def test_train_privacy_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        TrainPrivacyConfig(unknown=1)  # type: ignore[call-arg]


def test_train_config_has_privacy_default() -> None:
    tc = TrainConfig()
    assert isinstance(tc.privacy, TrainPrivacyConfig)
    assert tc.privacy.pii_redaction is True


def test_train_config_privacy_override() -> None:
    tc = TrainConfig(privacy=TrainPrivacyConfig(pii_redaction=False))
    assert tc.privacy.pii_redaction is False


# --- Task 3: RedactionStats models -----------------------------------------


def test_table_redaction_stats_model() -> None:
    s = TableRedactionStats(
        table="users", values_masked=3, entity_counts={"EMAIL_ADDRESS": 2, "PHONE_NUMBER": 1}
    )
    assert s.values_masked == 3
    assert s.entity_counts["EMAIL_ADDRESS"] == 2


def test_redaction_stats_aggregate() -> None:
    t = TableRedactionStats(table="users", values_masked=2, entity_counts={"EMAIL_ADDRESS": 2})
    stats = RedactionStats(
        tables=(t,),
        total_values_masked=2,
        entity_totals={"EMAIL_ADDRESS": 2},
        presidio_available=True,
    )
    assert stats.total_values_masked == 2
    assert stats.tables[0].table == "users"
    assert stats.presidio_available is True


def test_redaction_stats_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RedactionStats(unknown=1)  # type: ignore[call-arg]


# --- Task 4: dp_sgd_guard --------------------------------------------------


def test_dp_sgd_guard_noop_when_disabled() -> None:
    dp_sgd_guard(TrainPrivacyConfig(dp_sgd=False))  # must not raise


def test_dp_sgd_guard_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="DP-SGD"):
        dp_sgd_guard(TrainPrivacyConfig(dp_sgd=True))


# --- Task 5: TrainingRedactor fallback / disabled --------------------------


def test_redactor_fallback_when_presidio_missing(caplog: pytest.LogCaptureFixture) -> None:
    samples = {"users": pl.DataFrame({"id": [1], "email": ["a@b.com"]})}
    redactor = TrainingRedactor()
    with patch.dict(sys.modules, {"presidio_analyzer": None, "presidio_anonymizer": None}):
        out, stats = redactor.redact(samples, config=TrainPrivacyConfig())
    assert stats.presidio_available is False
    assert stats.total_values_masked == 0
    assert out["users"]["email"].to_list() == ["a@b.com"]  # untouched
    assert any("Presidio" in r.message for r in caplog.records)


def test_redactor_disabled_returns_input_unchanged() -> None:
    samples = {"users": pl.DataFrame({"email": ["a@b.com"]})}
    redactor = TrainingRedactor()
    out, stats = redactor.redact(samples, config=TrainPrivacyConfig(pii_redaction=False))
    assert out is samples
    assert stats.total_values_masked == 0
    assert stats.presidio_available is True


# --- Task 6: TrainingRedactor masking with a fake Presidio -----------------


@dataclass
class _FakeResult:
    entity_type: str
    start: int
    end: int


class _FakeAnalyzer:
    """Flags the literal substring ``a@b.com`` as an EMAIL_ADDRESS."""

    def analyze(self, *, text: str, entities: object, language: str) -> list[_FakeResult]:
        idx = text.find("a@b.com")
        if idx == -1:
            return []
        return [_FakeResult("EMAIL_ADDRESS", idx, idx + len("a@b.com"))]


@dataclass
class _FakeAnonymized:
    text: str


class _FakeAnonymizer:
    def anonymize(self, *, text: str, analyzer_results: list[_FakeResult]) -> _FakeAnonymized:
        out = text
        for r in analyzer_results:
            out = out.replace(text[r.start : r.end], f"<{r.entity_type}>")
        return _FakeAnonymized(out)


def test_redactor_masks_pii_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = {
        "users": pl.DataFrame(
            {"id": [1, 2], "email": ["a@b.com", "safe text"], "note": ["hi a@b.com", None]}
        )
    }
    redactor = TrainingRedactor()
    monkeypatch.setattr(
        TrainingRedactor, "_load_engines", lambda _self: (_FakeAnalyzer(), _FakeAnonymizer())
    )
    out, stats = redactor.redact(samples, config=TrainPrivacyConfig())

    assert stats.presidio_available is True
    assert stats.total_values_masked == 2  # users.email[0] and users.note[0]
    assert stats.entity_totals == {"EMAIL_ADDRESS": 2}
    emails = out["users"]["email"].to_list()
    assert emails == ["<EMAIL_ADDRESS>", "safe text"]
    notes = out["users"]["note"].to_list()
    assert notes == ["hi <EMAIL_ADDRESS>", None]
    assert out["users"]["id"].to_list() == [1, 2]
    assert samples["users"]["email"].to_list() == ["a@b.com", "safe text"]


def test_redactor_passes_pii_entities_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _SpyAnalyzer:
        def analyze(self, *, text: str, entities: object, language: str) -> list[_FakeResult]:
            seen["entities"] = entities
            return []

    samples = {"t": pl.DataFrame({"c": ["x"]})}
    redactor = TrainingRedactor()
    monkeypatch.setattr(
        TrainingRedactor, "_load_engines", lambda _self: (_SpyAnalyzer(), _FakeAnonymizer())
    )
    _out, stats = redactor.redact(
        samples, config=TrainPrivacyConfig(pii_entities=("EMAIL_ADDRESS",))
    )
    assert seen["entities"] == ["EMAIL_ADDRESS"]
    assert stats.total_values_masked == 0


def test_redactor_table_with_only_non_string_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = {"nums": pl.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})}
    redactor = TrainingRedactor()
    monkeypatch.setattr(
        TrainingRedactor, "_load_engines", lambda _self: (_FakeAnalyzer(), _FakeAnonymizer())
    )
    out, stats = redactor.redact(samples, config=TrainPrivacyConfig())
    assert out["nums"]["a"].to_list() == [1, 2]
    assert stats.total_values_masked == 0
    assert stats.tables[0].table == "nums"
