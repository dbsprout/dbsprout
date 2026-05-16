"""Training-time privacy safeguards (S-070).

Four defense-in-depth layers protect a fine-tuned adapter from memorizing
sensitive data:

1. **LoRA only** -- S-064 trains a small adapter, not full weights.
2. **Completion-only loss** -- S-064; the GReaT corpus (S-063) is a single
   ``text`` field with no prompt to memorize *by construction*, and the
   trainer forces ``completion_only_loss=True``.
3. **PII redaction (Presidio)** -- this module: detect and mask PII *values*
   in the sampled rows before GReaT serialization. Default on.
4. **DP-SGD (Opacus)** -- opt-in formal guarantee. This module ships the
   config knob and a clear not-yet-wired guard (:func:`dp_sgd_guard`); full
   Opacus integration into the training backend is a follow-up.

``presidio-analyzer`` / ``presidio-anonymizer`` are optional (the
``[privacy]`` extra) and imported lazily inside
:meth:`TrainingRedactor.redact`; when either is missing the redactor logs a
clear warning and passes rows through unchanged rather than crashing the
pipeline on a core-only install.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

_DP_SGD_HINT = (
    "DP-SGD (differential privacy) was requested ([train.privacy] dp_sgd = "
    "true) but is not yet wired into the training backend. Disable it "
    "([train.privacy] dp_sgd = false) or rely on the other three privacy "
    "layers (LoRA-only, completion-only loss, PII redaction). Full Opacus "
    "DP-SGD integration is tracked as a follow-up."
)


class TrainPrivacyConfig(BaseModel):
    """The ``[train.privacy]`` TOML section.

    ``pii_redaction`` defaults ``True`` (mask PII values before
    serialization). ``dp_sgd`` is opt-in and currently guarded -- see
    :func:`dp_sgd_guard`. ``pii_entities`` overrides the Presidio default
    entity set when provided (``None`` = Presidio defaults).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pii_redaction: bool = True
    dp_sgd: bool = False
    pii_entities: tuple[str, ...] | None = None


class TableRedactionStats(BaseModel):
    """Per-table PII-redaction outcome.

    ``values_masked`` counts cells in which at least one PII span was
    replaced; ``entity_counts`` maps the Presidio entity type to the number
    of spans replaced for it in this table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    values_masked: int = Field(default=0, ge=0)
    entity_counts: dict[str, int] = Field(default_factory=dict)


class RedactionStats(BaseModel):
    """Aggregate PII-redaction outcome across all tables.

    ``presidio_available`` is ``False`` when Presidio was not installed and
    redaction was skipped (the pipeline logs a warning in that case and
    leaves the rows untouched).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tables: tuple[TableRedactionStats, ...] = ()
    total_values_masked: int = Field(default=0, ge=0)
    entity_totals: dict[str, int] = Field(default_factory=dict)
    presidio_available: bool = True


def dp_sgd_guard(config: TrainPrivacyConfig) -> None:
    """Raise a clear error if DP-SGD is requested but unsupported.

    A no-op when ``config.dp_sgd`` is ``False``. Called by the training
    pipeline before the (non-DP) backend runs so the user gets an actionable
    message instead of a silently-ignored privacy setting.
    """
    if config.dp_sgd:
        raise RuntimeError(_DP_SGD_HINT)


class TrainingRedactor:
    """Mask PII *values* in sampled rows before GReaT serialization.

    Operates immutably on a ``table -> polars.DataFrame`` map: builds new
    DataFrames, never mutates inputs. ``presidio-analyzer`` /
    ``presidio-anonymizer`` are imported lazily; when either is missing the
    redactor logs a clear warning and returns the samples unchanged with
    ``presidio_available=False`` so the pipeline never crashes on a
    core-only install.
    """

    def redact(
        self,
        samples: dict[str, pl.DataFrame],
        *,
        config: TrainPrivacyConfig,
    ) -> tuple[dict[str, pl.DataFrame], RedactionStats]:
        """Return ``(redacted_samples, stats)``.

        When ``config.pii_redaction`` is ``False`` the input is returned
        as-is. When Presidio is unavailable the input is returned unchanged
        with ``presidio_available=False`` and a warning is logged.
        """
        if not config.pii_redaction:
            return samples, RedactionStats()

        engines = self._load_engines()
        if engines is None:
            logger.warning(
                "Presidio not installed -- skipping PII redaction. Install "
                "'dbsprout[privacy]' to enable it, or set [train.privacy] "
                "pii_redaction = false to silence this warning."
            )
            return samples, RedactionStats(presidio_available=False)

        analyzer, anonymizer = engines
        entities = list(config.pii_entities) if config.pii_entities else None
        new_samples: dict[str, pl.DataFrame] = {}
        table_stats: list[TableRedactionStats] = []
        for table, df in sorted(samples.items()):
            new_df, ts = self._redact_table(
                table, df, analyzer=analyzer, anonymizer=anonymizer, entities=entities
            )
            new_samples[table] = new_df
            table_stats.append(ts)

        entity_totals: dict[str, int] = {}
        for ts in table_stats:
            for ent, n in ts.entity_counts.items():
                entity_totals[ent] = entity_totals.get(ent, 0) + n
        return new_samples, RedactionStats(
            tables=tuple(table_stats),
            total_values_masked=sum(ts.values_masked for ts in table_stats),
            entity_totals=entity_totals,
            presidio_available=True,
        )

    def _load_engines(self) -> tuple[Any, Any] | None:
        """Lazily build the Presidio analyzer + anonymizer, or ``None``."""
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
            from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415
        except ImportError:
            return None
        return AnalyzerEngine(), AnonymizerEngine()

    def _redact_table(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        analyzer: Any,
        anonymizer: Any,
        entities: list[str] | None,
    ) -> tuple[pl.DataFrame, TableRedactionStats]:
        """Redact every string column of one DataFrame (immutably)."""
        import polars as pl  # noqa: PLC0415 - lazy: keep polars off CLI startup

        values_masked = 0
        entity_counts: dict[str, int] = {}
        new_columns: dict[str, list[Any]] = {}
        for name in df.columns:
            series = df[name]
            if series.dtype != pl.String:
                new_columns[name] = series.to_list()
                continue
            redacted_cells: list[Any] = []
            for cell in series.to_list():
                if cell is None:
                    redacted_cells.append(None)
                    continue
                new_text, hits = self._redact_cell(
                    str(cell), analyzer=analyzer, anonymizer=anonymizer, entities=entities
                )
                if hits:
                    values_masked += 1
                    for ent, n in hits.items():
                        entity_counts[ent] = entity_counts.get(ent, 0) + n
                redacted_cells.append(new_text)
            new_columns[name] = redacted_cells
        new_df = pl.DataFrame(new_columns)
        return new_df, TableRedactionStats(
            table=table, values_masked=values_masked, entity_counts=entity_counts
        )

    def _redact_cell(
        self,
        text: str,
        *,
        analyzer: Any,
        anonymizer: Any,
        entities: list[str] | None,
    ) -> tuple[str, dict[str, int]]:
        """Analyze + anonymize one cell; return ``(new_text, entity_hits)``."""
        results = analyzer.analyze(text=text, entities=entities, language="en")
        if not results:
            return text, {}
        hits: dict[str, int] = {}
        for r in results:
            hits[r.entity_type] = hits.get(r.entity_type, 0) + 1
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return str(anonymized.text), hits
