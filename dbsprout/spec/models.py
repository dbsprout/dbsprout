"""Spec generation models — generator mappings and related types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeneratorMapping(BaseModel):
    """Maps a column to a specific data generator with confidence scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generator_name: str
    provider: str  # "mimesis", "faker", "numpy", "builtin"
    confidence: float = Field(ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
