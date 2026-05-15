"""Frozen Pydantic v2 models for the dbsprout model registry."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - runtime use by Pydantic field annotations

from pydantic import BaseModel, ConfigDict


class ModelEntry(BaseModel):
    """One entry in the bundled model registry manifest."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    repo: str
    filename: str
    size_bytes: int
    quantization: str
    parameters: str
    training: str = "base"
    license: str = ""
    default: bool = False


class InstalledModel(BaseModel):
    """A model discovered on disk under .dbsprout/models/."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: Path
    size_bytes: int
    kind: str
