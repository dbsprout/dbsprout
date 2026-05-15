"""Model registry loading, on-disk discovery, and resumable download."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from dbsprout.models.types import ModelEntry

_REGISTRY_PATH = Path(__file__).parent / "registry.json"
_ENTRY_LIST = TypeAdapter(list[ModelEntry])


def load_registry(path: Path | None = None) -> list[ModelEntry]:
    """Load and validate the bundled (or given) model registry manifest."""
    manifest = path if path is not None else _REGISTRY_PATH
    try:
        raw = manifest.read_text(encoding="utf-8")
        return _ENTRY_LIST.validate_json(raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        msg = f"invalid model registry at {manifest}: {exc}"
        raise ValueError(msg) from exc


class ModelManager:
    """Discovers installed models and downloads registry models.

    Fleshed out in Task 3.
    """

    def __init__(self, root: Path | str = ".dbsprout/models") -> None:
        self._root = Path(root)
