"""Model registry loading, on-disk discovery, and resumable download."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from dbsprout.models.types import InstalledModel, ModelEntry

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
    """Discovers installed models and resolves registry entries.

    Layout (reuses S-025 EmbeddedProvider conventions):
      <root>/base/<filename>      base models (registry filename verbatim)
      <root>/custom/<name>.gguf   fine-tuned models
    """

    def __init__(self, root: Path | str = ".dbsprout/models") -> None:
        self._root = Path(root)

    @property
    def base_dir(self) -> Path:
        return self._root / "base"

    @property
    def custom_dir(self) -> Path:
        return self._root / "custom"

    def registry(self) -> list[ModelEntry]:
        return load_registry()

    def resolve_entry(self, name: str) -> ModelEntry:
        for entry in self.registry():
            if entry.name == name:
                return entry
        msg = f"unknown model {name!r}"
        raise KeyError(msg)

    def install_path(self, entry: ModelEntry) -> Path:
        return self.base_dir / entry.filename

    def is_installed(self, entry: ModelEntry) -> bool:
        return self.install_path(entry).is_file()

    def list_installed(self) -> list[InstalledModel]:
        found: list[InstalledModel] = []
        for kind, directory in (("base", self.base_dir), ("custom", self.custom_dir)):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.gguf")):
                name = path.name if kind == "base" else path.stem
                found.append(
                    InstalledModel(
                        name=name,
                        path=path,
                        size_bytes=path.stat().st_size,
                        kind=kind,
                    )
                )
        return found
