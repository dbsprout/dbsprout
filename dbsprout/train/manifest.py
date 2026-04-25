"""Manifest JSON read/write."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

from dbsprout.train.models import SampleManifest

if TYPE_CHECKING:
    from pathlib import Path

_SUPPORTED_MANIFEST_VERSION: Final[int] = 1


def write_manifest(path: Path, manifest: SampleManifest) -> None:
    """Serialize *manifest* to *path* as pretty-printed JSON.

    Creates the parent directory if it does not yet exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest(path: Path) -> SampleManifest:
    """Read a manifest JSON file from *path* and validate its schema version.

    Raises :class:`ValueError` if the on-disk ``manifest_version`` is newer
    than this build of dbsprout knows how to read.
    """
    raw = json.loads(path.read_text())
    version = raw.get("manifest_version", 1)
    if version > _SUPPORTED_MANIFEST_VERSION:
        raise ValueError(
            f"manifest_version {version} is newer than supported "
            f"({_SUPPORTED_MANIFEST_VERSION}); upgrade dbsprout to read {path}."
        )
    return SampleManifest.model_validate(raw)
