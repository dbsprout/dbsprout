"""Frozen Pydantic v2 models for the dbsprout model registry."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

# A HuggingFace repo id is exactly ``owner/name``; both parts are restricted
# to the chars HF itself allows. Rejects traversal / injection / extra slashes
# before the value is ever interpolated into a download URL (SSRF latent fix).
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


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
    # Optional expected SHA-256 of the GGUF. Bundled entries leave this null
    # (no behavior change); when set, the download is integrity-checked.
    sha256: str | None = None

    @field_validator("filename")
    @classmethod
    def _filename_must_be_a_plain_name(cls, value: str) -> str:
        """Reject path separators / traversal / absolute paths in *filename*.

        The filename is joined onto ``<base>/`` to form the install path; a
        crafted value (``../``, ``/abs``, ``a/b``) could escape it
        (path-traversal). Only a single portable path component is allowed.
        """
        if value in {"", ".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
            msg = (
                f"invalid model filename {value!r}: must be a single path "
                "component (no '/', '\\', '..' or absolute paths)."
            )
            raise ValueError(msg)
        return value

    @field_validator("repo")
    @classmethod
    def _repo_must_be_owner_slash_name(cls, value: str) -> str:
        """Constrain *repo* to a HuggingFace ``owner/name`` id.

        Also rejects ``.``/``..`` path components so a crafted repo cannot
        traverse out of the HF URL host (SSRF latent fix).
        """
        if not _REPO_RE.match(value) or any(part in {".", ".."} for part in value.split("/")):
            msg = (
                f"invalid model repo {value!r}: expected a HuggingFace "
                "'owner/name' id (chars [A-Za-z0-9._-], exactly one '/', "
                "no '.'/'..' components)."
            )
            raise ValueError(msg)
        return value


class InstalledModel(BaseModel):
    """A model discovered on disk under .dbsprout/models/."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: Path
    size_bytes: int
    kind: str
