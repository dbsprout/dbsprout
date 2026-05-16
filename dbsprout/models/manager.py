"""Model registry loading, on-disk discovery, and resumable download."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from dbsprout.models.types import InstalledModel, ModelEntry

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

# Generous ceiling: large GGUF downloads can run for many minutes on slow
# links, but an unbounded socket must not hang the CLI forever.
_DOWNLOAD_TIMEOUT_S = 600.0

_REGISTRY_PATH = Path(__file__).parent / "registry.json"
_ENTRY_LIST = TypeAdapter(list[ModelEntry])

_HTTP_PARTIAL_CONTENT = 206


class DownloadError(RuntimeError):
    """Raised when a model download fails (network or HTTP error)."""


def load_registry(path: Path | None = None) -> list[ModelEntry]:
    """Load and validate the bundled (or given) model registry manifest."""
    manifest = path if path is not None else _REGISTRY_PATH
    try:
        raw = manifest.read_text(encoding="utf-8")
        return _ENTRY_LIST.validate_json(raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        msg = f"invalid model registry at {manifest}: {exc}"
        raise ValueError(msg) from exc


def _resolve_hf_url(repo: str, filename: str) -> str:
    """Build the HuggingFace `resolve` URL for a repo file (single mock seam)."""
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


class ModelManager:
    """Discovers installed models and resolves/downloads registry entries.

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

    def download(
        self,
        entry: ModelEntry,
        *,
        client: httpx.Client | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        force: bool = False,
    ) -> Path:
        """Download ``entry`` to ``<base>/<filename>``, resuming a .part if any.

        Skips (returns the existing path) when already installed and not
        ``force``. Raises :class:`DownloadError` on any network/HTTP failure;
        the partial ``.part`` file is kept so a re-run can resume.
        """
        dest = self.install_path(entry)

        # Defense-in-depth: even though ModelEntry validates ``filename``,
        # assert the resolved destination stays inside base_dir before any
        # write (path-traversal / SSRF latent fix).
        base = self.base_dir.resolve()
        if not dest.resolve().is_relative_to(base):
            msg = f"refusing to write outside {base}: {dest}"
            raise DownloadError(msg)

        if dest.is_file() and not force:
            return dest

        import httpx  # noqa: PLC0415 - keep httpx off the <500ms CLI startup path

        owns_client = client is None
        # ``follow_redirects=True``: HuggingFace ``resolve`` returns a 302 to a
        # CDN; without this the standalone manager client fails the download.
        http = (
            client
            if client is not None
            else httpx.Client(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True)
        )
        try:
            return self._stream_to_part(entry, dest, http, progress_cb)
        except (httpx.HTTPError, httpx.StreamError) as exc:
            msg = (
                f"download of {entry.name!r} failed: {exc}. "
                f"Re-run `dbsprout models download {entry.name}` to resume."
            )
            raise DownloadError(msg) from exc
        except OSError as exc:
            # Disk full / permission / I/O error while writing the .part.
            msg = (
                f"download of {entry.name!r} failed writing to disk: {exc}. "
                f"Re-run `dbsprout models download {entry.name}` to resume."
            )
            raise DownloadError(msg) from exc
        finally:
            if owns_client:
                http.close()

    def _stream_to_part(
        self,
        entry: ModelEntry,
        dest: Path,
        http: httpx.Client,
        progress_cb: Callable[[int, int], None] | None,
    ) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        existing = part.stat().st_size if part.is_file() else 0
        url = _resolve_hf_url(entry.repo, entry.filename)
        headers = {"Range": f"bytes={existing}-"} if existing else {}

        with http.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            resuming = response.status_code == _HTTP_PARTIAL_CONTENT and existing > 0
            mode = "ab" if resuming else "wb"
            downloaded = existing if resuming else 0
            remaining = response.headers.get("Content-Length")
            try:
                total = (downloaded + int(remaining)) if remaining else 0
            except ValueError as exc:
                msg = f"server sent an invalid Content-Length {remaining!r} for {entry.name!r}"
                raise DownloadError(msg) from exc
            # Hash the full content while writing so the integrity check costs
            # no extra read. A resumed (.part) download cannot be hash-verified
            # (the prefix bytes were not streamed this run), so verification is
            # skipped when resuming — the .part is kept for a fresh retry.
            hasher = hashlib.sha256() if (entry.sha256 and not resuming) else None
            with part.open(mode) as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
                    if hasher is not None:
                        hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        progress_cb(downloaded, total or downloaded)

        if hasher is not None:
            actual = hasher.hexdigest()
            if actual != entry.sha256:
                part.unlink(missing_ok=True)
                msg = (
                    f"sha256 mismatch for {entry.name!r}: expected "
                    f"{entry.sha256}, got {actual}. The corrupt partial file "
                    "was removed; re-run the download."
                )
                raise DownloadError(msg)

        os.replace(part, dest)  # atomic finalize of the completed download
        return dest
