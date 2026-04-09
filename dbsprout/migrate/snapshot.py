"""Schema snapshot storage for EPIC-006 Migration Awareness.

Provides an append-only, content-addressed store for ``DatabaseSchema``
snapshots.  Each snapshot is a JSON file with a wrapper envelope::

    {"metadata": {...}, "schema": {...}}

Files are named ``{ISO_timestamp}_{schema_hash[:8]}.json`` for
chronological ordering and hash-based lookup.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dbsprout.schema.models import DatabaseSchema

logger = logging.getLogger(__name__)


# ── Data models ─────────────────────────────────────────────────────────


class SnapshotInfo(BaseModel):
    """Lightweight descriptor returned by store operations."""

    model_config = ConfigDict(frozen=True)

    path: Path
    schema_hash: str
    timestamp: datetime
    table_count: int


class SnapshotMetadata(BaseModel):
    """Metadata envelope embedded in each snapshot JSON file."""

    model_config = ConfigDict(frozen=True)

    schema_hash: str
    timestamp: str  # ISO 8601
    table_count: int
    table_names: list[str]
    dialect: str | None


# ── Snapshot store ──────────────────────────────────────────────────────


class SnapshotStore:
    """Content-addressed, append-only snapshot store."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = base_dir if base_dir is not None else Path(".dbsprout") / "snapshots"

    # ── Public API ──────────────────────────────────────────────────

    def save(self, schema: DatabaseSchema) -> SnapshotInfo:
        """Persist a schema snapshot. Idempotent: same hash → same file.

        Raises ``ValueError`` for empty schemas (no tables).
        """
        if not schema.tables:
            msg = "Cannot snapshot an empty schema (no tables)."
            raise ValueError(msg)

        self.base_dir.mkdir(parents=True, exist_ok=True)

        schema_hash = schema.schema_hash()
        hash_prefix = schema_hash[:8]

        # Idempotency: return existing snapshot with matching hash
        existing = self._find_by_hash_prefix(hash_prefix)
        if existing is not None:
            return self._info_from_path(existing)

        now = datetime.now(timezone.utc)
        ts_str = now.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{ts_str}_{hash_prefix}.json"
        final_path = self.base_dir / filename
        tmp_path = self.base_dir / f"{filename}.tmp"

        metadata = SnapshotMetadata(
            schema_hash=schema_hash,
            timestamp=now.isoformat(),
            table_count=len(schema.tables),
            table_names=[t.name for t in schema.tables],
            dialect=schema.dialect,
        )
        payload = {
            "metadata": json.loads(metadata.model_dump_json()),
            "schema": json.loads(schema.model_dump_json()),
        }
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.rename(tmp_path, final_path)

        return SnapshotInfo(
            path=final_path,
            schema_hash=schema_hash,
            timestamp=now,
            table_count=len(schema.tables),
        )

    def list_snapshots(self) -> list[SnapshotInfo]:
        """Return all valid snapshots, newest-first by filename."""
        if not self.base_dir.exists():
            return []
        results: list[SnapshotInfo] = []
        for p in sorted(self.base_dir.glob("*.json"), reverse=True):
            if p.name.endswith(".tmp"):
                continue
            info = self._load_snapshot_info(p)
            if info is not None:
                results.append(info)
        return results

    def load_latest(self) -> DatabaseSchema | None:
        """Load the most recent schema snapshot, or ``None``."""
        for info in self.list_snapshots():
            schema = self._load_schema_from_path(info.path)
            if schema is not None:
                return schema
        return None

    def load_by_hash(self, hash_prefix: str) -> DatabaseSchema | None:
        """Load the schema whose hash starts with *hash_prefix*."""
        path = self._find_by_hash_prefix(hash_prefix)
        if path is None:
            return None
        return self._load_schema_from_path(path)

    def resolve(self, path_or_hash: str) -> DatabaseSchema | None:
        """Accept a file path or hash prefix and return the schema."""
        candidate = Path(path_or_hash)
        if candidate.exists():
            return self._load_schema_from_path(candidate)
        return self.load_by_hash(path_or_hash)

    # ── Private helpers ─────────────────────────────────────────────

    def _find_by_hash_prefix(self, prefix: str) -> Path | None:
        """Find a snapshot file whose hash suffix starts with *prefix*."""
        if not self.base_dir.exists():
            return None
        for p in self.base_dir.glob("*.json"):
            if p.name.endswith(".tmp"):
                continue
            stem = p.stem  # e.g. "20260408T120000Z_a1b2c3d4"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].startswith(prefix):
                return p
        return None

    def _info_from_path(self, path: Path) -> SnapshotInfo:
        """Build ``SnapshotInfo`` from an on-disk snapshot file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "metadata" in raw:
            meta = raw["metadata"]
            return SnapshotInfo(
                path=path,
                schema_hash=meta["schema_hash"],
                timestamp=datetime.fromisoformat(meta["timestamp"]),
                table_count=meta["table_count"],
            )
        # Legacy flat format: derive from schema content
        schema = DatabaseSchema.model_validate(raw)
        return SnapshotInfo(
            path=path,
            schema_hash=schema.schema_hash(),
            timestamp=datetime.now(timezone.utc),
            table_count=len(schema.tables),
        )

    def _load_snapshot_info(self, path: Path) -> SnapshotInfo | None:
        """Attempt to parse snapshot info; return ``None`` on failure."""
        try:
            return self._info_from_path(path)
        except Exception:
            logger.warning("Skipping corrupt snapshot file: %s", path.name)
            return None

    def _load_schema_from_path(self, path: Path) -> DatabaseSchema | None:
        """Load a ``DatabaseSchema`` from a snapshot file (wrapper or flat)."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            schema_data = raw.get("schema", raw)
            return DatabaseSchema.model_validate(schema_data)
        except Exception:
            logger.warning("Cannot load schema from snapshot: %s", path.name)
            return None
