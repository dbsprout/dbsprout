"""Hatchling build hook: build the React/Vite Workbench into the wheel.

Runs ``npm ci && npm run build`` in ``frontend/`` so the built SPA lands in
``dbsprout/web/spa/`` and is bundled into the wheel (see the ``artifacts`` entry
in ``[tool.hatch.build.targets.wheel]``).

Best-effort and idempotent:
* skips when ``frontend/`` or ``npm`` is absent (a contributor's editable
  ``uv sync`` without Node) — the app then serves the placeholder;
* skips when a build already exists, unless ``DBSPROUT_FORCE_FRONTEND_BUILD`` is
  set, so repeated installs do not re-run ``npm``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version, build_data  # signature fixed by hatchling; both unused here
        root = Path(self.root)
        frontend = root / "frontend"
        built = root / "dbsprout" / "web" / "spa" / "index.html"
        npm = shutil.which("npm")

        if not frontend.is_dir() or npm is None:
            self.app.display_warning(
                "skip frontend build (frontend/ or npm missing); serving placeholder"
            )
            return
        if built.is_file() and not os.environ.get("DBSPROUT_FORCE_FRONTEND_BUILD"):
            self.app.display_info("frontend build present; skipping rebuild")
            return

        self.app.display_info("building frontend (npm ci && npm run build)")
        subprocess.run([npm, "ci"], cwd=frontend, check=True)  # noqa: S603
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)  # noqa: S603
