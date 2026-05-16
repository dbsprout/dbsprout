"""Quality gate: the documentation site must build with ``--strict``.

``--strict`` turns broken internal links, missing nav targets, and
unrecognized config keys into build failures, so this is a meaningful
gate for a docs scaffold. Skips cleanly when the optional ``[docs]``
extra (mkdocs) is not installed so the core suite stays green.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


@pytest.mark.integration
def test_mkdocs_strict_build(tmp_path: Path) -> None:
    """``mkdocs build --strict`` succeeds and emits index.html."""
    mkdocs_exe = shutil.which("mkdocs")
    if mkdocs_exe is None:
        pytest.skip("mkdocs not installed (pip install dbsprout[docs])")

    assert MKDOCS_YML.is_file(), f"missing {MKDOCS_YML}"

    site_dir = tmp_path / "site"
    result = subprocess.run(  # noqa: S603
        [
            mkdocs_exe,
            "build",
            "--strict",
            "--config-file",
            str(MKDOCS_YML),
            "--site-dir",
            str(site_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert (site_dir / "index.html").is_file(), "index.html not generated"
