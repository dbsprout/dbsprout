"""Lazy-import contract for dbsprout.config (S-064 review finding #1).

`import dbsprout.config` must NOT eagerly pull the heavy ``dbsprout.train``
training submodules (``trainer``/``mlx_trainer``/``loader``/``exporter``) nor
``rich.progress``. That regression added ~150 ms to *every* CLI command,
breaking the documented ``<500 ms`` startup budget.

A subprocess with a pristine interpreter is used so the assertion is immune to
test-ordering pollution (other tests legitimately import the train package).
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import dbsprout.config  # noqa: F401
heavy = [
    m
    for m in (
        "dbsprout.train.trainer",
        "dbsprout.train.mlx_trainer",
        "dbsprout.train.loader",
        "dbsprout.train.exporter",
        "rich.progress",
    )
    if m in sys.modules
]
print(",".join(heavy))
"""


def test_import_config_does_not_pull_heavy_train_modules() -> None:
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted interpreter
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert leaked == [], (
        f"import dbsprout.config eagerly pulled heavy modules: {leaked}. "
        "TrainConfig must be imported under TYPE_CHECKING with a string "
        "forward-ref so the <500ms CLI startup budget holds."
    )


def test_train_config_still_usable_from_dbsprout_config() -> None:
    # The lazy-import wiring must not break runtime construction/validation.
    from dbsprout.config.models import DBSproutConfig  # noqa: PLC0415
    from dbsprout.train.config import TrainConfig  # noqa: PLC0415

    cfg = DBSproutConfig()
    assert isinstance(cfg.train, TrainConfig)
    assert cfg.train == TrainConfig()
