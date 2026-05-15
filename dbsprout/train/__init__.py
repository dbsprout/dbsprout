"""Training data extraction (Component 9).

The public re-exports below are resolved **lazily** via :pep:`562`
``__getattr__`` so merely importing this package (or a lightweight submodule
such as ``dbsprout.train.config``, which the TOML config layer needs for the
``[train]`` section) does NOT eagerly pull the heavy training submodules
(``trainer``/``mlx_trainer``/``loader``/``exporter``) or ``rich.progress``.
Eagerly re-exporting them used to add ~150 ms to every CLI command and broke
the documented ``<500 ms`` startup budget. Attribute access
(``dbsprout.train.Exporter``) and ``from dbsprout.train import Exporter`` keep
working unchanged; only the import is deferred until first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbsprout.train.config import LoRAAdapter, TrainConfig
    from dbsprout.train.exporter import Exporter, ExportResult
    from dbsprout.train.loader import LoadedModel, ModelLoader
    from dbsprout.train.mlx_trainer import MLXTrainer, select_trainer
    from dbsprout.train.models import (
        ClosureReport,
        ExtractorConfig,
        NullPolicy,
        SampleAllocation,
        SampleManifest,
        SampleResult,
        SerializationResult,
        SerializerConfig,
        TableExtractionResult,
        TableSerializationResult,
    )
    from dbsprout.train.trainer import QLoRATrainer

# Public name -> submodule that defines it. Submodules are imported on first
# attribute access only (PEP 562), keeping heavy deps off the startup path.
_EXPORTS: dict[str, str] = {
    "ClosureReport": "dbsprout.train.models",
    "ExportResult": "dbsprout.train.exporter",
    "Exporter": "dbsprout.train.exporter",
    "ExtractorConfig": "dbsprout.train.models",
    "LoRAAdapter": "dbsprout.train.config",
    "LoadedModel": "dbsprout.train.loader",
    "MLXTrainer": "dbsprout.train.mlx_trainer",
    "ModelLoader": "dbsprout.train.loader",
    "NullPolicy": "dbsprout.train.models",
    "QLoRATrainer": "dbsprout.train.trainer",
    "SampleAllocation": "dbsprout.train.models",
    "SampleManifest": "dbsprout.train.models",
    "SampleResult": "dbsprout.train.models",
    "SerializationResult": "dbsprout.train.models",
    "SerializerConfig": "dbsprout.train.models",
    "TableExtractionResult": "dbsprout.train.models",
    "TableSerializationResult": "dbsprout.train.models",
    "TrainConfig": "dbsprout.train.config",
    "select_trainer": "dbsprout.train.mlx_trainer",
}

__all__ = [
    "ClosureReport",
    "ExportResult",
    "Exporter",
    "ExtractorConfig",
    "LoRAAdapter",
    "LoadedModel",
    "MLXTrainer",
    "ModelLoader",
    "NullPolicy",
    "QLoRATrainer",
    "SampleAllocation",
    "SampleManifest",
    "SampleResult",
    "SerializationResult",
    "SerializerConfig",
    "TableExtractionResult",
    "TableSerializationResult",
    "TrainConfig",
    "select_trainer",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve a public re-export from its defining submodule."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    import importlib  # noqa: PLC0415

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return __all__
