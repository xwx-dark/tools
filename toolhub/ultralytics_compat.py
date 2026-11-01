"""Compatibility helpers for loading custom Ultralytics checkpoints."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys


_PARTIAL_LABEL_MODULE = "ultralytics.models.yolo.detect.partial_label_training"


def register_partial_label_training_module() -> None:
    """Make the workspace partial-label implementation importable by its checkpoint path."""
    if _PARTIAL_LABEL_MODULE in sys.modules:
        return

    source = Path(__file__).resolve().parent.parent / "partial_label_training.py"
    if not source.is_file():
        return

    detect_package = importlib.import_module("ultralytics.models.yolo.detect")
    if importlib.util.find_spec(_PARTIAL_LABEL_MODULE) is not None:
        return

    spec = importlib.util.spec_from_file_location(_PARTIAL_LABEL_MODULE, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load partial-label model code from '{source}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_PARTIAL_LABEL_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[_PARTIAL_LABEL_MODULE]
        raise
    setattr(detect_package, "partial_label_training", module)
