"""YOLO11 construction kept behind a small, lazy project-local API.

The trained shuttlecock checkpoint belongs to this project.  The generic
YOLO11 layer/parser implementation remains supplied by the pinned
``ultralytics`` dependency instead of maintaining a divergent private fork.
"""

from pathlib import Path
from typing import Any


def load_yolo11(weights_path: str) -> Any:
    """Load a YOLO11 checkpoint after validating its project-local path."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO11 checkpoint not found: {path}")

    from ultralytics import YOLO

    return YOLO(str(path))

