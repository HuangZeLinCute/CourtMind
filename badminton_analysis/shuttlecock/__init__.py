"""Unified shuttlecock detector package.

* ``ShuttleDetector`` / ``ShuttleDetection`` — shared contract.
* ``TrackNetV3Detector`` — temporal TrackNetV3 + InpaintNet trajectory tracking.
* ``TrackNetV4Detector`` — TensorFlow TrackNetV4 motion-aware tracking.
* ``YoloShuttleDetector`` — existing per-frame YOLO detector (legacy mode).
"""

from .base import ShuttleDetection, ShuttleDetector
from .ensemble import fuse_detections
from .temporal_ensemble import TemporalEnsemble
from .tracknet_v3 import TrackNetV3Detector
from .tracknet_v4 import TrackNetV4Detector
from .yolo_detector import YoloShuttleDetector

__all__ = [
    "ShuttleDetection",
    "ShuttleDetector",
    "fuse_detections",
    "TemporalEnsemble",
    "TrackNetV3Detector",
    "TrackNetV4Detector",
    "YoloShuttleDetector",
]
