"""YOLO single-frame shuttlecock detector (existing Good-Badminton model)."""

import os
from typing import List, Optional

from .base import ShuttleDetection, ShuttleDetector


class YoloShuttleDetector(ShuttleDetector):
    """Wraps the existing ultralytics YOLO ball model.

    Kept deliberately thin: it loads the same ``yolo11s-ball.pt`` checkpoint
    the original pipeline uses and applies the same per-frame inference, so
    the legacy behaviour is preserved exactly.
    """

    name = "yolo"

    def __init__(self, model_path: str = "weights/yolo11s-ball.pt", conf: float = 0.18):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO shuttlecock model not found: {model_path}")
        from ..models.yolo11 import load_yolo11

        self.model_path = model_path
        self.conf = float(conf)
        self._model = load_yolo11(model_path)
        self._device = 0 if self._model.device.type == "cuda" else "cpu"

    def process_video(self, video_path: str, progress_cb=None) -> List[ShuttleDetection]:
        raise NotImplementedError(
            "YOLO is frame-local; it is driven per-frame by the existing "
            "ShuttlecockTracker.detect_ball flow, not as a whole-video batch."
        )

    def close(self) -> None:
        pass
