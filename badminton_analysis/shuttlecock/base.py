"""Shuttlecock detector abstraction.

A detector returns, for one video, a per-frame trajectory that downstream
Good-Badminton code can consume frame by frame:

    trajectory[i] -> {"x": float, "y": float, "visible": bool, "confidence": float | None}

where ``i`` is the **0-based video frame index** (cap.read() order).  Coordinates
are always in original video pixels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ShuttleDetection:
    """One frame of shuttlecock position."""

    x: float
    y: float
    visible: bool
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "visible": bool(self.visible),
            "confidence": None if self.confidence is None else float(self.confidence),
        }


class ShuttleDetector:
    """Base class. Subclasses load their model once and produce trajectories."""

    name: str = "base"

    def process_video(self, video_path: str, progress_cb=None) -> List[ShuttleDetection]:
        raise NotImplementedError

    def close(self) -> None:
        pass
