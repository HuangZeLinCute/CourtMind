# -*- coding: utf-8 -*-
"""Application-level configuration (mostly derived from paths.py)."""
import os
from pathlib import Path

from .paths import PROJECT_ROOT

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
APP_NAME = "Good-Badminton API"
API_PREFIX = "/api"
HOST = os.environ.get("GB_HOST", "127.0.0.1")
PORT = int(os.environ.get("GB_PORT", "8000"))

CORS_ORIGINS = os.environ.get(
    "GB_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
).split(",")

# --------------------------------------------------------------------------- #
# Job system
# --------------------------------------------------------------------------- #
JOB_POLL_INTERVAL_SEC = 1.0

# --------------------------------------------------------------------------- #
# TrackNetV3 (temporal shuttlecock tracking)
# --------------------------------------------------------------------------- #
TRACKNET_TRACKER_PATH = os.environ.get(
    "GB_TRACKNET_TRACKER", str(PROJECT_ROOT / "weights" / "tracknet" / "tracknet_v3_tracker.pt")
)
TRACKNET_RECTIFIER_PATH = os.environ.get(
    "GB_TRACKNET_RECTIFIER", str(PROJECT_ROOT / "weights" / "tracknet" / "tracknet_v3_rectifier.pt")
)

# TrackNetV4 Type-B (TensorFlow/Keras, motion-aware fusion)
TRACKNET_V4_WEIGHTS_PATH = os.environ.get(
    "GB_TRACKNET_V4_WEIGHTS",
    str(PROJECT_ROOT / "weights" / "tracknet_v4" / "tracknet_v4_type_b.keras"),
)

# --------------------------------------------------------------------------- #
# Runtime defaults for the headless pipeline and CLI
# --------------------------------------------------------------------------- #
DEFAULT_OPTIONS = {
    "language": "zh",
    "pose_family": "yolo-pose",
    "pose_mode": "balanced",
    "yolo_pose_model": "weights/yolo11n-pose.pt",
    "ball_model": "weights/yolo11s-ball.pt",
    "shuttle_model": "yolo",
    "audio": True,
    "show_skeletons": True,
    "show_player_trajectories": True,
    "show_court_trajectory": True,
    "show_shuttlecock_trajectory": True,
    "show_player_stats": False,
    "show_pose_roi": False,
    "visualize_positions": True,
}

# Option keys the API accepts from clients (whitelist to avoid surprises).
OPTION_KEYS = set(DEFAULT_OPTIONS.keys())

os.environ.setdefault("PYTHONUTF8", "1")
