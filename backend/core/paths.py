# -*- coding: utf-8 -*-
"""Central path configuration for the Good-Badminton backend.

All paths are derived from the repository root so the app works regardless of
the current working directory (the FastAPI entrypoint also chdirs to root).
"""
from pathlib import Path

# backend/core/paths.py -> backend/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

COURTKEYNET_WEIGHTS = WEIGHTS_DIR / "courtkeynet_finetuned.safetensors"
BALL_MODEL = WEIGHTS_DIR / "yolo11s-ball.pt"
YOLO_POSE_MODEL = WEIGHTS_DIR / "yolo11n-pose.pt"
RTMPOSE_MODEL = WEIGHTS_DIR / "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"
RTMO_MODEL = WEIGHTS_DIR / "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx"

ANALYSIS_PIPELINE = PROJECT_ROOT / "backend" / "services" / "analysis_pipeline.py"


def ensure_dirs() -> None:
    for d in (UPLOADS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
