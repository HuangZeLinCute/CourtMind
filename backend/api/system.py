# -*- coding: utf-8 -*-
"""System endpoints: runtime status (GPU / models / device)."""
from pathlib import Path

from fastapi import APIRouter

from backend.core.config import (
    TRACKNET_RECTIFIER_PATH,
    TRACKNET_TRACKER_PATH,
    TRACKNET_V4_WEIGHTS_PATH,
)
from backend.core.paths import (
    BALL_MODEL,
    COURTKEYNET_WEIGHTS,
    RTMO_MODEL,
    RTMPOSE_MODEL,
    YOLO_POSE_MODEL,
)
from backend.schemas.job import SystemStatus

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
def system_status():
    """Report runtime status without loading any model."""
    gpu_available = False
    gpu_name = None
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    device = "cuda" if gpu_available else "cpu"
    try:
        import importlib.util

        tracknet_v4_dependency_ready = importlib.util.find_spec("keras") is not None
    except Exception:
        tracknet_v4_dependency_ready = False
    return SystemStatus(
        device=device,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        courtkeynet_weights=str(COURTKEYNET_WEIGHTS),
        courtkeynet_ready=COURTKEYNET_WEIGHTS.is_file(),
        ball_model=str(BALL_MODEL),
        ball_model_ready=BALL_MODEL.is_file(),
        yolo_pose_model=str(YOLO_POSE_MODEL),
        yolo_pose_model_ready=YOLO_POSE_MODEL.is_file(),
        rtmpose_model_ready=RTMPOSE_MODEL.is_file(),
        rtmo_model_ready=RTMO_MODEL.is_file(),
        tracknet_tracker_weights=str(TRACKNET_TRACKER_PATH),
        tracknet_tracker_ready=Path(TRACKNET_TRACKER_PATH).is_file(),
        tracknet_rectifier_weights=str(TRACKNET_RECTIFIER_PATH),
        tracknet_rectifier_ready=Path(TRACKNET_RECTIFIER_PATH).is_file(),
        tracknet_v4_weights=str(TRACKNET_V4_WEIGHTS_PATH),
        tracknet_v4_weights_ready=Path(TRACKNET_V4_WEIGHTS_PATH).is_file(),
        tracknet_v4_dependency_ready=tracknet_v4_dependency_ready,
    )
