# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import Optional


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    progress: int = 0
    current_frame: int = 0
    total_frames: int = 0
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    finished_at: Optional[float] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    output_dir: Optional[str] = None
    video_url: Optional[str] = None
    metadata_url: Optional[str] = None
    detections_url: Optional[str] = None
    report_url: Optional[str] = None
    visualizations: list[str] = []
    metadata: Optional[dict] = None
    report: Optional[dict] = None
    error: Optional[str] = None


class HistoryItem(BaseModel):
    job_id: str
    name: str
    status: str
    created_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_sec: Optional[float] = None
    video_url: Optional[str] = None
    metadata_url: Optional[str] = None
    error: Optional[str] = None


class SystemStatus(BaseModel):
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    courtkeynet_weights: str
    courtkeynet_ready: bool
    ball_model: str
    ball_model_ready: bool
    yolo_pose_model: str
    yolo_pose_model_ready: bool
    rtmpose_model_ready: bool
    rtmo_model_ready: bool
    tracknet_tracker_weights: str = ""
    tracknet_tracker_ready: bool = False
    tracknet_rectifier_weights: str = ""
    tracknet_rectifier_ready: bool = False
    tracknet_v4_weights: str = ""
    tracknet_v4_weights_ready: bool = False
    tracknet_v4_dependency_ready: bool = False
