# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    fps: float
    width: int
    height: int
    duration: float
    file_size_bytes: int
    url: Optional[str] = None


class VideoList(BaseModel):
    videos: list[VideoInfo]


class CourtCorners(BaseModel):
    """Four corner points in ORIGINAL video pixel coordinates: TL, TR, BR, BL."""

    corners: list[list[int]] = Field(..., min_length=4, max_length=4)


class CourtDetectionRequest(BaseModel):
    use_legacy: bool = False
    template_path: Optional[str] = None  # resolved server-side, not a client path


class CourtDetectionResponse(BaseModel):
    success: bool
    detector: str = "courtkeynet"
    corners: Optional[list[list[int]]] = None
    roi_corners: Optional[list[list[int]]] = None
    mid_height: Optional[float] = None
    preview_url: Optional[str] = None
    message: Optional[str] = None
