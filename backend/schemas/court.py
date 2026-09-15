# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional


class CourtCorners(BaseModel):
    """Four corner points in ORIGINAL video pixel coordinates: TL, TR, BR, BL."""

    corners: list[list[int]] = Field(..., min_length=4, max_length=4)


class CourtState(BaseModel):
    corners: Optional[list[list[int]]] = None
    detector: str = "courtkeynet"
    roi_corners: Optional[list[list[int]]] = None
    mid_height: Optional[float] = None
    best_frame_path: Optional[str] = None
    preview_url: Optional[str] = None
