# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import Literal, Optional


class AnalysisOptions(BaseModel):
    language: str = "zh"
    pose_family: str = "yolo-pose"
    pose_mode: str = "balanced"
    yolo_pose_model: Optional[str] = None
    ball_model: Optional[str] = None
    shuttle_model: Optional[
        Literal["yolo", "tracknet", "tracknet_v4", "ensemble"]
    ] = None
    audio: bool = True
    show_skeletons: bool = True
    show_player_trajectories: bool = True
    show_court_trajectory: bool = True
    show_shuttlecock_trajectory: bool = True
    show_player_stats: bool = False
    show_pose_roi: bool = False
    visualize_positions: bool = True


class AnalysisRequest(BaseModel):
    """Start an analysis job for an uploaded video.

    ``corners`` is optional: when omitted the stored court state (from
    detect-court / PUT court) is used. ``template_path`` is optional and only
    used for the legacy template flow; the client never sends raw paths, the
    server resolves uploaded templates by id.
    """

    corners: Optional[list[list[int]]] = None
    options: Optional[AnalysisOptions] = None
    legacy_template_id: Optional[str] = None
    name: Optional[str] = None


class AnalysisStartResponse(BaseModel):
    job_id: str
    status: str = "queued"
