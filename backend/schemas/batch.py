# -*- coding: utf-8 -*-
"""Request/response models for sequential multi-video analysis."""

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.analysis import AnalysisOptions


class BatchAnalysisRequest(BaseModel):
    video_ids: list[str] = Field(..., min_length=1, max_length=50)
    options: Optional[AnalysisOptions] = None


class BatchJobStatus(BaseModel):
    job_id: str
    video_id: str
    filename: str
    status: str
    progress: int = 0
    current_frame: int = 0
    total_frames: int = 0
    stage: Optional[str] = None
    error: Optional[str] = None


class BatchStatus(BaseModel):
    batch_id: str
    status: str
    total: int
    finished: int = 0
    completed: int = 0
    failed: int = 0
    progress: int = 0
    current_index: int = 0
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    jobs: list[BatchJobStatus] = Field(default_factory=list)


class BatchStartResponse(BatchStatus):
    pass

