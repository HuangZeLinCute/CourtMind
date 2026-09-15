# -*- coding: utf-8 -*-
"""Analysis endpoints: start an analysis job."""
from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import AnalysisRequest, AnalysisStartResponse
from backend.services import analysis_service, job_service, video_service

router = APIRouter(prefix="/videos/{video_id}", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisStartResponse, status_code=202)
def start_analysis(video_id: str, payload: AnalysisRequest):
    """Start an analysis job. Returns immediately with a job_id."""
    try:
        video_service.get_video_path(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    options = analysis_service.normalize_options(
        payload.options.model_dump() if payload.options else {}
    )
    job_id = job_service.create(
        video_id=video_id,
        filename=video_service.get_info(video_id)["filename"],
        options=options,
        name=payload.name,
    )
    analysis_service.spawn_job(
        job_id=job_id,
        video_id=video_id,
        corners=payload.corners,
        options=options,
        legacy_template_id=payload.legacy_template_id,
        analysis_name=payload.name,
    )
    return AnalysisStartResponse(job_id=job_id, status="queued")
