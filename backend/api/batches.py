# -*- coding: utf-8 -*-
"""Batch-analysis endpoints."""

from fastapi import APIRouter, HTTPException

from backend.schemas.batch import BatchAnalysisRequest, BatchStartResponse, BatchStatus
from backend.services import analysis_service, batch_service, video_service

router = APIRouter(prefix="/batches", tags=["batches"])


@router.post("", response_model=BatchStartResponse, status_code=202)
def start_batch(payload: BatchAnalysisRequest):
    """Queue up to 50 uploaded videos for sequential analysis."""
    # Preserve input order but do not process the same upload twice.
    video_ids = list(dict.fromkeys(payload.video_ids))
    missing = []
    for video_id in video_ids:
        try:
            video_service.get_video_path(video_id)
        except FileNotFoundError:
            missing.append(video_id)
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Uploaded video(s) not found: {', '.join(missing)}",
        )

    options = analysis_service.normalize_options(
        payload.options.model_dump() if payload.options else {}
    )
    return BatchStartResponse(**batch_service.create(video_ids, options))


@router.get("/{batch_id}", response_model=BatchStatus)
def get_batch(batch_id: str):
    batch = batch_service.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    return BatchStatus(**batch)

