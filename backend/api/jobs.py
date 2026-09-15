# -*- coding: utf-8 -*-
"""Job endpoints: status polling + results + history."""
from fastapi import APIRouter, HTTPException

from backend.schemas.job import HistoryItem, JobResult, JobStatus
from backend.services import job_service, timeline_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


# NOTE: specific routes must be declared BEFORE /{job_id} so "history" is not
# captured as a job id.
@router.get("/history", response_model=list[HistoryItem])
def get_history():
    """Recent analyses: in-memory jobs first, then a scan of outputs/."""
    return [HistoryItem(**item) for item in job_service.history()]


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = job_service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        stage=job.get("stage"),
        message=job.get("message"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        finished_at=job.get("finished_at"),
    )


@router.get("/{job_id}/results", response_model=JobResult)
def get_job_results(job_id: str):
    result = job_service.job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobResult(**result)


@router.get("/{job_id}/timeline")
def get_job_timeline(job_id: str):
    try:
        return timeline_service.build_timeline(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
