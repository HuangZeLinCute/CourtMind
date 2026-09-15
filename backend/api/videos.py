# -*- coding: utf-8 -*-
"""Video endpoints: upload, info, list."""
from fastapi import APIRouter, HTTPException, UploadFile

from backend.schemas.video import VideoInfo, VideoList
from backend.services import video_service

router = APIRouter(prefix="/videos", tags=["videos"])

_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


@router.post("", response_model=VideoInfo, status_code=201)
async def upload_video(file: UploadFile):
    """Upload a match video. Returns video metadata + URL."""
    content = await file.read()
    if len(content) > _MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds 2 GB limit.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        return video_service.save_upload(file.filename or "input.mp4", content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot read video: {exc}") from exc


@router.get("", response_model=VideoList)
def list_videos():
    return VideoList(videos=video_service.list_uploads())


@router.get("/{video_id}", response_model=VideoInfo)
def get_video(video_id: str):
    try:
        return video_service.get_info(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
