# -*- coding: utf-8 -*-
"""Court endpoints: detect court, save/manual corners, legacy template."""
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.schemas.court import CourtCorners, CourtState
from backend.schemas.video import CourtDetectionResponse
from backend.services import court_service, video_service

router = APIRouter(prefix="/videos/{video_id}", tags=["court"])


@router.post("/detect-court", response_model=CourtDetectionResponse)
def detect_court(video_id: str,
                 use_legacy: bool = Query(False, description="Use the legacy HSV/Canny/Hough detector"),
                 template: UploadFile | None = File(default=None)):
    """Auto-detect the four court corners from the video (CourtKeyNet first).

    ``use_legacy=true`` with an uploaded ``template`` image switches to the
    original HSV/Canny/Hough detector (Advanced calibration).
    """
    try:
        video_service.get_video_path(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    template_path = None
    if use_legacy:
        if template is None:
            raise HTTPException(status_code=400, detail="Legacy mode requires a template image.")
        content = template.file.read()
        template_path = court_service.upload_legacy_template(video_id, content)

    try:
        state = court_service.detect_court(
            video_id, use_legacy=use_legacy, template_path=template_path
        )
    except Exception as exc:  # noqa: BLE001 - user-safe message
        raise HTTPException(status_code=500, detail=f"Court detection failed: {exc}") from exc

    if not state.get("success"):
        return CourtDetectionResponse(
            success=False,
            detector=state.get("detector", "courtkeynet"),
            corners=None,
            message=state.get("message", "Court detection failed."),
        )
    return CourtDetectionResponse(
        success=True,
        detector=state.get("detector", "courtkeynet"),
        corners=state.get("corners"),
        roi_corners=state.get("roi_corners"),
        mid_height=state.get("mid_height"),
        preview_url=state.get("preview_url"),
        message="ok",
    )


@router.put("/court", response_model=CourtState)
def save_court(video_id: str, payload: CourtCorners):
    """Manually set the four corners (used by the corner editor)."""
    try:
        state = court_service.update_corners(video_id, payload.corners)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not save corners: {exc}") from exc
    return CourtState(**state)


@router.get("/court", response_model=CourtState)
def get_court(video_id: str):
    """Return the currently stored court state (after detection / manual edit)."""
    state = court_service.load_court_state(video_id)
    return CourtState(
        corners=state.get("corners"),
        detector=state.get("detector", "courtkeynet"),
        roi_corners=state.get("roi_corners"),
        mid_height=state.get("mid_height"),
        best_frame_path=state.get("best_frame_path"),
        preview_url=state.get("preview_url"),
    )
