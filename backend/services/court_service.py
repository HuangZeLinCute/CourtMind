# -*- coding: utf-8 -*-
"""Court detection service.

Wraps the existing Good-Badminton court detection chain:

    CourtKeyNet (cached, single load) -> original detector fallback -> manual

The CourtKeyNet model is loaded ONCE per process and reused across requests.
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from backend.core.paths import UPLOADS_DIR
from backend.services.video_service import get_video_path, save_template
from backend.services.analysis_pipeline import _scale_corners_to_video, imread_safe, prepare_court

# --------------------------------------------------------------------------- #
# CourtKeyNet single-instance cache
# --------------------------------------------------------------------------- #
_detector_lock = threading.Lock()
_detector_instance = None
_detector_error = None


def get_courtkeynet_detector():
    """Return the process-wide CourtKeyNetDetector (loaded exactly once)."""
    global _detector_instance, _detector_error
    if _detector_instance is None and _detector_error is None:
        with _detector_lock:
            if _detector_instance is None and _detector_error is None:
                try:
                    from badminton_analysis.court.courtkeynet_detector import (
                        CourtKeyNetDetector,
                    )

                    _detector_instance = CourtKeyNetDetector()
                except Exception as exc:  # noqa: BLE001 - surface as error, allow retry
                    _detector_error = str(exc)
    return _detector_instance


def reset_detector_cache():
    """Drop the cached detector (e.g. after a manual re-point via env)."""
    global _detector_instance, _detector_error
    with _detector_lock:
        _detector_instance = None
        _detector_error = None


# --------------------------------------------------------------------------- #
# Court state persistence (per uploaded video)
# --------------------------------------------------------------------------- #
def _state_path(video_id: str) -> Path:
    return UPLOADS_DIR / video_id / "court.json"


def load_court_state(video_id: str) -> dict:
    path = _state_path(video_id)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_court_state(video_id: str, state: dict) -> None:
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    (_state_path(video_id)).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _draw_preview(frame_bgr, corners):
    preview = frame_bgr.copy()
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(preview, [pts], True, (0, 255, 0), 3)
    for idx, (x, y) in enumerate(pts.tolist(), 1):
        cv2.circle(preview, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(idx),
            (int(x) + 14, int(y) - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    return preview


def _persist_detection(video_id: str, corners, roi_corners, mid_height,
                       best_frame, detector_name) -> dict:
    """Save best frame, preview and court state; return preview URL."""
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    best_frame_path = video_dir / "best_frame.jpg"
    cv2.imwrite(str(best_frame_path), best_frame)

    corners = [[int(x), int(y)] for x, y in corners]
    preview = _draw_preview(best_frame, corners)
    preview_path = video_dir / "court_preview.jpg"
    cv2.imwrite(str(preview_path), preview)

    state = {
        "corners": corners,
        "detector": detector_name or "courtkeynet",
        "roi_corners": roi_corners,
        "mid_height": mid_height,
        "best_frame_path": str(best_frame_path),
        "preview_url": f"/uploads/{video_id}/court_preview.jpg",
    }
    save_court_state(video_id, state)
    return state


# --------------------------------------------------------------------------- #
# Detection entry points
# --------------------------------------------------------------------------- #
def detect_court(video_id: str, use_legacy: bool = False, template_path=None) -> dict:
    """Run court detection for an uploaded video.

    Returns the persisted court state dict.
    """
    video_path = get_video_path(video_id)

    if use_legacy:
        if template_path is None:
            template_path = UPLOADS_DIR / video_id / "template.png"
        if not Path(template_path).is_file():
            raise FileNotFoundError("Legacy court template has not been uploaded.")
        result = prepare_court(str(template_path))
        if result["corners"] is None:
            return {"success": False, "detector": "legacy", "corners": None,
                    "message": "Legacy detector could not find the court."}
        # Scale template corners to video resolution for the rest of the flow.
        corners = _scale_corners_to_video(result["corners"], str(template_path), str(video_path))
        best_frame = _read_first_frame(video_path)
        if best_frame is None:
            return {"success": False, "message": "Cannot read video frames."}
        state = _persist_detection(
            video_id, corners, result["roi_corners"], result["mid_height"],
            best_frame, "legacy",
        )
        state["success"] = True
        state["detector"] = "legacy"
        return state

    # Default: CourtKeyNet (cached) -> original detector fallback
    from badminton_analysis.court.courtkeynet_detector import resolve_court_from_video

    detector = get_courtkeynet_detector()
    corners, roi_corners, mid_height, best_frame, detector_name = resolve_court_from_video(
        str(video_path), detector=detector
    )
    if corners is None or best_frame is None:
        return {
            "success": False,
            "detector": detector_name or "courtkeynet",
            "corners": None,
            "message": "Court detection failed (CourtKeyNet and fallback both failed).",
        }
    state = _persist_detection(video_id, corners, roi_corners, mid_height,
                               best_frame, detector_name)
    state["success"] = True
    return state


def update_corners(video_id: str, corners) -> dict:
    """Manually set court corners (drawn/edited by the user in the UI)."""
    video_path = get_video_path(video_id)
    prev = load_court_state(video_id)

    corners = [[int(x), int(y)] for x, y in corners]
    best_frame = None
    if prev.get("best_frame_path") and Path(prev["best_frame_path"]).is_file():
        best_frame = imread_safe(prev["best_frame_path"])
    if best_frame is None:
        best_frame = _read_first_frame(video_path)
    if best_frame is None:
        raise RuntimeError("Cannot read video frames.")

    from badminton_analysis.court.mapper import CourtMapper, compute_expanded_roi

    h, w = best_frame.shape[:2]
    roi_corners = compute_expanded_roi(corners, (h, w, 3))
    mid_height = CourtMapper(corners).mid_height

    state = _persist_detection(video_id, corners, roi_corners, mid_height,
                               best_frame, "manual")
    state["success"] = True
    return state


def upload_legacy_template(video_id: str, content: bytes) -> Path:
    return save_template(video_id, content)


def _read_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
