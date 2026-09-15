# -*- coding: utf-8 -*-
"""Video upload / metadata / listing service."""
import shutil
import uuid
from pathlib import Path

import cv2

from backend.core.paths import UPLOADS_DIR, ensure_dirs


def _new_video_id() -> str:
    return uuid.uuid4().hex[:12]


def _read_video_meta(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open uploaded video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "total_frames": total,
        "duration": round(total / fps, 3) if fps > 0 else 0.0,
    }


def save_upload(filename: str, content: bytes) -> dict:
    """Persist an uploaded video and return its info dict."""
    ensure_dirs()
    video_id = _new_video_id()
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name or "input.mp4"
    video_path = video_dir / "input.mp4"
    video_path.write_bytes(content)

    meta = _read_video_meta(video_path)
    return {
        "video_id": video_id,
        "filename": safe_name,
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "duration": meta["duration"],
        "file_size_bytes": video_path.stat().st_size,
        "url": f"/uploads/{video_id}/input.mp4",
    }


def get_video_path(video_id: str) -> Path:
    path = UPLOADS_DIR / video_id / "input.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_id}")
    return path


def get_info(video_id: str) -> dict:
    """Return the metadata dict for an uploaded video."""
    video_path = get_video_path(video_id)
    meta = _read_video_meta(video_path)
    return {
        "video_id": video_id,
        "filename": video_path.name,
        "fps": meta["fps"],
        "width": meta["width"],
        "height": meta["height"],
        "duration": meta["duration"],
        "file_size_bytes": video_path.stat().st_size,
        "url": f"/uploads/{video_id}/input.mp4",
    }


def list_uploads() -> list[dict]:
    """List uploaded videos with their metadata."""
    ensure_dirs()
    items = []
    for d in sorted(UPLOADS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        video_path = d / "input.mp4"
        if not video_path.is_file():
            continue
        try:
            meta = _read_video_meta(video_path)
        except Exception:
            continue
        items.append({
            "video_id": d.name,
            "filename": video_path.name,
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "duration": meta["duration"],
            "file_size_bytes": video_path.stat().st_size,
            "url": f"/uploads/{d.name}/input.mp4",
        })
    return items


def save_template(video_id: str, content: bytes) -> Path:
    """Persist a legacy court template tied to a video id."""
    video_dir = UPLOADS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    template_path = video_dir / "template.png"
    template_path.write_bytes(content)
    return template_path


def copy_to(video_id: str, dest: Path) -> None:
    """Copy the uploaded video to a destination (used by the analysis job)."""
    src = get_video_path(video_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
