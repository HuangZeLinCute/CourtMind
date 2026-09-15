# -*- coding: utf-8 -*-
"""In-memory job store with thread-safe updates + outputs/ history scan.

No database is used (per project constraints): jobs live in a dict guarded by a
lock. Completed analyses are additionally discoverable by scanning ``outputs/``
so history survives a backend restart.
"""
import json
import threading
import time
import uuid
from pathlib import Path

from backend.core.paths import OUTPUTS_DIR

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------------------- #
def _clean_name(name: str | None, fallback: str) -> str:
    cleaned = " ".join((name or "").strip().split())[:120]
    return cleaned or Path(fallback).stem or fallback


def create(video_id: str, filename: str, options: dict, name: str | None = None) -> str:
    job_id = _new_job_id()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "video_id": video_id,
            "filename": filename,
            "name": _clean_name(name, filename),
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "stage": "Queued",
            "message": None,
            "error": None,
            "options": options,
            "created_at": time.time(),
            "finished_at": None,
            "result": None,
        }
    return job_id


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return dict(job)
    # Not in memory (e.g. after a backend restart): fall back to scanning the
    # output directory.
    return _scan_output_job(job_id)


def _update(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def set_stage(job_id: str, stage: str) -> None:
    _update(job_id, stage=stage)


def update_progress(job_id: str, frame_count: int, total_frames: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["current_frame"] = frame_count
        job["total_frames"] = total_frames
        if total_frames > 0:
            job["progress"] = int(round(min(99.0, 100.0 * frame_count / total_frames)))
        job["status"] = "processing"


def complete(job_id: str, result: dict) -> None:
    _update(job_id, status="completed", progress=100, finished_at=time.time(),
            stage="Completed", result=result)


def fail(job_id: str, exc: Exception, tb: str = "") -> None:
    # Keep a concise user-safe message; the full traceback goes to the log.
    import logging

    logging.getLogger("goodbadminton.backend").error(
        "Job %s failed:\n%s", job_id, tb or exc
    )
    _update(job_id, status="failed", finished_at=time.time(), stage="Failed",
            error=str(exc))


# --------------------------------------------------------------------------- #
# Output-directory scanning (restart-surviving history)
# --------------------------------------------------------------------------- #
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _scan_output_job(job_id: str) -> dict | None:
    """Build a completed job dict from ``outputs/{job_id}`` if it exists.

    Used so that analyses produced before a backend restart remain queryable
    through the same job endpoints.
    """
    d = OUTPUTS_DIR / job_id
    meta_path = d / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    video = meta.get("video", {})
    total = int(video.get("total_frames", 0))

    # Prefer the final annotated video (web_detect_*) over the intermediate one.
    video_file = None
    for candidate in sorted(d.iterdir()):
        if candidate.suffix.lower() in _VIDEO_SUFFIXES:
            if candidate.name.startswith("web_detect_") or video_file is None:
                video_file = candidate

    # Collect visualizations recursively (position_visualizations/, detect_images/…).
    visualizations: list[str] = []
    for p in sorted(d.rglob("*")):
        if "chat_frames" not in p.parts and p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            visualizations.append(str(p))

    detections_path = d / "detections.jsonl"
    report_path = d / "match_report.json"
    return {
        "job_id": job_id,
        "video_id": None,
        "filename": video.get("name", job_id),
        "name": _clean_name(
            (meta.get("analysis") or {}).get("name"), video.get("name", job_id)
        ),
        "status": "completed",
        "progress": 100,
        "current_frame": total,
        "total_frames": total,
        "stage": "Completed",
        "message": None,
        "error": None,
        "duration_sec": video.get("duration_sec"),
        "created_at": meta_path.stat().st_mtime,
        "finished_at": meta_path.stat().st_mtime,
        "options": {},
        "result": {
            "output_dir": str(d),
            "video": str(video_file) if video_file else None,
            "metadata": str(meta_path),
            "detections": str(detections_path) if detections_path.is_file() else None,
            "report": str(report_path) if report_path.is_file() else None,
            "visualizations": visualizations,
        },
    }


# --------------------------------------------------------------------------- #
# Result URL mapping
# --------------------------------------------------------------------------- #
def _to_url(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(OUTPUTS_DIR)
            return f"/outputs/{p.as_posix()}"
        except ValueError:
            return f"/{p.as_posix()}"
    return f"/{p.as_posix()}"


def job_result(job_id: str) -> dict | None:
    job = get(job_id)
    if not job:
        return None
    result = job.get("result") or {}
    if job["status"] == "completed" and result:
        metadata = None
        if result.get("metadata") and Path(result["metadata"]).is_file():
            try:
                metadata = json.loads(
                    Path(result["metadata"]).read_text(encoding="utf-8")
                )
            except Exception:
                metadata = None
        report = None
        if result.get("report") and Path(result["report"]).is_file():
            try:
                report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            except Exception:
                report = None
        visualizations = [
            _to_url(v) for v in result.get("visualizations", [])
            if Path(v).is_file()
        ]
        return {
            "job_id": job_id,
            "status": job["status"],
            "output_dir": result.get("output_dir"),
            "video_url": _to_url(result["video"]) if result.get("video") and Path(result["video"]).is_file() else None,
            "metadata_url": _to_url(result["metadata"]) if result.get("metadata") and Path(result["metadata"]).is_file() else None,
            "detections_url": _to_url(result["detections"]) if result.get("detections") and Path(result["detections"]).is_file() else None,
            "report_url": _to_url(result["report"]) if result.get("report") and Path(result["report"]).is_file() else None,
            "visualizations": visualizations,
            "metadata": metadata,
            "report": report,
            "error": None,
        }
    return {
        "job_id": job_id,
        "status": job["status"],
        "output_dir": None,
        "video_url": None,
        "metadata_url": None,
        "detections_url": None,
        "report_url": None,
        "visualizations": [],
        "metadata": None,
        "report": None,
        "error": job.get("error"),
    }


# --------------------------------------------------------------------------- #
# History (in-memory jobs + scan of outputs/)
# --------------------------------------------------------------------------- #
def history() -> list[dict]:
    items = []
    with _lock:
        for job in _jobs.values():
            items.append({
                "job_id": job["job_id"],
                "name": job.get("name") or _clean_name(None, job["filename"]),
                "status": job["status"],
                "created_at": job["created_at"],
                "finished_at": job["finished_at"],
                "error": job.get("error"),
            })
    # Scan outputs/ for directories containing metadata.json
    if OUTPUTS_DIR.is_dir():
        in_memory = {item["job_id"] for item in items}
        for d in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            job_id = d.name
            if job_id in in_memory:
                continue
            scanned = _scan_output_job(job_id)
            if scanned is None:
                continue
            result = scanned.get("result") or {}
            video_file = result.get("video")
            items.append({
                "job_id": job_id,
                "name": scanned.get("name") or scanned["filename"],
                "status": "completed",
                "created_at": scanned["created_at"],
                "finished_at": scanned["finished_at"],
                "duration_sec": scanned.get("duration_sec"),
                "video_url": _to_url(video_file) if video_file and Path(video_file).is_file() else None,
                "metadata_url": f"/outputs/{job_id}/metadata.json",
                "error": None,
            })
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return items
