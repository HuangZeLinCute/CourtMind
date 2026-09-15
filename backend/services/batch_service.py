# -*- coding: utf-8 -*-
"""Sequential batch scheduler for memory-heavy video analysis.

Every video remains a normal job.  A batch only coordinates those jobs and
runs them one at a time so TrackNetV4 + TrackNetV3 + YOLO11 are never loaded by
multiple batch workers concurrently.
"""

import threading
import time
import traceback
import uuid

from backend.services import analysis_service, court_service, job_service, video_service

_lock = threading.Lock()
_batches: dict[str, dict] = {}
_TERMINAL = {"completed", "failed"}


def _new_batch_id() -> str:
    return f"batch_{uuid.uuid4().hex[:12]}"


def create(video_ids: list[str], options: dict) -> dict:
    """Create all child jobs first, then start one sequential daemon worker."""
    batch_id = _new_batch_id()
    entries = []
    for video_id in video_ids:
        info = video_service.get_info(video_id)
        job_id = job_service.create(
            video_id=video_id,
            filename=info["filename"],
            options=options,
        )
        entries.append({
            "job_id": job_id,
            "video_id": video_id,
            "filename": info["filename"],
        })

    with _lock:
        _batches[batch_id] = {
            "batch_id": batch_id,
            "status": "queued",
            "entries": entries,
            "options": dict(options),
            "current_index": 0,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }

    thread = threading.Thread(
        target=_run,
        args=(batch_id,),
        daemon=True,
        name=f"analysis-{batch_id}",
    )
    thread.start()
    return get(batch_id)


def _run(batch_id: str) -> None:
    with _lock:
        batch = _batches[batch_id]
        batch["status"] = "processing"
        batch["started_at"] = time.time()
        entries = list(batch["entries"])
        options = dict(batch["options"])

    for index, entry in enumerate(entries, 1):
        with _lock:
            _batches[batch_id]["current_index"] = index

        job_id = entry["job_id"]
        video_id = entry["video_id"]
        try:
            state = court_service.load_court_state(video_id)
            corners = state.get("corners")
            if not corners:
                job_service.set_stage(job_id, "Court detection")
                state = court_service.detect_court(video_id)
                if not state.get("success") or not state.get("corners"):
                    raise RuntimeError(
                        state.get("message") or "Automatic court detection failed."
                    )
                corners = state["corners"]

            analysis_service.run_analysis_job(
                job_id=job_id,
                video_id=video_id,
                corners=corners,
                options=options,
            )
        except Exception as exc:  # court/preparation failure before pipeline
            job_service.fail(job_id, exc, traceback.format_exc())

    with _lock:
        batch = _batches.get(batch_id)
        if batch is not None:
            batch["status"] = "finished"
            batch["finished_at"] = time.time()


def get(batch_id: str) -> dict | None:
    with _lock:
        raw = _batches.get(batch_id)
        if raw is None:
            return None
        batch = dict(raw)
        entries = [dict(item) for item in raw["entries"]]

    jobs = []
    completed = 0
    failed = 0
    progress_total = 0
    for entry in entries:
        job = job_service.get(entry["job_id"]) or {}
        status = job.get("status", "queued")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        progress = 100 if status in _TERMINAL else int(job.get("progress", 0))
        progress_total += progress
        jobs.append({
            **entry,
            "status": status,
            "progress": int(job.get("progress", 0)),
            "current_frame": int(job.get("current_frame", 0)),
            "total_frames": int(job.get("total_frames", 0)),
            "stage": job.get("stage"),
            "error": job.get("error"),
        })

    total = len(entries)
    finished = completed + failed
    if finished == total:
        status = "completed_with_errors" if failed else "completed"
    else:
        status = batch["status"]

    return {
        "batch_id": batch_id,
        "status": status,
        "total": total,
        "finished": finished,
        "completed": completed,
        "failed": failed,
        "progress": round(progress_total / total) if total else 0,
        "current_index": int(batch.get("current_index", 0)),
        "created_at": batch.get("created_at"),
        "started_at": batch.get("started_at"),
        "finished_at": batch.get("finished_at"),
        "jobs": jobs,
    }

